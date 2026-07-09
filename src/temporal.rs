//! H.261 temporal reference and picture-rate handling — §3.1 + §4.2.1.2 of
//! ITU-T Rec. H.261 (03/93).
//!
//! # Temporal reference (§4.2.1.2)
//!
//! The 5-bit `TR` field in the picture header (§4.2.1.2) is "formed by
//! incrementing its value in the previously transmitted picture header by one
//! plus the number of non-transmitted pictures (at 29.97 Hz) since that last
//! transmitted one. The arithmetic is performed with only the five LSBs." So
//! between two consecutive *transmitted* pictures the field advances by
//!
//! ```text
//! TR_cur = (TR_prev + 1 + non_transmitted) mod 32
//! ```
//!
//! and the observed mod-32 delta is `1 + non_transmitted` (at least `1`). This
//! module exposes the delta primitives ([`tr_delta`] / [`non_transmitted`]) and
//! a cumulative unwrapping tracker ([`TrTracker`]) that turns a stream of 5-bit
//! `TR` fields back into a monotonic timeline of source-picture periods — the
//! basis for presentation timing and for advancing the §4.3.1 freeze-picture
//! timeout by the true number of elapsed picture intervals rather than one per
//! decoded picture.
//!
//! # Picture-rate restriction (§3.1)
//!
//! §3.1: "Means shall be provided to restrict the maximum picture rate of
//! encoders by having at least 0, 1, 2 or 3 non-transmitted pictures between
//! transmitted ones. Selection of this minimum number … shall be by external
//! means." [`PictureRate`] models that choice as the *interval* (in
//! source-picture periods, `1 + non_transmitted`) between transmitted pictures,
//! and yields the [`PictureRate::tr_increment`] the encoder stamps into
//! successive `TR` fields. The four §3.1-highlighted rates are `interval`
//! 1..=4 (≈ 29.97 / 14.99 / 9.99 / 7.49 Hz), but any interval 1..=32 is
//! representable.

/// Source-picture rate numerator (§3.1 / §3.2.2 note): pictures occur
/// `30000/1001` times per second at the full rate.
pub const SOURCE_RATE_NUM: u32 = 30_000;
/// Denominator of [`SOURCE_RATE_NUM`].
pub const SOURCE_RATE_DEN: u32 = 1001;

/// The §4.2.1.2 temporal-reference delta between two consecutive *transmitted*
/// pictures, i.e. `1 + non_transmitted` (the number of source-picture periods
/// that elapsed).
///
/// Both arguments are the raw 5-bit `TR` field (only the low five bits are
/// significant; higher bits are ignored). The result is in `1..=32`:
///
/// * a normal increment (0 non-transmitted pictures) yields `1`;
/// * one/two/three non-transmitted pictures yield `2`/`3`/`4`; and
/// * a raw mod-32 difference of `0` is interpreted as `32` — a valid
///   consecutive transmitted picture always advances `TR` by at least one, so
///   the field can only return to its previous value after a full 32-period
///   wrap. `32` is the smallest (and most conservative) elapsed count
///   consistent with an unchanged field.
///
/// The arithmetic mirrors §4.2.1.2's "performed with only the five LSBs".
#[inline]
pub fn tr_delta(prev: u8, cur: u8) -> u32 {
    let prev = (prev & 0x1F) as u32;
    let cur = (cur & 0x1F) as u32;
    let raw = (cur + 32 - prev) % 32;
    if raw == 0 {
        32
    } else {
        raw
    }
}

/// The number of *non-transmitted* pictures (§3.1 / §4.2.1.2) between two
/// consecutive transmitted pictures — [`tr_delta`] minus one, so `0..=31`.
#[inline]
pub fn non_transmitted(prev: u8, cur: u8) -> u32 {
    tr_delta(prev, cur) - 1
}

/// The §3.1 picture-rate restriction, expressed as the interval — in
/// source-picture periods — between transmitted pictures.
///
/// An `interval` of `1` is the full 29.97 Hz rate (0 non-transmitted pictures);
/// `2`/`3`/`4` are the ≈ 14.99 / 9.99 / 7.49 Hz rates §3.1 highlights (1/2/3
/// non-transmitted pictures). Any interval `1..=32` is representable, matching
/// the 5-bit `TR` field's per-step range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PictureRate {
    interval: u8,
}

impl Default for PictureRate {
    /// The full 29.97 Hz rate (interval 1, no non-transmitted pictures).
    fn default() -> Self {
        Self::FULL
    }
}

impl PictureRate {
    /// The full source rate: one transmitted picture every source-picture
    /// period (interval 1, `TR` increment 1). This is the encoder default and
    /// keeps a coded sequence byte-identical to one that never modelled a rate
    /// restriction.
    pub const FULL: PictureRate = PictureRate { interval: 1 };

    /// Build a rate from the §3.1 minimum number of non-transmitted pictures
    /// between transmitted ones. `0` is the full rate; `1`/`2`/`3` are the
    /// §3.1-highlighted reduced rates. Values above `31` saturate at `31`
    /// (interval 32 — the largest step the 5-bit `TR` field can carry).
    pub fn from_non_transmitted(n: u8) -> Self {
        let n = n.min(31);
        Self { interval: n + 1 }
    }

    /// Build a rate directly from the interval (source-picture periods between
    /// transmitted pictures). Clamped to `1..=32`.
    pub fn from_interval(interval: u8) -> Self {
        Self {
            interval: interval.clamp(1, 32),
        }
    }

    /// The interval in source-picture periods (`1..=32`).
    pub fn interval(self) -> u8 {
        self.interval
    }

    /// The number of non-transmitted pictures per transmitted picture
    /// (`interval - 1`, `0..=31`).
    pub fn non_transmitted(self) -> u8 {
        self.interval - 1
    }

    /// The amount to add to the `TR` field (mod 32) for each successive
    /// transmitted picture at this rate — equal to [`Self::interval`] (§4.2.1.2:
    /// `1 + non_transmitted`).
    pub fn tr_increment(self) -> u8 {
        self.interval
    }

    /// The nominal transmitted-picture rate as a rational `(num, den)` Hz:
    /// `30000 / (1001 * interval)`.
    pub fn nominal_hz(self) -> (u32, u32) {
        (SOURCE_RATE_NUM, SOURCE_RATE_DEN * self.interval as u32)
    }
}

/// Cumulative temporal-reference tracker (§4.2.1.2).
///
/// Feed each decoded picture's 5-bit `TR` field to [`TrTracker::observe`]; the
/// tracker unwraps the mod-32 arithmetic into a monotonic
/// [`TrTracker::presentation_index`] measured in source-picture periods (the
/// first transmitted picture is index 0). It also reports the per-step delta
/// (`1 + non_transmitted`) so a caller can drive presentation timing, detect
/// dropped pictures, or advance the §4.3.1 freeze timeout by the true elapsed
/// interval rather than a fixed one-per-picture tick.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrTracker {
    prev: Option<u8>,
    presentation_index: u64,
    last_delta: u32,
}

impl TrTracker {
    /// A fresh tracker with no picture observed yet.
    pub fn new() -> Self {
        Self {
            prev: None,
            presentation_index: 0,
            last_delta: 0,
        }
    }

    /// Reset to the initial state (e.g. on a decoder `reset`).
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Consume a decoded picture's 5-bit `TR` field.
    ///
    /// For the first picture, seeds the tracker and returns `None` (there is no
    /// previous transmitted picture to measure against). For every subsequent
    /// picture, returns the §4.2.1.2 delta (`1 + non_transmitted`, `1..=32`) and
    /// advances [`Self::presentation_index`] by that many source-picture
    /// periods.
    pub fn observe(&mut self, tr: u8) -> Option<u32> {
        let tr = tr & 0x1F;
        match self.prev {
            None => {
                self.prev = Some(tr);
                self.last_delta = 0;
                None
            }
            Some(prev) => {
                let delta = tr_delta(prev, tr);
                self.presentation_index += delta as u64;
                self.prev = Some(tr);
                self.last_delta = delta;
                Some(delta)
            }
        }
    }

    /// The monotonic presentation index of the most recently observed picture,
    /// in source-picture periods since the first transmitted picture (index 0).
    pub fn presentation_index(&self) -> u64 {
        self.presentation_index
    }

    /// The §4.2.1.2 delta of the most recent [`Self::observe`] step
    /// (`1 + non_transmitted`), or `0` before a second picture is observed.
    pub fn last_delta(&self) -> u32 {
        self.last_delta
    }

    /// The number of non-transmitted pictures immediately before the most
    /// recently observed picture (`last_delta - 1`), or `0` before a second
    /// picture is observed.
    pub fn last_non_transmitted(&self) -> u32 {
        self.last_delta.saturating_sub(1)
    }

    /// The raw 5-bit `TR` field of the most recently observed picture, or `None`
    /// before any picture is observed.
    pub fn last_tr(&self) -> Option<u8> {
        self.prev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tr_delta_normal_increment_is_one() {
        for prev in 0u8..32 {
            let cur = (prev + 1) & 0x1F;
            assert_eq!(tr_delta(prev, cur), 1, "prev={prev} cur={cur}");
            assert_eq!(non_transmitted(prev, cur), 0);
        }
    }

    #[test]
    fn tr_delta_counts_non_transmitted() {
        // §4.2.1.2: delta = 1 + non_transmitted.
        assert_eq!(tr_delta(0, 2), 2);
        assert_eq!(non_transmitted(0, 2), 1);
        assert_eq!(tr_delta(30, 1), 3); // wraps: 30 -> 31 -> 0 -> 1
        assert_eq!(non_transmitted(30, 1), 2);
        assert_eq!(tr_delta(29, 1), 4);
    }

    #[test]
    fn tr_delta_wrap_to_same_value_is_full_cycle() {
        // A repeated field can only mean a full 32-period wrap; report 32.
        for tr in 0u8..32 {
            assert_eq!(tr_delta(tr, tr), 32);
        }
    }

    #[test]
    fn tr_delta_ignores_high_bits() {
        // Only the low five bits are significant (§4.2.1.2).
        assert_eq!(tr_delta(0b1110_0001, 0b0000_0010), 1);
    }

    #[test]
    fn picture_rate_full_is_default() {
        assert_eq!(PictureRate::default(), PictureRate::FULL);
        assert_eq!(PictureRate::FULL.interval(), 1);
        assert_eq!(PictureRate::FULL.tr_increment(), 1);
        assert_eq!(PictureRate::FULL.non_transmitted(), 0);
        assert_eq!(PictureRate::FULL.nominal_hz(), (30_000, 1001));
    }

    #[test]
    fn picture_rate_from_non_transmitted_matches_spec_rates() {
        // §3.1: at least 0, 1, 2 or 3 non-transmitted pictures.
        let r0 = PictureRate::from_non_transmitted(0);
        let r1 = PictureRate::from_non_transmitted(1);
        let r2 = PictureRate::from_non_transmitted(2);
        let r3 = PictureRate::from_non_transmitted(3);
        assert_eq!(r0.tr_increment(), 1);
        assert_eq!(r1.tr_increment(), 2);
        assert_eq!(r2.tr_increment(), 3);
        assert_eq!(r3.tr_increment(), 4);
        assert_eq!(r1.nominal_hz(), (30_000, 2002));
        assert_eq!(r3.nominal_hz(), (30_000, 4004));
    }

    #[test]
    fn picture_rate_saturates() {
        assert_eq!(PictureRate::from_non_transmitted(255).interval(), 32);
        assert_eq!(PictureRate::from_interval(0).interval(), 1);
        assert_eq!(PictureRate::from_interval(200).interval(), 32);
    }

    #[test]
    fn tracker_first_picture_seeds_without_delta() {
        let mut t = TrTracker::new();
        assert_eq!(t.observe(5), None);
        assert_eq!(t.presentation_index(), 0);
        assert_eq!(t.last_tr(), Some(5));
        assert_eq!(t.last_delta(), 0);
    }

    #[test]
    fn tracker_accumulates_presentation_index() {
        let mut t = TrTracker::new();
        t.observe(0);
        assert_eq!(t.observe(1), Some(1));
        assert_eq!(t.presentation_index(), 1);
        assert_eq!(t.observe(3), Some(2)); // skipped one
        assert_eq!(t.presentation_index(), 3);
        assert_eq!(t.last_non_transmitted(), 1);
        assert_eq!(t.observe(4), Some(1));
        assert_eq!(t.presentation_index(), 4);
    }

    #[test]
    fn tracker_unwraps_across_mod32_boundary() {
        let mut t = TrTracker::new();
        t.observe(30);
        t.observe(31); // +1 -> index 1
        t.observe(1); // 31 -> 0 -> 1 => +2 -> index 3
        assert_eq!(t.presentation_index(), 3);
        assert_eq!(t.last_delta(), 2);
    }

    #[test]
    fn tracker_reset_clears_state() {
        let mut t = TrTracker::new();
        t.observe(4);
        t.observe(8);
        assert!(t.presentation_index() > 0);
        t.reset();
        assert_eq!(t.presentation_index(), 0);
        assert_eq!(t.last_tr(), None);
    }

    #[test]
    fn tracker_matches_full_rate_encoder_sequence() {
        // A full-rate encoder stamps TR 0,1,2,... so each step is +1 and the
        // presentation index equals the transmitted-picture count.
        let mut t = TrTracker::new();
        for i in 0u32..100 {
            t.observe((i & 0x1F) as u8);
        }
        assert_eq!(t.presentation_index(), 99);
    }
}
