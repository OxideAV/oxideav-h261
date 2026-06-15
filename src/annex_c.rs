//! H.261 §5.3 + Annex C — Codec delay measurement method.
//!
//! Annex C "forms an integral part of this Recommendation" (like the
//! Annex A IDCT-accuracy procedure): it specifies a *measurable*
//! procedure by which a particular codec design's video encoder and
//! decoder delays are established, so that audio-delay compensation can
//! be fixed for lip-sync when H.261 is part of a conversational service
//! (§5.3). Other delay-measurement methods may be used, but they must be
//! designed to produce results similar to the Annex C method.
//!
//! ## The measuring points (Figure C.1)
//!
//! * **Point A** — the video input to the video coder.
//! * **Point B** — the channel output from the video terminal (i.e.
//!   *including* any FEC, channel framing, etc. — see [`crate::bch`]).
//! * **Point C** — the video output from the decoder.
//!
//! The **encoder delay** is the time from when the visible identification
//! mark changes at point A to the time the change is detected at point B.
//! The **decoder delay** is the corresponding measurement taken between
//! points B and C. The overall video delay (A → C) is the sum.
//!
//! ## The identification mark (§C, third bullet)
//!
//! A video sequence lasting more than 100 seconds is connected to the
//! coder input. The sequence must:
//!
//! * contain a typical moving scene consistent with the codec's purpose;
//! * produce a minimum coded picture rate of **7.5 Hz** at the bit rate
//!   in use;
//! * contain a visible identification mark that changes every **97**
//!   video input frames and is located within the picture area
//!   represented by **the first GOB** (e.g. the first block toggling
//!   black ↔ white). The mark must be detectable at point B and must not
//!   significantly contribute to overall coding performance;
//! * be arranged so the bitstream contains **less than 10 %** stuffing
//!   (MBA stuffing + FEC fill bits).
//!
//! ## How this module helps
//!
//! Like [`crate::hrd`], this module is a measurement / compliance helper:
//! nothing it does changes the on-wire bitstream. It provides:
//!
//! 1. [`MarkGenerator`] — stamps the toggling identification mark into the
//!    first block of the first GOB of a luma plane, on the §C 97-frame
//!    schedule, so a caller can build a conformant Annex C test sequence
//!    from any source frames.
//! 2. [`MarkDetector`] — reads the mark's mean luma in a decoded (or
//!    channel-tapped) frame and reports the *transition* instants at a
//!    mid-level threshold (the §C note on pre/post-temporal processing
//!    recommends taking a mid-level for establishing the transition at
//!    points B and C).
//! 3. [`DelayMeter`] — accumulates the mark-change instants observed at
//!    points A, B, C across the sequence and reports the averaged encoder,
//!    decoder, and overall delays (§C: "several measurements should be
//!    made … and the average period obtained").
//! 4. [`SequenceRequirements`] — validates the §C.1 test-sequence gate
//!    (duration, coded rate, stuffing fraction, mark interval) so a
//!    measurement run can be rejected up front if the input does not meet
//!    the Annex C preconditions.
//!
//! All time is expressed in **frame-interval units** at the source frame
//! rate (the §C identification mark changes on an integer number of input
//! frames), with helpers to convert to seconds when a concrete frame rate
//! is supplied. The arithmetic is integer / rational so a >100 s sequence
//! (≥ 3000 frames at 29.97 Hz) accumulates without floating-point drift.

use crate::picture::SourceFormat;

/// §C identification-mark change interval, in video **input** frames.
///
/// "The visible identification should change every 97 video input frames"
/// (Annex C). The mark toggles (e.g. black ↔ white) on this period.
pub const MARK_INTERVAL_FRAMES: u32 = 97;

/// §C minimum test-sequence duration, in seconds.
///
/// "A video sequence lasting more than 100 seconds is connected to the
/// video coder input."
pub const MIN_SEQUENCE_SECONDS: u32 = 100;

/// §C minimum coded picture rate, in Hz × 10 (= 7.5 Hz).
///
/// "It should produce a minimum coded picture rate of 7.5 Hz at the bit
/// rate in use." Stored ×10 so the 7.5 boundary is exact in integer
/// arithmetic.
pub const MIN_CODED_RATE_TIMES_10: u32 = 75;

/// §C maximum stuffing fraction, in percent.
///
/// "The codec and video sequence should be arranged so that the
/// bitstream contains less than 10 % stuffing (MBA stuffing + error
/// correction fill bits)."
pub const MAX_STUFFING_PERCENT: u32 = 10;

/// The two luminance levels the §C identification mark toggles between.
///
/// Annex C gives "black to white" as the worked example. Black is luma 16
/// and white is luma 235 in the studio range H.261 reconstructs to, but
/// the procedure only requires two clearly separable levels whose
/// transition is detectable at a mid-level threshold; full-range 0 / 255
/// maximises separation against codec ringing. The defaults here use the
/// full-range pair and expose the mid-level threshold accordingly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MarkLevels {
    /// Luma value written when the mark is in its "low" (e.g. black) state.
    pub low: u8,
    /// Luma value written when the mark is in its "high" (e.g. white) state.
    pub high: u8,
}

impl Default for MarkLevels {
    fn default() -> Self {
        // Black ↔ white, full range, maximal separation for detection.
        MarkLevels { low: 0, high: 255 }
    }
}

impl MarkLevels {
    /// The mid-level threshold used to establish a transition at points B
    /// and C (§C note: "it may be necessary to take a mid-level for
    /// establishing the transition of the identification mark at points B
    /// and C").
    ///
    /// Rounded to nearest; for the default `0 / 255` pair this is `128`.
    pub fn mid_level(self) -> u8 {
        // Round-to-nearest of the two levels' midpoint; for an even sum
        // this is the exact midpoint, for an odd sum it rounds up (= the
        // div_ceil of the sum).
        (self.low as u16 + self.high as u16).div_ceil(2) as u8
    }
}

/// The §C identification-mark state on a given input frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkState {
    /// The mark is at its low level (e.g. black).
    Low,
    /// The mark is at its high level (e.g. white).
    High,
}

impl MarkState {
    /// The luma value this state writes for the given [`MarkLevels`].
    pub fn level(self, levels: MarkLevels) -> u8 {
        match self {
            MarkState::Low => levels.low,
            MarkState::High => levels.high,
        }
    }
}

/// The geometry of the identification-mark region: the first 8×8 block of
/// the first GOB of the picture.
///
/// In H.261 the picture origin is the top-left; the first GOB covers the
/// top-left of the picture and its first macroblock's first luminance
/// block is the top-left 8×8 luma samples. §C says the mark is "located
/// within the picture area represented by the first GOB" and gives "the
/// first block in the picture" as the worked example, so the canonical
/// mark region is the `[0,8) × [0,8)` luma block.
pub const MARK_BLOCK_SIZE: usize = 8;

/// Generates the §C toggling identification mark on its 97-frame schedule.
///
/// Construct with the source format (so the mark lands in the first GOB of
/// the right geometry) and the two toggle levels, then call
/// [`MarkGenerator::state_for_frame`] to learn the mark state on a given
/// input frame index, or [`MarkGenerator::stamp`] to write it into a luma
/// plane in place.
#[derive(Clone, Copy, Debug)]
pub struct MarkGenerator {
    fmt: SourceFormat,
    levels: MarkLevels,
}

impl MarkGenerator {
    /// New generator for `fmt` with the given toggle levels.
    pub fn new(fmt: SourceFormat, levels: MarkLevels) -> Self {
        MarkGenerator { fmt, levels }
    }

    /// New generator with the default black ↔ white toggle levels.
    pub fn with_defaults(fmt: SourceFormat) -> Self {
        MarkGenerator::new(fmt, MarkLevels::default())
    }

    /// The §C mark state on input frame `frame` (0-based).
    ///
    /// The mark changes every [`MARK_INTERVAL_FRAMES`] frames: frames
    /// `0..97` are `Low`, `97..194` are `High`, `194..291` are `Low`, … —
    /// i.e. the state is `High` iff `floor(frame / 97)` is odd.
    pub fn state_for_frame(&self, frame: u32) -> MarkState {
        if (frame / MARK_INTERVAL_FRAMES) % 2 == 1 {
            MarkState::High
        } else {
            MarkState::Low
        }
    }

    /// `true` if the mark *changes* between input frame `frame - 1` and
    /// `frame` (i.e. `frame` is a positive multiple of 97).
    ///
    /// Frame 0 is never a change (there is no preceding frame).
    pub fn is_change_frame(&self, frame: u32) -> bool {
        frame != 0 && frame % MARK_INTERVAL_FRAMES == 0
    }

    /// Stamp the §C identification mark for input frame `frame` into the
    /// first 8×8 luma block of `y` (stride `y_stride`).
    ///
    /// Writes the toggle level for the mark state at `frame` across the
    /// top-left `8 × 8` luma samples. The chroma planes are not touched —
    /// the mark is a luma-only feature so it does not perturb the colour
    /// of the scene beyond the first block.
    ///
    /// Panics if `y` is too small to hold the first block at the given
    /// stride (an 8×8 block needs `7 * y_stride + 8` samples).
    pub fn stamp(&self, frame: u32, y: &mut [u8], y_stride: usize) {
        let level = self.state_for_frame(frame).level(self.levels);
        for row in 0..MARK_BLOCK_SIZE {
            let base = row * y_stride;
            y[base..base + MARK_BLOCK_SIZE].fill(level);
        }
    }

    /// The source format this generator stamps for.
    pub fn source_format(&self) -> SourceFormat {
        self.fmt
    }

    /// The toggle levels.
    pub fn levels(&self) -> MarkLevels {
        self.levels
    }
}

/// Detects the §C identification mark in a frame at a measuring point.
///
/// At points B (channel-tapped, decoded for measurement) and C (decoder
/// output) the mark transition is established at a **mid-level**
/// threshold, per the §C note about pre/post-temporal processing
/// smearing a hard black↔white edge. [`MarkDetector::sample`] returns the
/// observed [`MarkState`] of a frame by comparing the mark block's mean
/// luma against that threshold.
#[derive(Clone, Copy, Debug)]
pub struct MarkDetector {
    threshold: u8,
}

impl MarkDetector {
    /// New detector using the mid-level threshold of `levels`.
    pub fn new(levels: MarkLevels) -> Self {
        MarkDetector {
            threshold: levels.mid_level(),
        }
    }

    /// New detector with an explicit luma threshold (`>= threshold` ⇒
    /// `High`).
    pub fn with_threshold(threshold: u8) -> Self {
        MarkDetector { threshold }
    }

    /// The mid-level threshold this detector compares against.
    pub fn threshold(&self) -> u8 {
        self.threshold
    }

    /// Mean luma of the first 8×8 block of `y` (stride `y_stride`),
    /// rounded to nearest.
    ///
    /// Panics if `y` is too small to hold the first block.
    pub fn mark_block_mean(y: &[u8], y_stride: usize) -> u8 {
        let mut sum = 0u32;
        for row in 0..MARK_BLOCK_SIZE {
            let base = row * y_stride;
            for &v in &y[base..base + MARK_BLOCK_SIZE] {
                sum += v as u32;
            }
        }
        let n = (MARK_BLOCK_SIZE * MARK_BLOCK_SIZE) as u32;
        ((sum + n / 2) / n) as u8
    }

    /// Observe the mark state of a frame's luma plane: `High` iff the mark
    /// block's mean luma is `>=` the mid-level threshold.
    pub fn sample(&self, y: &[u8], y_stride: usize) -> MarkState {
        if Self::mark_block_mean(y, y_stride) >= self.threshold {
            MarkState::High
        } else {
            MarkState::Low
        }
    }
}

/// One observed mark-change instant at a measuring point, in frame-interval
/// units from the start of the sequence.
///
/// At point A the instant is exact (the generator changes the mark on a
/// known input frame). At points B and C it is the frame index at which the
/// detector *first* observes the new state — the delay is the difference
/// of these instants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MarkEvent {
    /// Which toggle the mark moved to.
    pub state: MarkState,
    /// Frame-interval index from the start of the sequence at which the
    /// change occurred / was detected.
    pub frame: u32,
}

/// Accumulates §C mark-change instants at points A, B, C and reports the
/// averaged encoder / decoder / overall delays.
///
/// "Several measurements should be made during the sequence length and the
/// average period obtained. Several tests should be made to ensure that a
/// consistent average figure can be obtained for both encoder and decoder
/// delay times" (Annex C). This meter pairs each A-change with the matching
/// B-change (same target state, in order) and each B-change with the
/// matching C-change, then averages the per-pair deltas.
#[derive(Clone, Debug, Default)]
pub struct DelayMeter {
    /// Mark changes injected at point A (the video coder input).
    pub point_a: Vec<MarkEvent>,
    /// Mark changes detected at point B (the channel output).
    pub point_b: Vec<MarkEvent>,
    /// Mark changes detected at point C (the decoder output).
    pub point_c: Vec<MarkEvent>,
}

/// The averaged §C delay figures, in frame-interval units.
///
/// Delays are rational (`numer / denom` frame intervals) so the average of
/// several integer-frame deltas keeps full precision; use
/// [`DelaySummary::encoder_frames`] etc. for an `f64` view or
/// [`DelaySummary::encoder_seconds`] for seconds at a given frame rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelaySummary {
    /// Sum of A→B per-change deltas, in frame intervals.
    pub encoder_delta_sum: u64,
    /// Number of A→B change pairs averaged.
    pub encoder_pairs: u32,
    /// Sum of B→C per-change deltas, in frame intervals.
    pub decoder_delta_sum: u64,
    /// Number of B→C change pairs averaged.
    pub decoder_pairs: u32,
}

impl DelaySummary {
    fn avg(sum: u64, pairs: u32) -> Option<f64> {
        if pairs == 0 {
            None
        } else {
            Some(sum as f64 / pairs as f64)
        }
    }

    /// Average encoder delay (A → B), in frame intervals. `None` if no
    /// A→B pair was observed.
    pub fn encoder_frames(&self) -> Option<f64> {
        Self::avg(self.encoder_delta_sum, self.encoder_pairs)
    }

    /// Average decoder delay (B → C), in frame intervals. `None` if no
    /// B→C pair was observed.
    pub fn decoder_frames(&self) -> Option<f64> {
        Self::avg(self.decoder_delta_sum, self.decoder_pairs)
    }

    /// Average overall video delay (A → C), in frame intervals = encoder
    /// delay + decoder delay. `None` unless both legs were observed.
    pub fn overall_frames(&self) -> Option<f64> {
        match (self.encoder_frames(), self.decoder_frames()) {
            (Some(e), Some(d)) => Some(e + d),
            _ => None,
        }
    }

    /// Average encoder delay in seconds at a source frame rate of
    /// `fps_times_10000 / 10000` Hz (e.g. `299_700` for 29.97 Hz).
    pub fn encoder_seconds(&self, fps_times_10000: u32) -> Option<f64> {
        self.encoder_frames()
            .map(|f| f * 10_000.0 / fps_times_10000 as f64)
    }

    /// Average decoder delay in seconds at the given source frame rate.
    pub fn decoder_seconds(&self, fps_times_10000: u32) -> Option<f64> {
        self.decoder_frames()
            .map(|f| f * 10_000.0 / fps_times_10000 as f64)
    }

    /// Average overall delay in seconds at the given source frame rate.
    pub fn overall_seconds(&self, fps_times_10000: u32) -> Option<f64> {
        self.overall_frames()
            .map(|f| f * 10_000.0 / fps_times_10000 as f64)
    }
}

impl DelayMeter {
    /// New empty meter.
    pub fn new() -> Self {
        DelayMeter::default()
    }

    /// Record a mark change observed at point A (the video coder input).
    pub fn record_a(&mut self, ev: MarkEvent) {
        self.point_a.push(ev);
    }

    /// Record a mark change observed at point B (the channel output).
    pub fn record_b(&mut self, ev: MarkEvent) {
        self.point_b.push(ev);
    }

    /// Record a mark change observed at point C (the decoder output).
    pub fn record_c(&mut self, ev: MarkEvent) {
        self.point_c.push(ev);
    }

    /// Pair up the change instants at two points (same target state, in
    /// order) and sum the per-pair frame deltas.
    ///
    /// The two event lists are matched positionally: the i-th change at the
    /// upstream point is paired with the i-th change at the downstream
    /// point, and the pair is only counted if they agree on the target
    /// state (a defensive check that the two taps observed the same toggle
    /// sequence) and the downstream instant is not earlier. Returns
    /// `(sum_of_deltas, pair_count)`.
    fn pair_deltas(up: &[MarkEvent], down: &[MarkEvent]) -> (u64, u32) {
        let mut sum = 0u64;
        let mut pairs = 0u32;
        for (u, d) in up.iter().zip(down.iter()) {
            if u.state == d.state && d.frame >= u.frame {
                sum += (d.frame - u.frame) as u64;
                pairs += 1;
            }
        }
        (sum, pairs)
    }

    /// Compute the averaged §C delay summary from the recorded instants.
    pub fn summary(&self) -> DelaySummary {
        let (enc_sum, enc_pairs) = Self::pair_deltas(&self.point_a, &self.point_b);
        let (dec_sum, dec_pairs) = Self::pair_deltas(&self.point_b, &self.point_c);
        DelaySummary {
            encoder_delta_sum: enc_sum,
            encoder_pairs: enc_pairs,
            decoder_delta_sum: dec_sum,
            decoder_pairs: dec_pairs,
        }
    }
}

/// Why an Annex C test sequence fails the §C.1 preconditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceError {
    /// Fewer input frames than 100 s at the stated frame rate (§C: "more
    /// than 100 seconds").
    TooShort {
        /// The minimum frame count for >100 s at the stated rate.
        min_frames: u32,
        /// The actual frame count supplied.
        got_frames: u32,
    },
    /// The coded picture rate is below the §C 7.5 Hz minimum.
    CodedRateTooLow {
        /// Stated coded rate ×10.
        coded_rate_times_10: u32,
    },
    /// The stuffing fraction reaches or exceeds the §C 10 % limit.
    TooMuchStuffing {
        /// Observed stuffing percent (rounded).
        stuffing_percent: u32,
    },
    /// The sequence is shorter than one mark interval, so no §C mark
    /// change can be measured.
    NoMarkChange {
        /// The required minimum (97 + 1) frames.
        min_frames: u32,
        /// The actual frame count supplied.
        got_frames: u32,
    },
}

/// The §C.1 test-sequence preconditions, validated against a candidate run.
///
/// Build one from the measured run parameters and call
/// [`SequenceRequirements::validate`] before trusting a [`DelaySummary`].
#[derive(Clone, Copy, Debug)]
pub struct SequenceRequirements {
    /// Total input frame count of the sequence.
    pub frame_count: u32,
    /// Source frame rate ×10000 (e.g. `299_700` for 29.97 Hz).
    pub fps_times_10000: u32,
    /// Achieved coded picture rate ×10 (e.g. `300` for 30.0 Hz, `75` for
    /// 7.5 Hz).
    pub coded_rate_times_10: u32,
    /// Total stuffing bits in the bitstream (MBA stuffing + FEC fill).
    pub stuffing_bits: u64,
    /// Total bitstream bits (so the stuffing *fraction* can be checked).
    pub total_bits: u64,
}

impl SequenceRequirements {
    /// The minimum frame count for a sequence "lasting more than 100
    /// seconds" at the stated source frame rate.
    ///
    /// `ceil(100 s × fps)` plus one frame to make the duration strictly
    /// greater than 100 s.
    pub fn min_frames_for_duration(&self) -> u32 {
        // 100 * fps = 100 * fps_times_10000 / 10000, ceil, then +1 for ">".
        let num = MIN_SEQUENCE_SECONDS as u64 * self.fps_times_10000 as u64;
        let base = num.div_ceil(10_000) as u32;
        base + 1
    }

    /// The observed stuffing fraction, in percent (rounded to nearest).
    pub fn stuffing_percent(&self) -> u32 {
        if self.total_bits == 0 {
            return 0;
        }
        ((self.stuffing_bits * 100 + self.total_bits / 2) / self.total_bits) as u32
    }

    /// Validate all §C.1 preconditions, returning the first failure.
    pub fn validate(&self) -> Result<(), SequenceError> {
        // A mark change needs at least one full interval + 1 frame.
        let min_mark = MARK_INTERVAL_FRAMES + 1;
        if self.frame_count < min_mark {
            return Err(SequenceError::NoMarkChange {
                min_frames: min_mark,
                got_frames: self.frame_count,
            });
        }
        let min_frames = self.min_frames_for_duration();
        if self.frame_count < min_frames {
            return Err(SequenceError::TooShort {
                min_frames,
                got_frames: self.frame_count,
            });
        }
        if self.coded_rate_times_10 < MIN_CODED_RATE_TIMES_10 {
            return Err(SequenceError::CodedRateTooLow {
                coded_rate_times_10: self.coded_rate_times_10,
            });
        }
        let stuff = self.stuffing_percent();
        if stuff >= MAX_STUFFING_PERCENT {
            return Err(SequenceError::TooMuchStuffing {
                stuffing_percent: stuff,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_levels_mid_level_is_halfway() {
        assert_eq!(MarkLevels::default().mid_level(), 128);
        assert_eq!(MarkLevels { low: 16, high: 235 }.mid_level(), 126);
        assert_eq!(
            MarkLevels {
                low: 100,
                high: 100
            }
            .mid_level(),
            100
        );
    }

    #[test]
    fn mark_toggles_every_97_frames() {
        let g = MarkGenerator::with_defaults(SourceFormat::Qcif);
        // First interval is Low.
        assert_eq!(g.state_for_frame(0), MarkState::Low);
        assert_eq!(g.state_for_frame(96), MarkState::Low);
        // Second interval is High.
        assert_eq!(g.state_for_frame(97), MarkState::High);
        assert_eq!(g.state_for_frame(193), MarkState::High);
        // Third interval is Low again.
        assert_eq!(g.state_for_frame(194), MarkState::Low);
        // Change frames are positive multiples of 97.
        assert!(!g.is_change_frame(0));
        assert!(g.is_change_frame(97));
        assert!(!g.is_change_frame(98));
        assert!(g.is_change_frame(194));
    }

    #[test]
    fn stamp_then_detect_round_trips_both_states() {
        let levels = MarkLevels::default();
        let g = MarkGenerator::new(SourceFormat::Qcif, levels);
        let det = MarkDetector::new(levels);
        let (w, h) = SourceFormat::Qcif.dimensions();
        let mut y = vec![64u8; (w * h) as usize];

        // Low interval frame.
        g.stamp(10, &mut y, w as usize);
        assert_eq!(det.sample(&y, w as usize), MarkState::Low);
        assert_eq!(MarkDetector::mark_block_mean(&y, w as usize), 0);

        // High interval frame.
        g.stamp(100, &mut y, w as usize);
        assert_eq!(det.sample(&y, w as usize), MarkState::High);
        assert_eq!(MarkDetector::mark_block_mean(&y, w as usize), 255);
    }

    #[test]
    fn detector_uses_mid_level_threshold() {
        let levels = MarkLevels::default(); // 0 / 255, mid = 128
        let det = MarkDetector::new(levels);
        let (w, _h) = SourceFormat::Qcif.dimensions();
        let stride = w as usize;
        // A block at exactly the mid level reads High (>= threshold).
        let mut y = vec![0u8; (w * SourceFormat::Qcif.dimensions().1) as usize];
        for row in 0..MARK_BLOCK_SIZE {
            y[row * stride..row * stride + MARK_BLOCK_SIZE].fill(128);
        }
        assert_eq!(det.sample(&y, stride), MarkState::High);
        // One below the threshold reads Low.
        for row in 0..MARK_BLOCK_SIZE {
            y[row * stride..row * stride + MARK_BLOCK_SIZE].fill(127);
        }
        assert_eq!(det.sample(&y, stride), MarkState::Low);
    }

    #[test]
    fn delay_meter_averages_constant_delays() {
        // Synthetic A/B/C instants: encoder delay = 3 frames, decoder = 2.
        let mut meter = DelayMeter::new();
        let states = [MarkState::High, MarkState::Low, MarkState::High];
        let a_frames = [97u32, 194, 291];
        for (s, &af) in states.iter().zip(a_frames.iter()) {
            meter.record_a(MarkEvent {
                state: *s,
                frame: af,
            });
            meter.record_b(MarkEvent {
                state: *s,
                frame: af + 3,
            });
            meter.record_c(MarkEvent {
                state: *s,
                frame: af + 3 + 2,
            });
        }
        let sum = meter.summary();
        assert_eq!(sum.encoder_frames(), Some(3.0));
        assert_eq!(sum.decoder_frames(), Some(2.0));
        assert_eq!(sum.overall_frames(), Some(5.0));
        // At 29.97 Hz, 3 frame intervals ≈ 0.1001 s.
        let enc_s = sum.encoder_seconds(299_700).unwrap();
        assert!((enc_s - 3.0 / 29.97).abs() < 1e-9);
    }

    #[test]
    fn delay_meter_averages_varying_delays() {
        let mut meter = DelayMeter::new();
        // Encoder deltas 2, 4 → avg 3. Decoder deltas 1, 3 → avg 2.
        meter.record_a(MarkEvent {
            state: MarkState::High,
            frame: 97,
        });
        meter.record_b(MarkEvent {
            state: MarkState::High,
            frame: 99,
        });
        meter.record_c(MarkEvent {
            state: MarkState::High,
            frame: 100,
        });
        meter.record_a(MarkEvent {
            state: MarkState::Low,
            frame: 194,
        });
        meter.record_b(MarkEvent {
            state: MarkState::Low,
            frame: 198,
        });
        meter.record_c(MarkEvent {
            state: MarkState::Low,
            frame: 201,
        });
        let sum = meter.summary();
        assert_eq!(sum.encoder_frames(), Some(3.0));
        assert_eq!(sum.decoder_frames(), Some(2.0));
    }

    #[test]
    fn delay_meter_rejects_mismatched_state_pairs() {
        let mut meter = DelayMeter::new();
        meter.record_a(MarkEvent {
            state: MarkState::High,
            frame: 97,
        });
        // Detected the opposite state at B — not a valid pair.
        meter.record_b(MarkEvent {
            state: MarkState::Low,
            frame: 100,
        });
        let sum = meter.summary();
        assert_eq!(sum.encoder_frames(), None);
        assert_eq!(sum.encoder_pairs, 0);
    }

    #[test]
    fn sequence_min_frames_strictly_exceeds_100s() {
        let req = SequenceRequirements {
            frame_count: 0,
            fps_times_10000: 299_700, // 29.97 Hz
            coded_rate_times_10: 300,
            stuffing_bits: 0,
            total_bits: 1,
        };
        // 100 s × 29.97 = 2997 frames; ">" needs 2998.
        assert_eq!(req.min_frames_for_duration(), 2998);
    }

    #[test]
    fn sequence_validation_gate() {
        let ok = SequenceRequirements {
            frame_count: 3000,
            fps_times_10000: 299_700,
            coded_rate_times_10: 300, // 30 Hz ≥ 7.5
            stuffing_bits: 50,
            total_bits: 1000, // 5 % < 10 %
        };
        assert_eq!(ok.validate(), Ok(()));

        // Too short.
        let short = SequenceRequirements {
            frame_count: 2000,
            ..ok
        };
        assert!(matches!(
            short.validate(),
            Err(SequenceError::TooShort { .. })
        ));

        // Coded rate below 7.5 Hz.
        let slow = SequenceRequirements {
            coded_rate_times_10: 74,
            ..ok
        };
        assert!(matches!(
            slow.validate(),
            Err(SequenceError::CodedRateTooLow { .. })
        ));
        // Exactly 7.5 Hz is allowed (the §C minimum).
        let at_min = SequenceRequirements {
            coded_rate_times_10: 75,
            ..ok
        };
        assert_eq!(at_min.validate(), Ok(()));

        // 10 % stuffing or more is rejected (< 10 % required).
        let stuffed = SequenceRequirements {
            stuffing_bits: 100,
            total_bits: 1000,
            ..ok
        };
        assert!(matches!(
            stuffed.validate(),
            Err(SequenceError::TooMuchStuffing { .. })
        ));
        // 9 % is fine.
        let lean = SequenceRequirements {
            stuffing_bits: 90,
            total_bits: 1000,
            ..ok
        };
        assert_eq!(lean.validate(), Ok(()));

        // Shorter than one mark interval.
        let tiny = SequenceRequirements {
            frame_count: 50,
            ..ok
        };
        assert!(matches!(
            tiny.validate(),
            Err(SequenceError::NoMarkChange { .. })
        ));
    }

    #[test]
    fn stuffing_percent_rounds() {
        let req = SequenceRequirements {
            frame_count: 3000,
            fps_times_10000: 299_700,
            coded_rate_times_10: 300,
            stuffing_bits: 95,
            total_bits: 1000,
        };
        assert_eq!(req.stuffing_percent(), 10); // 9.5 % rounds to 10
    }
}
