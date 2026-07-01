//! H.261 multipoint control signals — §2.8 / §4.3 of ITU-T Rec. H.261.
//!
//! H.261 §2.8 states that "features necessary to support switched multipoint
//! operation are included." §4.3 enumerates the three control signals that
//! implement those features. Two of the three travel by **external means**
//! (for example Recommendation H.221, i.e. the audiovisual multiplex that
//! carries the H.261 video stream) — they are *not* part of the H.261
//! elementary bitstream — while the third is signalled in-band by a single
//! PTYPE bit:
//!
//! * **§4.3.1 Freeze picture request** — "Causes the decoder to freeze its
//!   displayed picture until a freeze picture release signal is received or a
//!   timeout period of at least six seconds has expired." Transmitted by
//!   external means.
//! * **§4.3.2 Fast update request** — "Causes the encoder to encode its next
//!   picture in INTRA mode with coding parameters such as to avoid buffer
//!   overflow." Transmitted by external means.
//! * **§4.3.3 Freeze picture release** — "A signal from an encoder which has
//!   responded to a fast update request and allows a decoder to exit from its
//!   freeze picture mode … This signal is transmitted by bit 3 of PTYPE (see
//!   §4.2.1) in the picture header of the first picture coded in response to
//!   the fast update request."
//!
//! This module models the two receiver-side (decoder) and one transmitter-side
//! (encoder) *state machines* that consume those signals. It changes nothing
//! about the on-wire bitstream by itself — like the [`crate::hrd`] and
//! [`crate::annex_c`] measurement helpers, it is control-plane logic that the
//! [`crate::decoder`] / [`crate::encoder`] wire into their picture loops (the
//! freeze bit *does* end up in the bitstream, but it is emitted by the
//! encoder's existing [`crate::encoder::Ptype::freeze_picture_release`] flag —
//! this module only decides *when* to set it and *when* a decoder acts on it).
//!
//! # Freeze-picture state machine (decoder side, §4.3.1 + §4.3.3)
//!
//! [`FreezeState`] tracks whether the decoder is currently freezing its
//! displayed output. It enters the frozen state on a §4.3.1 external freeze
//! request and leaves it on **either** of the two §4.3.1 / §4.3.3 release
//! conditions:
//!
//! * a decoded picture whose PTYPE bit 3 (freeze-picture-release) is set
//!   (§4.3.3), or
//! * the §4.3.1 timeout of "at least six seconds" measured from the freeze
//!   request.
//!
//! The timeout is expressed in *decoded-picture ticks* (each call to
//! [`FreezeState::on_decoded_picture`] advances the clock by one nominal
//! source-picture interval of 1001/30000 s ≈ 33.37 ms, per §3.1) so the state
//! machine needs no wall clock. A caller that has a real monotonic clock can
//! instead drive [`FreezeState::on_elapsed`] with an elapsed-nanoseconds
//! delta. Both paths honour the "at least six seconds" lower bound: the timeout
//! fires only once the accumulated time reaches or exceeds
//! [`FREEZE_TIMEOUT_NANOS`].
//!
//! # Fast-update state machine (encoder side, §4.3.2 + §4.3.3)
//!
//! [`FastUpdateState`] latches a pending §4.3.2 fast-update request. The
//! encoder polls [`FastUpdateState::take_for_next_picture`] once per picture:
//! when a request is pending it returns a [`FastUpdateResponse`] that mandates
//! (a) the next picture be coded INTRA (§4.3.2) and (b) its PTYPE carry the
//! freeze-picture-release bit (§4.3.3), then clears the latch. Otherwise it
//! returns the no-op response (normal mode decision, no forced release bit).

/// The §4.3.1 freeze-picture timeout lower bound, in nanoseconds.
///
/// §4.3.1 requires the decoder to hold the frozen picture "until a freeze
/// picture release signal is received or a timeout period of **at least six
/// seconds** has expired." Six seconds is the *minimum*; a decoder may hold
/// longer, but must not release earlier on the timeout path. We use exactly
/// six seconds as the release threshold.
pub const FREEZE_TIMEOUT_NANOS: u64 = 6_000_000_000;

/// The nominal H.261 source-picture interval, as a rational `(num, den)`
/// seconds — §3.1: pictures occur `30000/1001` times per second, so one
/// interval is `1001/30000` s. Used to convert a decoded-picture tick into an
/// elapsed-time increment for the §4.3.1 timeout without a wall clock.
pub const PICTURE_INTERVAL_NUM: u64 = 1001;
/// Denominator of [`PICTURE_INTERVAL_NUM`].
pub const PICTURE_INTERVAL_DEN: u64 = 30_000;

/// One nominal source-picture interval in nanoseconds
/// (`1001/30000 s = 33_366_666.67 ns`, truncated to whole nanoseconds).
///
/// `1_000_000_000 * 1001 / 30000 = 33_366_666` (integer division). The
/// truncation is at most one nanosecond per tick, negligible against the
/// six-second window (180 ticks), and always rounds the accumulated time
/// *down*, so the timeout can only fire slightly later than an exact clock
/// would — never earlier — preserving the §4.3.1 "at least six seconds" lower
/// bound.
pub const PICTURE_INTERVAL_NANOS: u64 = 1_000_000_000 * PICTURE_INTERVAL_NUM / PICTURE_INTERVAL_DEN;

/// Decoder-side freeze-picture state machine (§4.3.1 + §4.3.3).
///
/// Construct with [`FreezeState::new`] (idle, not frozen). Drive a §4.3.1
/// external freeze request with [`FreezeState::request_freeze`]; then, for
/// every decoded picture, call [`FreezeState::on_decoded_picture`] passing the
/// picture's freeze-picture-release PTYPE bit. The return value tells the
/// display layer whether to show the newly decoded picture
/// ([`DisplayAction::Show`]) or keep repeating the last displayed one
/// ([`DisplayAction::Freeze`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FreezeState {
    frozen: bool,
    /// Accumulated time since the freeze request, in nanoseconds. Only
    /// meaningful while `frozen`.
    elapsed_nanos: u64,
}

/// What the display layer should do with a just-decoded picture, per the
/// freeze-picture state machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayAction {
    /// Not frozen (or just released): display the decoded picture normally.
    Show,
    /// Frozen: hold the previously displayed picture; do not advance the
    /// display to this decoded picture.
    Freeze,
}

impl Default for FreezeState {
    fn default() -> Self {
        Self::new()
    }
}

impl FreezeState {
    /// A fresh, un-frozen state machine.
    pub fn new() -> Self {
        Self {
            frozen: false,
            elapsed_nanos: 0,
        }
    }

    /// `true` while the decoder is holding a frozen display picture.
    pub fn is_frozen(self) -> bool {
        self.frozen
    }

    /// Nanoseconds accumulated toward the §4.3.1 timeout since the current
    /// freeze request. Zero when not frozen.
    pub fn elapsed_nanos(self) -> u64 {
        if self.frozen {
            self.elapsed_nanos
        } else {
            0
        }
    }

    /// Apply a §4.3.1 external freeze-picture request. Idempotent while already
    /// frozen — a fresh request restarts the six-second timeout clock (a
    /// re-request extends the hold, matching "until … or a timeout … has
    /// expired" measured from the most recent request).
    pub fn request_freeze(&mut self) {
        self.frozen = true;
        self.elapsed_nanos = 0;
    }

    /// Consume a decoded picture. `freeze_release` is that picture's PTYPE bit
    /// 3 (§4.2.1 / §4.3.3). Advances the §4.3.1 timeout clock by exactly one
    /// nominal source-picture interval ([`PICTURE_INTERVAL_NANOS`]).
    ///
    /// Returns whether the display should advance to this picture
    /// ([`DisplayAction::Show`]) or keep holding the frozen one
    /// ([`DisplayAction::Freeze`]).
    ///
    /// Release order (§4.3.1 / §4.3.3): a set `freeze_release` bit releases the
    /// freeze *before* this picture is shown, so the released picture is
    /// itself displayed. Otherwise the timeout is checked after advancing the
    /// clock; a picture that triggers the timeout is shown (the freeze has
    /// ended by the time it arrives).
    pub fn on_decoded_picture(&mut self, freeze_release: bool) -> DisplayAction {
        if !self.frozen {
            return DisplayAction::Show;
        }
        // §4.3.3: an explicit release bit ends the freeze; the releasing
        // picture is displayed.
        if freeze_release {
            self.frozen = false;
            self.elapsed_nanos = 0;
            return DisplayAction::Show;
        }
        // §4.3.1: advance the timeout clock by one picture interval and check
        // the ≥ 6 s bound.
        self.elapsed_nanos = self.elapsed_nanos.saturating_add(PICTURE_INTERVAL_NANOS);
        if self.elapsed_nanos >= FREEZE_TIMEOUT_NANOS {
            self.frozen = false;
            self.elapsed_nanos = 0;
            DisplayAction::Show
        } else {
            DisplayAction::Freeze
        }
    }

    /// Advance the §4.3.1 timeout clock by an explicit elapsed-nanoseconds
    /// delta (for a caller that has a real monotonic clock rather than
    /// counting decoded-picture ticks). Returns whether the timeout has fired
    /// and the freeze has been released.
    ///
    /// A no-op (returns `false`) when not frozen. When the accumulated time
    /// reaches [`FREEZE_TIMEOUT_NANOS`], the freeze is released and this
    /// returns `true`. This path never inspects a release bit — combine it
    /// with [`FreezeState::on_decoded_picture`] for the §4.3.3 in-band release.
    pub fn on_elapsed(&mut self, delta_nanos: u64) -> bool {
        if !self.frozen {
            return false;
        }
        self.elapsed_nanos = self.elapsed_nanos.saturating_add(delta_nanos);
        if self.elapsed_nanos >= FREEZE_TIMEOUT_NANOS {
            self.frozen = false;
            self.elapsed_nanos = 0;
            true
        } else {
            false
        }
    }
}

/// Encoder-side fast-update state machine (§4.3.2 + §4.3.3).
///
/// A §4.3.2 fast-update request (from a far-end decoder via external means,
/// e.g. an RTCP FIR / an H.245 videoFastUpdatePicture command, or a local
/// scene-cut heuristic) is latched with [`FastUpdateState::request`]. The
/// encoder polls [`FastUpdateState::take_for_next_picture`] before coding each
/// picture; a pending request produces a [`FastUpdateResponse`] that mandates
/// an INTRA picture carrying the §4.3.3 freeze-release bit, and clears the
/// latch so the request is honoured exactly once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FastUpdateState {
    pending: bool,
}

/// The encoder action mandated for the next picture by the fast-update state
/// machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastUpdateResponse {
    /// §4.3.2: the next picture MUST be coded INTRA when `true`.
    pub force_intra: bool,
    /// §4.3.3: the next picture's PTYPE bit 3 (freeze-picture-release) MUST be
    /// set when `true` — signalling a far-end decoder frozen by §4.3.1 that it
    /// may exit freeze mode. Set together with `force_intra` in response to a
    /// fast-update request.
    pub set_freeze_release: bool,
}

impl FastUpdateResponse {
    /// The no-op response: normal mode decision, no forced release bit.
    pub const NONE: FastUpdateResponse = FastUpdateResponse {
        force_intra: false,
        set_freeze_release: false,
    };

    /// `true` if this response mandates any deviation from normal coding.
    pub fn is_active(self) -> bool {
        self.force_intra || self.set_freeze_release
    }
}

impl FastUpdateState {
    /// A fresh state machine with no pending request.
    pub fn new() -> Self {
        Self { pending: false }
    }

    /// `true` while a §4.3.2 fast-update request is latched and not yet
    /// consumed.
    pub fn is_pending(self) -> bool {
        self.pending
    }

    /// Latch a §4.3.2 fast-update request. Idempotent — a second request
    /// before the first is consumed collapses into the same single response
    /// (one INTRA picture satisfies any number of pending fast-update
    /// requests).
    pub fn request(&mut self) {
        self.pending = true;
    }

    /// Poll the latch for the next picture. Returns the mandated
    /// [`FastUpdateResponse`] and clears any pending request, so the fast
    /// update is applied to exactly one picture.
    pub fn take_for_next_picture(&mut self) -> FastUpdateResponse {
        if self.pending {
            self.pending = false;
            FastUpdateResponse {
                force_intra: true,
                set_freeze_release: true,
            }
        } else {
            FastUpdateResponse::NONE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picture_interval_nanos_matches_spec_rate() {
        // §3.1: 30000/1001 pictures/s ⇒ 1001/30000 s per picture.
        // 1_000_000_000 * 1001 / 30000 = 33_366_666 ns (truncated).
        assert_eq!(PICTURE_INTERVAL_NANOS, 33_366_666);
        // 180 ticks (6 s / 33.37 ms) is the ballpark timeout in picture units.
        let ticks_to_timeout = FREEZE_TIMEOUT_NANOS.div_ceil(PICTURE_INTERVAL_NANOS);
        assert_eq!(ticks_to_timeout, 180);
    }

    #[test]
    fn freeze_default_shows_normally() {
        let mut fs = FreezeState::new();
        assert!(!fs.is_frozen());
        // Not frozen: every picture is shown, release bit irrelevant.
        assert_eq!(fs.on_decoded_picture(false), DisplayAction::Show);
        assert_eq!(fs.on_decoded_picture(true), DisplayAction::Show);
    }

    #[test]
    fn freeze_holds_until_release_bit() {
        let mut fs = FreezeState::new();
        fs.request_freeze();
        assert!(fs.is_frozen());
        // A few pictures with no release bit stay frozen (well under 6 s).
        for _ in 0..10 {
            assert_eq!(fs.on_decoded_picture(false), DisplayAction::Freeze);
            assert!(fs.is_frozen());
        }
        // §4.3.3: the releasing picture is itself shown, and the freeze ends.
        assert_eq!(fs.on_decoded_picture(true), DisplayAction::Show);
        assert!(!fs.is_frozen());
        // Subsequent pictures show normally.
        assert_eq!(fs.on_decoded_picture(false), DisplayAction::Show);
    }

    #[test]
    fn freeze_times_out_after_six_seconds() {
        let mut fs = FreezeState::new();
        fs.request_freeze();
        // 179 picture ticks < 6 s ⇒ still frozen.
        for _ in 0..179 {
            assert_eq!(fs.on_decoded_picture(false), DisplayAction::Freeze);
        }
        assert!(fs.is_frozen());
        assert!(fs.elapsed_nanos() < FREEZE_TIMEOUT_NANOS);
        // The 180th tick crosses 6 s ⇒ timeout releases; this picture shown.
        assert_eq!(fs.on_decoded_picture(false), DisplayAction::Show);
        assert!(!fs.is_frozen());
    }

    #[test]
    fn freeze_timeout_never_fires_before_six_seconds() {
        // §4.3.1 "at least six seconds": the timeout must never fire earlier
        // than 6 s. Accumulate 179 intervals and assert still below.
        let mut fs = FreezeState::new();
        fs.request_freeze();
        for _ in 0..179 {
            let _ = fs.on_decoded_picture(false);
        }
        // 179 * 33_366_666 = 5_972_633_214 ns < 6_000_000_000.
        assert!(fs.elapsed_nanos() < FREEZE_TIMEOUT_NANOS);
        assert!(fs.is_frozen());
    }

    #[test]
    fn re_request_restarts_timeout() {
        let mut fs = FreezeState::new();
        fs.request_freeze();
        for _ in 0..100 {
            let _ = fs.on_decoded_picture(false);
        }
        let before = fs.elapsed_nanos();
        assert!(before > 0);
        fs.request_freeze(); // §4.3.1 re-request restarts the clock
        assert_eq!(fs.elapsed_nanos(), 0);
        assert!(fs.is_frozen());
    }

    #[test]
    fn on_elapsed_wall_clock_timeout() {
        let mut fs = FreezeState::new();
        // Not frozen: no-op.
        assert!(!fs.on_elapsed(10_000_000_000));
        fs.request_freeze();
        // Just under 6 s: still frozen.
        assert!(!fs.on_elapsed(5_999_999_999));
        assert!(fs.is_frozen());
        // Cross the boundary: released.
        assert!(fs.on_elapsed(1));
        assert!(!fs.is_frozen());
    }

    #[test]
    fn on_elapsed_exact_boundary_releases() {
        let mut fs = FreezeState::new();
        fs.request_freeze();
        // Exactly 6 s ⇒ release (">= " bound).
        assert!(fs.on_elapsed(FREEZE_TIMEOUT_NANOS));
        assert!(!fs.is_frozen());
    }

    #[test]
    fn fast_update_default_is_noop() {
        let mut fu = FastUpdateState::new();
        assert!(!fu.is_pending());
        let r = fu.take_for_next_picture();
        assert_eq!(r, FastUpdateResponse::NONE);
        assert!(!r.is_active());
    }

    #[test]
    fn fast_update_forces_one_intra_release_picture() {
        let mut fu = FastUpdateState::new();
        fu.request();
        assert!(fu.is_pending());
        let r = fu.take_for_next_picture();
        assert!(r.force_intra); // §4.3.2
        assert!(r.set_freeze_release); // §4.3.3
        assert!(r.is_active());
        assert!(!fu.is_pending());
        // Consumed exactly once: the following picture is normal.
        assert_eq!(fu.take_for_next_picture(), FastUpdateResponse::NONE);
    }

    #[test]
    fn fast_update_coalesces_repeat_requests() {
        let mut fu = FastUpdateState::new();
        fu.request();
        fu.request(); // second request before consuming collapses
        let r = fu.take_for_next_picture();
        assert!(r.is_active());
        // One INTRA picture satisfies both.
        assert_eq!(fu.take_for_next_picture(), FastUpdateResponse::NONE);
    }
}
