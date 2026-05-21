//! H.261 §5.2 + Annex B — Hypothetical Reference Decoder (HRD) buffer model.
//!
//! The HRD is a deterministic buffer-occupancy model the encoder must
//! satisfy so a conforming decoder with a known channel rate and buffer
//! size will not under-run mid-picture. It is purely a coder-side
//! compliance check — no bits cross the wire to express it. This module
//! provides the two compliance checks the spec ties to it:
//!
//! 1. **Per-picture cap (§5.2).** No single coded picture (including its
//!    PSC, headers, PSPARE / GSPARE filler, and MBA stuffing — but
//!    *not* the [`crate::bch`] §5.4 framing / `Fi` / fill / BCH parity)
//!    may exceed **256 kbits for CIF** or **64 kbits for QCIF**. This
//!    is independent of bit rate and is a hard ceiling on the output
//!    of a single `encode_intra_picture` / `encode_inter_picture` call.
//!
//! 2. **HRD buffer occupancy (Annex B).** Given a sequence of coded
//!    picture sizes `d_1, d_2, ...` at the encoder output, a constant
//!    channel rate `R` (bit/s), and the per-format frame interval
//!    (CIF = 1/29.97 s, sub-rates 1/(29.97/N) for N = 2, 3, 4 per
//!    §3.1.2), the HRD walks the buffer occupancy:
//!
//!    ```text
//!      b_n+1 = b_n + integral_{t_n}^{t_n+1} R(t) dt - d_n+1
//!    ```
//!
//!    starting from `b_0 = 0` (Annex B.3). The compliance requirement
//!    is `d_n+1 >= b_n + integral R dt - B` — equivalently, after the
//!    picture is removed, the post-removal occupancy `b_n+1` must be
//!    `< B` (Annex B.4). `B = 4 * R_max / 29.97`, where `R_max` is the
//!    peak rate the connection is provisioned for, and the HRD buffer
//!    size is `B + 256 kbits`.
//!
//! Both checks are surfaced as pure functions: callers pass in the
//! per-picture sizes they have / are about to emit; the HRD reports the
//! first violation (if any) and the resulting buffer trajectory. The
//! encoder uses these as compliance assertions in its public test
//! surface; nothing in the on-wire bitstream changes.
//!
//! ## Why this isn't wired into the encoder by default
//!
//! H.261 was designed for synchronous `p × 64 kbit/s` ISDN channels
//! where the encoder has hard real-time bit-rate control via per-GOB
//! MQUANT (§4.2.3.3, see [`crate::encoder`]). When that controller is
//! tuned correctly the HRD compliance follows automatically. For a
//! file-based workflow (the typical OxideAV use case) the encoder
//! emits a single picture at a time and the caller is free to stitch
//! pictures together at any cadence, so a runtime HRD assertion would
//! force a frame-rate / channel-rate assumption that the API doesn't
//! otherwise need. Callers that *do* care about HRD compliance — e.g.
//! to validate a stream against an ISDN-modelled receiver — drive
//! this module explicitly with their picture-size sequence and the
//! channel parameters they target.

use crate::picture::SourceFormat;

/// Per-picture bit-count cap from §5.2.
///
/// `QCIF -> 64 kbits, CIF -> 256 kbits`. Includes the picture start
/// code, all picture / GOB / MB layer headers, MBA stuffing, and
/// PSPARE / GSPARE bytes. Excludes the [`crate::bch`] §5.4
/// transmission-coder framing (`Si`, `Fi`, fill, BCH parity).
pub const fn picture_bits_cap(fmt: SourceFormat) -> u32 {
    match fmt {
        SourceFormat::Qcif => 64 * 1024,
        SourceFormat::Cif => 256 * 1024,
    }
}

/// Nominal coded-picture frame rate, in frames per second × 10000 to
/// preserve the 29.97 (= 30000 / 1001 ≈ 29.9700) digits without floats.
///
/// The HRD operates on integer-rational arithmetic so its per-picture
/// budget never drifts by floating-point round-off across long
/// sequences. Callers wanting a `f64` denominator can use
/// [`frame_rate_hz`].
pub const FRAME_RATE_TIMES_10000: u32 = 299_700;

/// Convenience: the HRD nominal frame rate as `f64`.
pub fn frame_rate_hz() -> f64 {
    FRAME_RATE_TIMES_10000 as f64 / 10_000.0
}

/// HRD parameters: peak channel rate and the derived buffer size `B`.
///
/// `B = 4 * R_max / 29.97` (Annex B.2). The receiving-buffer size is
/// `B + 256 kbits`. Both are stored in bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HrdParams {
    /// Peak (or sustained, if constant) channel rate `R_max`, in bit/s.
    pub r_max_bps: u32,
    /// HRD parameter `B = 4 * R_max / 29.97`, in bits (rounded
    /// half-down to nearest integer). Use [`HrdParams::new`] to compute
    /// this consistently from `r_max_bps`.
    pub b_bits: u64,
}

impl HrdParams {
    /// Construct an [`HrdParams`] from a peak channel bit-rate.
    ///
    /// Computes `B = round(4 * R_max / 29.97)` via the same
    /// integer-rational arithmetic the rest of the module uses
    /// (`B = 4 * R_max * 10_000 / 299_700`, integer division).
    pub fn new(r_max_bps: u32) -> Self {
        // 4 * R / 29.97 = 4 * R * 10000 / 299700.
        let b_bits = (4u64 * r_max_bps as u64 * 10_000) / FRAME_RATE_TIMES_10000 as u64;
        Self { r_max_bps, b_bits }
    }

    /// Receiving-buffer size in bits = `B + 256 kbits` (Annex B.2).
    pub fn rx_buffer_bits(&self) -> u64 {
        self.b_bits + 256 * 1024
    }
}

/// Outcome of a per-picture cap check (§5.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PictureCapStatus {
    /// Picture fits the §5.2 cap.
    Ok,
    /// Picture exceeded the cap.
    Overflow {
        /// Picture's coded size, in bits (caller-supplied).
        actual_bits: u32,
        /// The cap value the picture was checked against.
        cap_bits: u32,
    },
}

/// Check a single picture against the §5.2 per-picture cap.
///
/// `coded_bits` should be the picture's coded-bit count exactly as the
/// encoder would emit it — PSC + headers + MB data + MBA stuffing +
/// PSPARE / GSPARE — but excluding the [`crate::bch`] FEC layer (S, Fi,
/// fill, parity), per the parenthetical in §5.2.
pub fn check_picture_cap(coded_bits: u32, fmt: SourceFormat) -> PictureCapStatus {
    let cap = picture_bits_cap(fmt);
    if coded_bits <= cap {
        PictureCapStatus::Ok
    } else {
        PictureCapStatus::Overflow {
            actual_bits: coded_bits,
            cap_bits: cap,
        }
    }
}

/// Indices into [`HrdTrace::buffer_bits_after_arrival`] follow the
/// caller's picture sequence: entry `i` is the post-removal buffer
/// occupancy `b_{i+1}` after picture `d_{i+1}` is instantaneously
/// removed at time `t_{i+1}`. There is no leading `b_0` (Annex B.3
/// fixes it at 0). The length equals the number of input pictures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HrdTrace {
    /// Post-removal occupancy `b_n` for each picture, in bits.
    pub buffer_bits_after_arrival: Vec<u64>,
    /// First underflow index, if any: `Some(i)` means picture `i+1`
    /// arrived to a buffer state that violates `d_{n+1} >= b_n +
    /// integral R - B`. The trace continues past the underflow so the
    /// caller can inspect later pictures' occupancies for context.
    pub first_underflow: Option<usize>,
}

/// Walk the HRD buffer occupancy across a sequence of coded picture
/// sizes (in bits), at a constant channel rate `R = params.r_max_bps`.
///
/// `pictures_per_skip` is the per-picture skip factor `N` from §3.1.2
/// (`1` = full 29.97 fps, `2` = 14.99 fps, `3` = 9.99 fps, `4` = 7.49
/// fps). The inter-picture interval is `N / 29.97` seconds.
///
/// The model:
///
/// 1. Start with `b_0 = 0` (Annex B.3).
/// 2. For each picture `n`:
///    - Add `integral R dt = R * N / 29.97` bits arriving over the
///      interval before this picture is removed.
///    - Check that `b_n+1 = b_n + arrived - d_n+1 >= 0` — i.e.
///      `d_n+1 <= b_n + arrived`. If not, mark `first_underflow`
///      (and continue so the caller sees the full trace).
///    - Subtract `d_n+1` to get the new occupancy.
/// 3. Compliance also requires the *pre-removal* occupancy stays
///    `< B + 256 kbits` (the receiving buffer doesn't overflow). For
///    a well-behaved encoder this is automatic from §5.2; we surface
///    it as a separate check via [`check_overflow`].
///
/// Note that the spec frames the inequality as `d_n+1 >= b_n + integral
/// R - B` — that's a *lower* bound on `d_n+1` (the next picture must be
/// big enough to drain the buffer below `B`). Our `first_underflow`
/// flags the *opposite* failure (the picture is too big to remove
/// instantaneously without taking the buffer negative). The two are
/// distinct: spec underflow says "decoder buffer can't fill enough to
/// hold the picture-ready state"; ours says "buffer doesn't have the
/// picture's bits yet when removal is scheduled". Both indicate the
/// encoder violated its bit budget — they're complementary views.
pub fn walk_buffer(pictures: &[u32], pictures_per_skip: u32, params: HrdParams) -> HrdTrace {
    assert!(pictures_per_skip >= 1, "skip factor must be ≥ 1");

    let mut trace = HrdTrace {
        buffer_bits_after_arrival: Vec::with_capacity(pictures.len()),
        first_underflow: None,
    };

    // Per-interval arrival = R * N / 29.97 bits. Using the same
    // integer-rational form: arrived = R * N * 10000 / 299700.
    let arrived_per_interval: u64 = (params.r_max_bps as u64 * pictures_per_skip as u64 * 10_000)
        / FRAME_RATE_TIMES_10000 as u64;

    let mut occupancy: u64 = 0;
    for (i, &d) in pictures.iter().enumerate() {
        // Bits arrive over the interval before the (n+1)-th removal.
        occupancy += arrived_per_interval;
        // Removal of picture d_{n+1}: subtracting more than `occupancy`
        // is an underflow (decoder would stall waiting for the picture
        // to finish arriving).
        if (d as u64) > occupancy {
            if trace.first_underflow.is_none() {
                trace.first_underflow = Some(i);
            }
            // Clamp to 0 so the trace continues sanely past the
            // violation. (A real decoder would stall; we surface the
            // first violation index and let the trace progress.)
            occupancy = 0;
        } else {
            occupancy -= d as u64;
        }
        trace.buffer_bits_after_arrival.push(occupancy);
    }

    trace
}

/// Check whether the *pre-removal* buffer occupancy would ever exceed
/// the receiver's `B + 256 kbits` capacity. This is the upper-bound
/// dual of [`walk_buffer`]'s lower-bound underflow check.
///
/// Returns the first picture index where the pre-removal occupancy
/// would exceed `B + 256 kbits`, or `None` if compliant.
pub fn check_overflow(
    pictures: &[u32],
    pictures_per_skip: u32,
    params: HrdParams,
) -> Option<usize> {
    let arrived_per_interval: u64 = (params.r_max_bps as u64 * pictures_per_skip as u64 * 10_000)
        / FRAME_RATE_TIMES_10000 as u64;
    let cap = params.rx_buffer_bits();
    let mut occupancy: u64 = 0;
    for (i, &d) in pictures.iter().enumerate() {
        occupancy = occupancy.saturating_add(arrived_per_interval);
        if occupancy > cap {
            return Some(i);
        }
        occupancy = occupancy.saturating_sub(d as u64);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_picture_cap_qcif_64k() {
        assert_eq!(picture_bits_cap(SourceFormat::Qcif), 64 * 1024);
        assert_eq!(picture_bits_cap(SourceFormat::Cif), 256 * 1024);
    }

    #[test]
    fn check_picture_cap_below_limit_is_ok() {
        assert_eq!(
            check_picture_cap(50_000, SourceFormat::Qcif),
            PictureCapStatus::Ok
        );
        assert_eq!(
            check_picture_cap(200_000, SourceFormat::Cif),
            PictureCapStatus::Ok
        );
    }

    #[test]
    fn check_picture_cap_at_limit_is_ok() {
        // 64 * 1024 = 65536; equality should be Ok (the spec says
        // "must not exceed", inclusive).
        assert_eq!(
            check_picture_cap(64 * 1024, SourceFormat::Qcif),
            PictureCapStatus::Ok
        );
        assert_eq!(
            check_picture_cap(256 * 1024, SourceFormat::Cif),
            PictureCapStatus::Ok
        );
    }

    #[test]
    fn check_picture_cap_over_limit_flags_overflow() {
        let r = check_picture_cap(70_000, SourceFormat::Qcif);
        match r {
            PictureCapStatus::Overflow {
                actual_bits,
                cap_bits,
            } => {
                assert_eq!(actual_bits, 70_000);
                assert_eq!(cap_bits, 64 * 1024);
            }
            PictureCapStatus::Ok => panic!("expected Overflow"),
        }
    }

    #[test]
    fn hrd_params_derive_b_for_64kbit_channel() {
        // R_max = 64 000 bit/s; B = 4 * 64 000 / 29.97 ≈ 8542 bits.
        let p = HrdParams::new(64_000);
        // 4 * 64000 * 10000 / 299700 = 25600000000 / 299700 = 85418.7…
        // integer-div → 8541 (NOT 8542 — we truncate, matching the
        // integer-rational specification of the module).
        assert_eq!(p.b_bits, 8541);
        assert_eq!(p.rx_buffer_bits(), 8541 + 256 * 1024);
    }

    #[test]
    fn hrd_params_derive_b_for_2mbit_channel() {
        // R_max = 2 048 000 bit/s; B = 4 * 2 048 000 / 29.97 ≈ 273 340 bits.
        let p = HrdParams::new(2_048_000);
        // 4 * 2048000 * 10000 / 299700 = 81920000000000 / 299700 ≈ 273_340.007
        // ⇒ integer-div truncates to 273_340.
        assert_eq!(p.b_bits, 273_340);
    }

    #[test]
    fn walk_buffer_constant_rate_constant_picture_holds_steady() {
        // R = 64000 bit/s, full rate (N=1). Per-interval arrival =
        // 64000 / 29.97 ≈ 2135.46 bits. If each picture is exactly
        // 2135 bits the buffer stays near 0.
        let params = HrdParams::new(64_000);
        let per_interval: u64 = (64_000u64 * 10_000) / FRAME_RATE_TIMES_10000 as u64;
        assert_eq!(per_interval, 2135);
        let pics = vec![per_interval as u32; 10];
        let trace = walk_buffer(&pics, 1, params);
        assert_eq!(trace.first_underflow, None);
        for &b in &trace.buffer_bits_after_arrival {
            assert_eq!(b, 0, "buffer should drain to exactly 0 each interval");
        }
    }

    #[test]
    fn walk_buffer_smaller_pictures_accumulate_into_buffer() {
        // Each picture half of per-interval ⇒ buffer grows monotonically.
        let params = HrdParams::new(64_000);
        let per_interval: u64 = (64_000u64 * 10_000) / FRAME_RATE_TIMES_10000 as u64;
        let half = (per_interval / 2) as u32;
        let pics = vec![half; 5];
        let trace = walk_buffer(&pics, 1, params);
        assert_eq!(trace.first_underflow, None);
        // After n intervals: occupancy ≈ n * (per_interval - half).
        let drift_per_step = per_interval - half as u64;
        for (i, &b) in trace.buffer_bits_after_arrival.iter().enumerate() {
            let expected = ((i + 1) as u64) * drift_per_step;
            assert_eq!(b, expected, "step {i}: drift expected {expected}, got {b}");
        }
    }

    #[test]
    fn walk_buffer_oversized_picture_triggers_underflow() {
        let params = HrdParams::new(64_000);
        // First picture requires more bits than arrived ⇒ underflow at index 0.
        let pics = vec![100_000, 1000, 1000];
        let trace = walk_buffer(&pics, 1, params);
        assert_eq!(trace.first_underflow, Some(0));
    }

    #[test]
    fn walk_buffer_skip_factor_doubles_arrival_per_interval() {
        // N=2 ⇒ interval doubles ⇒ arrival per interval doubles too.
        let params = HrdParams::new(64_000);
        let per_interval_n1: u64 = (64_000u64 * 10_000) / FRAME_RATE_TIMES_10000 as u64;
        let pics = vec![per_interval_n1 as u32 * 2; 5];
        let trace = walk_buffer(&pics, 2, params);
        assert_eq!(trace.first_underflow, None);
        for &b in &trace.buffer_bits_after_arrival {
            assert_eq!(b, 0);
        }
    }

    #[test]
    fn check_overflow_does_not_trip_under_normal_drain() {
        // Constant rate, exactly draining: never overflows.
        let params = HrdParams::new(64_000);
        let per_interval: u64 = (64_000u64 * 10_000) / FRAME_RATE_TIMES_10000 as u64;
        let pics = vec![per_interval as u32; 100];
        assert_eq!(check_overflow(&pics, 1, params), None);
    }

    #[test]
    fn check_overflow_trips_when_pictures_are_tiny() {
        // Tiny pictures ⇒ buffer accumulates ⇒ eventually exceeds
        // B + 256 kbits. For R = 64 kbit/s, B = 8541; cap = 8541 +
        // 262144 = 270 685. Per-interval arrival = 2135. With pictures
        // of 0 bits we hit the cap after ceil(270685 / 2135) = 127 frames.
        let params = HrdParams::new(64_000);
        let pics = vec![0u32; 200];
        let idx = check_overflow(&pics, 1, params).expect("should overflow");
        // The overflow index is the first frame where pre-removal occupancy
        // exceeds the cap. 127 * 2135 = 271145 > 270685; 126 * 2135 = 269010
        // (still under). So index 126 (0-indexed = 127th frame).
        assert_eq!(idx, 126);
    }

    #[test]
    fn frame_rate_is_29_97() {
        // Sanity: the public f64 form matches the constant.
        assert!((frame_rate_hz() - 29.97).abs() < 1e-9);
    }
}
