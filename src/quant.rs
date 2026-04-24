//! Forward quantisation for H.261 (inverse of the dequantisation in
//! `block::dequant_ac`).
//!
//! Per §3.2.5 there are exactly two quantiser modes:
//!
//! * **INTRA DC** — a fixed linear quantiser with step = 8 and no dead-zone.
//!   The reconstructed DC level is `8 * n` for bitstream value `n` in
//!   `1..=254` (with `0` and `128` forbidden), and `1024` for `n = 255`.
//!   We therefore encode INTRA DC as `clamp(round(level / 8), 1, 254)`
//!   (avoiding the forbidden bitstream value `128`), with the special
//!   `255` path reserved for `level == 1024`.
//!
//! * **AC / inter coefficients** — 31 linear quantisers indexed by
//!   `QUANT` in `1..=31`, each with step = `2 * QUANT` and a central
//!   dead-zone. The decoder's formula (§4.2.4) is:
//!
//!   ```text
//!   |REC| = QUANT * (2*|level| + 1)    (QUANT odd)
//!   |REC| = QUANT * (2*|level| + 1) - 1 (QUANT even)
//!   ```
//!
//!   The matching forward quantisation (chosen to minimise reconstruction
//!   error under the above formula and to respect the dead-zone) is:
//!
//!   ```text
//!   |level| = floor(|coeff| / (2 * QUANT))
//!   ```
//!
//!   i.e. the classic "truncate" MPEG-style AC quantiser. Level 0 is the
//!   dead-zone band `|coeff| < 2*QUANT`. Levels are clamped to the
//!   signalled 8-bit signed range `-127..=127` (the escape code can carry
//!   `-127..=-1, 1..=127`; value `-128` is forbidden, value `0` is forbidden).

/// Quantise the INTRA DC coefficient per Table 6/H.261.
///
/// Table 6 maps bitstream value `n` to reconstruction level `8 * n`
/// except the special `n == 255` which maps to `level = 1024`. Values
/// `n == 0` and `n == 128` are forbidden in the bitstream. The decoder
/// expects exactly this format.
///
/// Forward: we quantise `level` with step 8 and no dead-zone. The only
/// nuisance is that the naive `n = round(level/8)` equals `128` when
/// `level == 1024`; in that case we must emit `0xFF` instead of `0x80`.
/// For levels in `[1020, 1028)` that also naively round to 128 we pick
/// whichever of `127` / `129` / `255` gives the lowest reconstruction
/// error, since picking `255` reconstructs to 1024 (exact for level=1024,
/// error 4 at level=1020 or 1028).
pub fn quant_intra_dc(level: i32) -> u8 {
    // Naive rounding to nearest multiple of 8.
    let naive = (level + 4).div_euclid(8);
    // Legal non-FF FLC range: 1..=127, 129..=254. Plus 255 encodes 1024 specially.
    // Build candidates and pick closest reconstruction.
    let mut best: u8 = 1;
    let mut best_err: i32 = i32::MAX;
    let consider = |v: u8, level: i32, best: &mut u8, best_err: &mut i32| {
        let rec = if v == 0xFF { 1024 } else { 8 * v as i32 };
        let err = (rec - level).abs();
        if err < *best_err {
            *best_err = err;
            *best = v;
        }
    };
    // Most of the time the naive rounding lands on a legal value.
    if (1..=127).contains(&naive) || (129..=254).contains(&naive) {
        return naive as u8;
    }
    // Otherwise search a narrow neighbourhood of legal codes. We always
    // have 127, 129, 255 available near the 128 hole; at the edges we fall
    // back to 1 or 254.
    for &v in &[1u8, 126, 127, 129, 130, 254, 255] {
        consider(v, level, &mut best, &mut best_err);
    }
    // Clamp to the legal range if `naive` is below 1 or above 254 (can happen
    // for degenerate inputs outside the natural 8-bit intra range).
    if naive <= 0 {
        return 1;
    }
    if naive >= 255 && level != 1024 {
        return 254;
    }
    best
}

/// Forward quantise a single AC coefficient for QUANT `1..=31`. Returns
/// the signed level to be VLC-coded (range `-127..=127`, zero means the
/// coefficient is omitted from the RLE run).
pub fn quant_ac(coeff: i32, quant: u32) -> i32 {
    debug_assert!((1..=31).contains(&quant), "H.261 QUANT must be in 1..=31");
    let step = 2 * quant as i64;
    let abs = (coeff as i64).unsigned_abs() as i64;
    let mag = abs / step;
    let sign = if coeff < 0 { -1i64 } else { 1i64 };
    let level = sign * mag;
    level.clamp(-127, 127) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::dequant_ac;

    #[test]
    fn intra_dc_roundtrip_simple() {
        // Bitstream code 0x10 -> REC = 0x10 * 8 = 128. Coeff 128 -> code 16.
        assert_eq!(quant_intra_dc(128), 16);
        assert_eq!(quant_intra_dc(8), 1);
        assert_eq!(quant_intra_dc(16), 2);
        // Coeff 1024 maps to 0xFF.
        assert_eq!(quant_intra_dc(1024), 0xFF);
    }

    #[test]
    fn intra_dc_avoids_forbidden_codes() {
        // Coeff 1024 is the special 0xFF value.
        assert_eq!(quant_intra_dc(1024), 0xFF);
        // Never emit 0 or 128.
        for level in 0..=2040 {
            let v = quant_intra_dc(level);
            assert_ne!(v, 0, "forbidden code 0 emitted for level {level}");
            assert_ne!(v, 128, "forbidden code 128 emitted for level {level}");
        }
        // Ensure clamping works at the extremes.
        assert_eq!(quant_intra_dc(-10), 1);
        assert_eq!(quant_intra_dc(4000), 254);
    }

    #[test]
    fn ac_quant_dead_zone() {
        for q in [1u32, 2, 8, 15, 31] {
            let step = (2 * q) as i32;
            for c in -(step - 1)..=(step - 1) {
                assert_eq!(quant_ac(c, q), 0, "dead zone coeff={c} q={q}");
            }
        }
    }

    #[test]
    fn ac_quant_roundtrip_odd_q() {
        // With q=1: coeff=3 -> level=1 -> REC=3. Round-trip within 1 step.
        assert_eq!(quant_ac(3, 1), 1);
        assert_eq!(dequant_ac(1, 1), 3);
        assert_eq!(quant_ac(5, 1), 2);
        assert_eq!(dequant_ac(2, 1), 5);
    }

    #[test]
    fn ac_quant_roundtrip_even_q() {
        // q=2: coeff=5 -> level=1 -> REC=5.
        assert_eq!(quant_ac(5, 2), 1);
        assert_eq!(dequant_ac(1, 2), 5);
        assert_eq!(quant_ac(-5, 2), -1);
        assert_eq!(dequant_ac(-1, 2), -5);
    }

    #[test]
    fn ac_quant_round_trip_all_levels_q8() {
        // For any quantised level L, the forward formula should be a
        // left-inverse of the dequant on the centre of its quantisation cell.
        let q = 8u32;
        for level in -127i32..=127 {
            if level == 0 {
                continue;
            }
            let rec = dequant_ac(level, q);
            let requant = quant_ac(rec, q);
            assert_eq!(
                requant, level,
                "q=8 level={level} rec={rec} requant={requant}"
            );
        }
    }

    #[test]
    fn ac_quant_saturates() {
        assert_eq!(quant_ac(i32::MAX, 1), 127);
        assert_eq!(quant_ac(i32::MIN, 1), -127);
    }
}
