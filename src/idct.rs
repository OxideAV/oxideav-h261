//! 8x8 IDCT for H.261 (Annex A of ITU-T Rec. H.261).
//!
//! Annex A defines only an accuracy specification, not a specific algorithm.
//! We use a straightforward separable 1-D IDCT applied to rows then columns,
//! in `f32` precision (well within the Annex A tolerance: peak error <= 1,
//! mean error <= 0.015, mean square error <= 0.06).
//!
//! The transform follows
//!
//! ```text
//!   f(x, y) = (1/4) * sum_{u=0..7} sum_{v=0..7} C(u) C(v) F(u,v)
//!                     cos(pi (2x+1) u / 16) cos(pi (2y+1) v / 16)
//! ```
//!
//! with `C(0) = 1/sqrt(2)`, `C(k>0) = 1`. Input coefficients are 12-bit
//! signed values in `[-2048, 2047]`; output pels are clipped to `[-256, 255]`
//! (inter) or `[0, 255]` (intra, after adding back the DC bias of 128).

/// Normalisation: `C(0) = 1/sqrt(2)`, `C(k) = 1` otherwise. We fold the
/// scaling factor `1/4 * C(u) * C(v)` into the precomputed cosine table.
///
/// `COS_TAB[k][n]` = cos(pi (2n+1) k / 16).
const N: usize = 8;

static COS_TAB: once_cell_once_lock::CosTab = once_cell_once_lock::CosTab::new();

mod once_cell_once_lock {
    use std::sync::OnceLock;
    pub struct CosTab {
        inner: OnceLock<[[f32; 8]; 8]>,
    }
    impl CosTab {
        pub const fn new() -> Self {
            Self {
                inner: OnceLock::new(),
            }
        }
        pub fn get(&self) -> &[[f32; 8]; 8] {
            self.inner.get_or_init(|| {
                let mut t = [[0.0f32; 8]; 8];
                for k in 0..8 {
                    for n in 0..8 {
                        let theta = std::f64::consts::PI * (2.0 * n as f64 + 1.0) * k as f64 / 16.0;
                        t[k][n] = theta.cos() as f32;
                    }
                }
                t
            })
        }
    }
}

/// Run a 1-D 8-point IDCT. Input `in_row[0..8]` are frequency-domain samples
/// (already scaled by the dequantisation formula), output `out_row[0..8]` are
/// spatial samples.
fn idct_1d(in_row: &[f32; N], out_row: &mut [f32; N]) {
    let ct = COS_TAB.get();
    let c0 = 1.0 / std::f32::consts::SQRT_2;
    for n in 0..N {
        let mut sum = 0.0f32;
        for k in 0..N {
            let c = if k == 0 { c0 } else { 1.0 };
            sum += c * in_row[k] * ct[k][n];
        }
        out_row[n] = sum;
    }
}

/// Full 8x8 IDCT. `block` is overwritten in-place by the time domain samples.
/// Callers should clip to `[-256, 255]` (and add 128 for intra) afterwards.
///
/// Mathematically: output[y*8+x] = (1/4) * sum_uv C(u) C(v) F(u,v)
///                                 cos(pi(2x+1)u/16) cos(pi(2y+1)v/16).
/// We implement the separable form: row IDCT then column IDCT; the overall
/// `1/4` factor is split evenly as `1/2 * 1/2` applied once each pass.
pub fn idct8x8(block: &mut [f32; 64]) {
    // Row pass.
    let mut tmp = [0.0f32; 64];
    let mut row_in = [0.0f32; 8];
    let mut row_out = [0.0f32; 8];
    for y in 0..N {
        for x in 0..N {
            row_in[x] = block[y * N + x];
        }
        idct_1d(&row_in, &mut row_out);
        for x in 0..N {
            tmp[y * N + x] = row_out[x] * 0.5;
        }
    }
    // Column pass.
    let mut col_in = [0.0f32; 8];
    let mut col_out = [0.0f32; 8];
    for x in 0..N {
        for y in 0..N {
            col_in[y] = tmp[y * N + x];
        }
        idct_1d(&col_in, &mut col_out);
        for y in 0..N {
            block[y * N + x] = col_out[y] * 0.5;
        }
    }
}

/// Convenience wrapper: IDCT from signed integer coefficients into clipped
/// signed i16 residual samples, range `[-256, 255]`.
pub fn idct_signed(coeffs: &[i32; 64], out: &mut [i32; 64]) {
    let mut f = [0.0f32; 64];
    for i in 0..64 {
        f[i] = coeffs[i] as f32;
    }
    idct8x8(&mut f);
    for i in 0..64 {
        let v = f[i].round() as i32;
        out[i] = v.clamp(-256, 255);
    }
}

/// Convenience wrapper: IDCT from signed integer coefficients into 8-bit
/// intra samples `[0, 255]`. The caller's coefficients must already include
/// the DC offset (INTRA DC of 1024 corresponds to a spatial mean of 128).
pub fn idct_intra(coeffs: &[i32; 64], out: &mut [u8; 64]) {
    let mut f = [0.0f32; 64];
    for i in 0..64 {
        f[i] = coeffs[i] as f32;
    }
    idct8x8(&mut f);
    for i in 0..64 {
        let v = f[i].round() as i32;
        out[i] = v.clamp(0, 255) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_give_zeros() {
        let mut b = [0.0f32; 64];
        idct8x8(&mut b);
        for v in b.iter() {
            assert!(v.abs() < 1e-4);
        }
    }

    #[test]
    fn dc_only_flat_block() {
        // A single DC coefficient of value 8 should, after 1/4 * C(0)^2 * 8
        // = 1/4 * 1/2 * 8 = 1.0 per pel. I.e. the whole 8x8 output = 1.
        let mut b = [0.0f32; 64];
        b[0] = 8.0;
        idct8x8(&mut b);
        for v in b.iter() {
            assert!((*v - 1.0).abs() < 1e-3, "{v}");
        }
    }

    /// Verify the IDCT rounding direction is round-half-to-nearest (which
    /// for f32 `.round()` is round-half-away-from-zero per IEEE 754
    /// "rounding to nearest, ties away from zero"). Per H.261 Annex A
    /// the inverse-transform output is constrained only by accuracy
    /// (peak err <= 1, mean err <= 0.015, mse <= 0.06) — not by a
    /// specific rounding rule. round-half-to-even would also pass; we
    /// use the simpler round-half-away-from-zero which is what
    /// `f32::round` does in Rust.
    #[test]
    fn idct_rounding_is_round_half_away_from_zero() {
        // f32::round rounds halves away from zero.
        assert_eq!(0.5f32.round() as i32, 1);
        assert_eq!((-0.5f32).round() as i32, -1);
        assert_eq!(1.5f32.round() as i32, 2);
        assert_eq!((-1.5f32).round() as i32, -2);
    }

    /// Drift-stress: simulate a 50-step P-frame loop with zero residual
    /// (the limit case where any IDCT precision issue would compound).
    /// Start with a noisy 8x8 spatial block, FDCT it, quant at QUANT=8,
    /// dequant + IDCT to get the recon, then use that recon as the next
    /// "predictor" — but with no residual added (every "frame" sees the
    /// same source). After 50 iterations the recon should remain bit-
    /// identical to iteration 1 (because on iteration ≥2 the predictor
    /// is the prior recon, source-pred=0 residual, IDCT(0)=0, recon=pred).
    #[test]
    fn drift_stress_zero_residual_chain() {
        use crate::block::dequant_ac;
        use crate::fdct::fdct_signed;
        use crate::quant::quant_ac;
        // Synthetic spatial source.
        let mut src = [0u8; 64];
        for (i, s) in src.iter_mut().enumerate() {
            *s = (40 + ((i * 17) % 100)) as u8;
        }
        // Initial predictor = mid-grey.
        let mut pred = [128u8; 64];
        let q = 8u32;
        let mut prev_recon = [0u8; 64];
        for iter in 0..50 {
            // Compute residual = src - pred (signed).
            let mut resid = [0i32; 64];
            for i in 0..64 {
                resid[i] = src[i] as i32 - pred[i] as i32;
            }
            // FDCT + quant + dequant + IDCT to get reconstructed residual.
            let mut coeffs = [0i32; 64];
            fdct_signed(&resid, &mut coeffs);
            let mut levels = [0i32; 64];
            for i in 0..64 {
                levels[i] = quant_ac(coeffs[i], q);
            }
            let mut dequant = [0i32; 64];
            for i in 0..64 {
                dequant[i] = dequant_ac(levels[i], q);
            }
            let mut rec_resid = [0i32; 64];
            idct_signed(&dequant, &mut rec_resid);
            let mut recon = [0u8; 64];
            for i in 0..64 {
                recon[i] = (pred[i] as i32 + rec_resid[i]).clamp(0, 255) as u8;
            }
            if iter >= 2 {
                // After two iterations the recon is fixed: pred==prev_recon,
                // src-pred==0 (well, +/- the IDCT roundoff which the dead-
                // zone immediately squashes back to 0), so recon must equal
                // the previous recon exactly. Any drift here would mean the
                // IDCT/quant pair fails the "fixed point" property.
                assert_eq!(
                    recon, prev_recon,
                    "drift at iter {iter}: recon != prev_recon — IDCT/quant chain is not idempotent"
                );
            }
            prev_recon = recon;
            pred = recon;
        }
    }

    #[test]
    fn intra_dc_level_matches_128() {
        // The spec says INTRA DC reconstruction level 1024 (coded as 0xFF)
        // should give pel 128 (the "mid-grey" reference). With our IDCT
        // convention, DC = 1024 → pel = 1024/8 = 128.
        let mut b = [0i32; 64];
        b[0] = 1024;
        let mut out = [0u8; 64];
        idct_intra(&b, &mut out);
        for v in out.iter() {
            assert_eq!(*v, 128);
        }
    }
}
