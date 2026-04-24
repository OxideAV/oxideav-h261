//! Forward 8x8 DCT for H.261 (inverse of `idct.rs`).
//!
//! H.261 Annex A does not mandate a specific forward-transform algorithm;
//! it only bounds the inverse-transform accuracy. For simplicity and
//! numerical symmetry with our decoder's `f32` IDCT we use a plain
//! separable 1-D DCT applied to rows then columns, in `f32` precision.
//!
//! Transform (§3.2.4, inverse form; we implement the forward):
//!
//! ```text
//!   F(u, v) = (1/4) * C(u) * C(v) *
//!             sum_{x=0..7} sum_{y=0..7} f(x, y)
//!                  cos(pi (2x+1) u / 16) cos(pi (2y+1) v / 16)
//! ```
//!
//! with `C(0) = 1/sqrt(2)`, `C(k>0) = 1`.
//!
//! Input samples are expected to be in the natural spatial range
//! (for INTRA: pel values `0..=255` passed through unchanged; the DC
//! coefficient will then be in `0..=2040`). Output is `f32` and the
//! caller is responsible for rounding + clipping to the 12-bit signed
//! range `-2048..=2047`.

const N: usize = 8;

static COS_TAB: fdct_cache::CosTab = fdct_cache::CosTab::new();

mod fdct_cache {
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

/// 1-D 8-point DCT. `in_row` is spatial, `out_row` is frequency.
fn dct_1d(in_row: &[f32; N], out_row: &mut [f32; N]) {
    let ct = COS_TAB.get();
    let c0 = 1.0 / std::f32::consts::SQRT_2;
    for k in 0..N {
        let mut sum = 0.0f32;
        for n in 0..N {
            sum += in_row[n] * ct[k][n];
        }
        let c = if k == 0 { c0 } else { 1.0 };
        out_row[k] = sum * c;
    }
}

/// Full 8x8 forward DCT. `block` is overwritten in-place with coefficient
/// samples. The overall `1/4` factor is split evenly between the two
/// passes (`1/2` each).
pub fn fdct8x8(block: &mut [f32; 64]) {
    // Row pass.
    let mut tmp = [0.0f32; 64];
    let mut row_in = [0.0f32; 8];
    let mut row_out = [0.0f32; 8];
    for y in 0..N {
        for x in 0..N {
            row_in[x] = block[y * N + x];
        }
        dct_1d(&row_in, &mut row_out);
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
        dct_1d(&col_in, &mut col_out);
        for y in 0..N {
            block[y * N + x] = col_out[y] * 0.5;
        }
    }
}

/// Convenience wrapper: forward DCT from 8-bit intra pel samples into
/// signed 12-bit coefficients, clamped to `[-2048, 2047]`.
///
/// Input `pels` are in the natural INTRA range `0..=255`; the DC output is
/// naturally in `0..=2040`, so no sign concerns for the DC.
pub fn fdct_intra(pels: &[u8; 64], out: &mut [i32; 64]) {
    let mut f = [0.0f32; 64];
    for i in 0..64 {
        f[i] = pels[i] as f32;
    }
    fdct8x8(&mut f);
    for i in 0..64 {
        let v = f[i].round() as i32;
        out[i] = v.clamp(-2048, 2047);
    }
}

/// Convenience wrapper: forward DCT from signed residual samples (typical
/// range `-255..=255`) into signed 12-bit coefficients.
pub fn fdct_signed(resid: &[i32; 64], out: &mut [i32; 64]) {
    let mut f = [0.0f32; 64];
    for i in 0..64 {
        f[i] = resid[i] as f32;
    }
    fdct8x8(&mut f);
    for i in 0..64 {
        let v = f[i].round() as i32;
        out[i] = v.clamp(-2048, 2047);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idct::idct8x8;

    #[test]
    fn roundtrip_flat_block() {
        // A constant pel block should roundtrip exactly through DCT+IDCT.
        let mut b = [0.0f32; 64];
        for v in b.iter_mut() {
            *v = 100.0;
        }
        let orig = b;
        fdct8x8(&mut b);
        // DC should be 8 * 100 = 800 (since 1/4 * sqrt(2)*sqrt(2) * sum).
        assert!((b[0] - 800.0).abs() < 0.1, "DC = {}", b[0]);
        for i in 1..64 {
            assert!(b[i].abs() < 1e-3, "AC[{i}] = {}", b[i]);
        }
        idct8x8(&mut b);
        for i in 0..64 {
            assert!(
                (b[i] - orig[i]).abs() < 1e-2,
                "roundtrip at {i}: {} vs {}",
                b[i],
                orig[i]
            );
        }
    }

    #[test]
    fn fdct_intra_dc_from_mid_grey() {
        // A flat 128 block should produce a DC coefficient near 1024 and zero AC.
        let pels = [128u8; 64];
        let mut out = [0i32; 64];
        fdct_intra(&pels, &mut out);
        assert!(
            (out[0] - 1024).abs() <= 1,
            "DC expected 1024, got {}",
            out[0]
        );
        for i in 1..64 {
            assert_eq!(out[i], 0, "AC[{i}] should be zero");
        }
    }

    #[test]
    fn fdct_roundtrip_random_block() {
        // Sanity: a "ramp" intra block round-trips to within a small pel error.
        let mut pels = [0u8; 64];
        for j in 0..8 {
            for i in 0..8 {
                pels[j * 8 + i] = ((j + i) * 16) as u8;
            }
        }
        let mut coeffs = [0i32; 64];
        fdct_intra(&pels, &mut coeffs);
        // Now inverse transform.
        let mut back = [0u8; 64];
        crate::idct::idct_intra(&coeffs, &mut back);
        for i in 0..64 {
            let err = (back[i] as i32 - pels[i] as i32).abs();
            assert!(err <= 2, "roundtrip err at {i}: got {} want {}", back[i], pels[i]);
        }
    }
}
