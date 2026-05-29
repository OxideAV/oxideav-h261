//! Criterion benchmarks for the H.261 8×8 transform hot paths.
//!
//! Round 175 (depth-mode benchmarks). The crate is feature-complete
//! across decode + encode for both source formats (QCIF / CIF) plus
//! the §5.4 BCH outer framing, the §5.2 + Annex B HRD model, and the
//! RFC 4587 / RFC 3550 RTP + RTCP wire formats. Per the workspace
//! "saturated → fuzz/bench/profile" memo this round wires up
//! `criterion` so any future optimisation pass (e.g. a SIMD IDCT or
//! a fixed-point FDCT replacement) has a baseline to A/B against.
//!
//! This file covers the **8×8 DCT block** — the inner hot path
//! invoked once per coded block in both the encoder (`fdct_intra`,
//! `fdct_signed`) and the decoder (`idct_intra`, `idct_signed`).
//! The two `_intra` variants clamp / saturate to `u8`; the two
//! `_signed` variants keep the `i32` residual range.
//!
//! Each benchmark times one block (`8×8 = 64` samples). Throughput
//! is reported in elements (samples) so per-sample cycle-equivalents
//! land naturally in criterion's report. Multi-block aggregate cost
//! is exercised in the `encode` and `decode` benches.
//!
//! Run with:
//!     cargo bench -p oxideav-h261 --bench transform

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_h261::fdct::{fdct_intra, fdct_signed};
use oxideav_h261::idct::{idct_intra, idct_signed};

/// Cheap deterministic xorshift32 — synthesises "natural-ish" block
/// inputs. A pure-DC fixture would flatten coefficient cost and hide
/// the per-row butterfly work, so the helpers below mix a low-
/// frequency gradient with small per-sample noise.
fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Build an 8×8 `u8` pel block roughly in the middle of the 8-bit
/// range with a smooth diagonal gradient plus low-amplitude noise.
/// The FDCT cares about all 8 frequencies so we want both DC and AC
/// content represented.
fn build_intra_block(seed: u32) -> [u8; 64] {
    let mut state = seed;
    let mut out = [0u8; 64];
    for j in 0..8 {
        for i in 0..8 {
            let grad = 64 + (i as i32 + j as i32) * 6; // 64..148
            let noise = (xorshift32(&mut state) as i32) >> 28; // -8..7
            out[j * 8 + i] = (grad + noise).clamp(0, 255) as u8;
        }
    }
    out
}

/// Build an 8×8 `i32` residual block with values roughly in `±128`.
/// Mimics what `fdct_signed` sees after a P-MB predictor subtract.
fn build_signed_block(seed: u32) -> [i32; 64] {
    let mut state = seed;
    let mut out = [0i32; 64];
    for j in 0..8 {
        for i in 0..8 {
            let stripe = if (i + j) % 2 == 0 { 24 } else { -24 };
            let noise = (xorshift32(&mut state) as i32) >> 26; // -32..31
            out[j * 8 + i] = stripe + noise;
        }
    }
    out
}

/// Build an 8×8 `i32` coefficient block that resembles what the
/// dequantiser feeds into the IDCT: a non-zero DC plus a sparse
/// scattering of small AC terms (zigzag-first eight positions),
/// the rest zero. Mirrors the typical quantised intra block.
fn build_intra_coeffs(seed: u32) -> [i32; 64] {
    let mut state = seed;
    let mut out = [0i32; 64];
    out[0] = 1024; // DC, post-dequant
                   // First eight zigzag positions get small AC terms.
    let zz = [1, 8, 16, 9, 2, 3, 10, 17];
    for &pos in &zz {
        let v = ((xorshift32(&mut state) as i32) >> 28).abs(); // 0..7
        out[pos] = v * 8;
    }
    out
}

/// Like `build_intra_coeffs` but with the DC scaled down to mimic a
/// P-residual block (DC is small or zero, ACs dominate).
fn build_signed_coeffs(seed: u32) -> [i32; 64] {
    let mut state = seed;
    let mut out = [0i32; 64];
    out[0] = 0;
    let zz = [1, 8, 16, 9, 2, 3, 10, 17, 24, 4];
    for &pos in &zz {
        let v = (xorshift32(&mut state) as i32) >> 26; // -32..31
        out[pos] = v;
    }
    out
}

fn bench_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("transform_8x8");
    group.throughput(Throughput::Elements(64));

    // FDCT — encoder forward pass.
    let intra_pels = build_intra_block(0xCAFE_F00D);
    group.bench_function(BenchmarkId::new("fdct_intra", "u8_block"), |b| {
        let mut out = [0i32; 64];
        b.iter(|| {
            fdct_intra(black_box(&intra_pels), black_box(&mut out));
            black_box(&out);
        });
    });

    let signed_resid = build_signed_block(0xDEAD_BEEF);
    group.bench_function(BenchmarkId::new("fdct_signed", "i32_block"), |b| {
        let mut out = [0i32; 64];
        b.iter(|| {
            fdct_signed(black_box(&signed_resid), black_box(&mut out));
            black_box(&out);
        });
    });

    // IDCT — decoder inverse pass.
    let intra_coeffs = build_intra_coeffs(0x1234_5678);
    group.bench_function(BenchmarkId::new("idct_intra", "u8_block"), |b| {
        let mut out = [0u8; 64];
        b.iter(|| {
            idct_intra(black_box(&intra_coeffs), black_box(&mut out));
            black_box(&out);
        });
    });

    let signed_coeffs = build_signed_coeffs(0x9ABC_DEF0);
    group.bench_function(BenchmarkId::new("idct_signed", "i32_block"), |b| {
        let mut out = [0i32; 64];
        b.iter(|| {
            idct_signed(black_box(&signed_coeffs), black_box(&mut out));
            black_box(&out);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_transform);
criterion_main!(benches);
