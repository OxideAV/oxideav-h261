//! Criterion benchmarks for the H.261 §3.2.5 / §4.2.4 (de)quantisation
//! leaf primitives — the per-coefficient arithmetic the encoder and
//! decoder run on every transmitted DCT coefficient.
//!
//! Round 336 (depth-mode benchmarks). The `transform` bench covers the
//! inner (I)DCT butterflies and `filter_mc` covers the P-block
//! reconstruction primitives, but neither isolates the (de)quantiser:
//!
//!   * **decode** — `block::dequant_ac` reconstructs `REC` from a
//!     signed VLC level for QUANT `1..=31`. Per §4.2.4 the arithmetic
//!     branches on QUANT parity (`REC = QUANT*(2|level|+1)` for odd
//!     QUANT, `-1` for even QUANT), so it runs once per non-zero
//!     coefficient on every coded block — the decoder hot path the
//!     fuzzer's `decode_h261` target reaches through `decode_intra_block`
//!     / `decode_inter_block`.
//!   * **encode** — `quant::quant_ac` is the matching forward AC
//!     quantiser (`|level| = floor(|coeff| / (2*QUANT))` with the
//!     central dead-zone) and `quant::quant_intra_dc` is the §3.2.5
//!     Table 6 INTRA-DC quantiser (step 8, with the forbidden-code
//!     `0x80` / special `0xFF` handling). Both run on every coefficient
//!     the encoder emits.
//!
//! All three are pure leaf functions with no isolated baseline, so a
//! future optimisation pass (e.g. a branchless QUANT-parity dequant, a
//! reciprocal-multiply forward quantiser, or a SIMD coefficient sweep)
//! now has an A/B distinct from the transform numbers.
//!
//! Each benchmark times a full 64-coefficient 8×8 block so per-element
//! cycle-equivalents land in criterion's report. The AC paths are run in
//! two coefficient regimes (a sparse, mostly-dead-zone block and a dense
//! block where every coefficient survives) and at both QUANT parities,
//! because the dead-zone and the parity branch are what dominate cost.
//!
//! Run with:
//!     cargo bench -p oxideav-h261 --bench quant

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_h261::block::dequant_ac;
use oxideav_h261::quant::{quant_ac, quant_intra_dc};

/// Cheap deterministic xorshift32 (same generator the `transform` /
/// `filter_mc` benches use) so the coefficient content is reproducible
/// and free of on-disk fixtures.
fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Build a 64-entry block of signed VLC levels in the legal
/// `-127..=127` range (zero allowed) for the decode-side `dequant_ac`
/// sweep. A `sparse` block leaves most entries zero (the common case —
/// real DCT blocks are dominated by the dead-zone after quantisation);
/// a dense block fills every slot with a non-zero level so the parity
/// branch fires on all 64 coefficients.
fn build_levels(seed: u32, sparse: bool) -> [i32; 64] {
    let mut state = seed;
    let mut out = [0i32; 64];
    for slot in out.iter_mut() {
        let r = xorshift32(&mut state);
        // Map to -127..=127.
        let v = (r % 255) as i32 - 127;
        if sparse {
            // Keep ~1/4 of the coefficients non-zero, like a real
            // post-quantisation block where most AC energy is gone.
            if r & 0b11 != 0 {
                *slot = 0;
                continue;
            }
        }
        *slot = v;
    }
    out
}

/// Build a 64-entry block of unquantised signed DCT coefficients for the
/// encode-side `quant_ac` sweep. Amplitudes span the dead-zone (small
/// values that quantise to 0) up to the clamp ceiling, so both the
/// dead-zone reject and the saturation paths are represented.
fn build_coeffs(seed: u32) -> [i32; 64] {
    let mut state = seed;
    let mut out = [0i32; 64];
    for slot in out.iter_mut() {
        let r = xorshift32(&mut state) as i32;
        // -2048..=2047 — the §4.2.4 IDCT-input dynamic range.
        *slot = (r >> 20).clamp(-2048, 2047);
    }
    out
}

fn bench_dequant(c: &mut Criterion) {
    let mut group = c.benchmark_group("dequant_block_8x8");
    group.throughput(Throughput::Elements(64));

    let sparse = build_levels(0x00DE_9A47, true);
    let dense = build_levels(0x0010_BEEF, false);

    // QUANT parity changes the §4.2.4 arithmetic (odd: no `-1` correction;
    // even: subtract 1), so bench both. q=7 is odd, q=8 is even.
    for (qname, q) in [("q7_odd", 7u32), ("q8_even", 8u32)] {
        group.bench_function(
            BenchmarkId::new("dequant_ac", format!("sparse_{qname}")),
            |b| {
                b.iter(|| {
                    let mut acc = 0i32;
                    for &level in sparse.iter() {
                        acc = acc.wrapping_add(dequant_ac(black_box(level), black_box(q)));
                    }
                    black_box(acc)
                });
            },
        );
        group.bench_function(
            BenchmarkId::new("dequant_ac", format!("dense_{qname}")),
            |b| {
                b.iter(|| {
                    let mut acc = 0i32;
                    for &level in dense.iter() {
                        acc = acc.wrapping_add(dequant_ac(black_box(level), black_box(q)));
                    }
                    black_box(acc)
                });
            },
        );
    }

    group.finish();
}

fn bench_quant_ac(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant_ac_block_8x8");
    group.throughput(Throughput::Elements(64));

    let coeffs = build_coeffs(0xC0_FFEE);

    for (qname, q) in [("q7_odd", 7u32), ("q8_even", 8u32)] {
        group.bench_function(BenchmarkId::new("quant_ac", qname), |b| {
            b.iter(|| {
                let mut acc = 0i32;
                for &coeff in coeffs.iter() {
                    acc = acc.wrapping_add(quant_ac(black_box(coeff), black_box(q)));
                }
                black_box(acc)
            });
        });
    }

    group.finish();
}

fn bench_quant_intra_dc(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant_intra_dc");
    // One call per INTRA block, but it carries the forbidden-code /
    // special-1024 search, so sweep the whole §3.2.5 level range to
    // exercise both the common fast path and the narrow-neighbourhood
    // search near the forbidden `0x80` hole.
    group.throughput(Throughput::Elements(2041));

    group.bench_function(
        BenchmarkId::new("quant_intra_dc", "level_sweep_0_2040"),
        |b| {
            b.iter(|| {
                let mut acc = 0u32;
                for level in 0i32..=2040 {
                    acc = acc.wrapping_add(quant_intra_dc(black_box(level)) as u32);
                }
                black_box(acc)
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_dequant, bench_quant_ac, bench_quant_intra_dc);
criterion_main!(benches);
