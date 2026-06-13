//! Criterion benchmarks for the H.261 P-picture reconstruction hot
//! paths that sit *outside* the 8×8 transform: the §3.2.3 loop filter
//! and the §3.2.2 integer-pel motion-compensation block copy.
//!
//! Round 287 (depth-mode benchmarks). The `transform` bench already
//! covers the inner (I)DCT butterflies; `decode` / `encode` cover the
//! end-to-end picture cost. What was missing was an isolated baseline
//! for the two per-block primitives the decoder runs on every coded
//! P-block: `apply_loop_filter` (the separable 1/4-1/2-1/4 in-loop
//! filter) and `copy_block_integer` (the integer-pel reference fetch).
//! Both are attacker-reachable (the fuzz crate's `decode_h261` target
//! drives them) and both are pure functions, so an optimisation pass
//! (e.g. a SIMD filter or a branchless edge-clamp copy) now has an
//! A/B baseline distinct from the transform numbers.
//!
//! Each benchmark times one 8×8 block (64 samples); throughput is
//! reported in elements so per-sample cycle-equivalents land in
//! criterion's report. The motion-comp copy is exercised in three
//! regimes because its cost is dominated by the per-sample edge clamp:
//!   * `center` — MV keeps the whole block inside the plane (no clamp
//!     ever fires; the common case).
//!   * `corner_clamp` — MV pushes the block off the top-left corner so
//!     every sample clamps (worst case for the branch).
//!   * `mv_nonzero` — a typical small interior MV (±3 pel).
//!
//! Run with:
//!     cargo bench -p oxideav-h261 --bench filter_mc

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_h261::mb::{apply_loop_filter, copy_block_integer};

/// Cheap deterministic xorshift32 — synthesises "natural-ish" pel
/// content. A flat block would let the loop filter collapse to a
/// copy and hide the per-tap arithmetic, so we mix a gradient with
/// low-amplitude noise (same approach as the `transform` bench).
fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Build an 8×8 `u8` pel block with a smooth diagonal gradient plus
/// low-amplitude noise so both the flat interior taps and the AC
/// content the filter is meant to smooth are represented.
fn build_block(seed: u32) -> [u8; 64] {
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

/// Build a reference plane large enough to hold a QCIF luma row of
/// blocks, filled with gradient + noise content. Returned with its
/// stride so the motion-comp copy sees realistic memory access.
fn build_ref_plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    let mut plane = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let grad = 32 + ((x + y) as i32 & 0x7f); // 32..159
            let noise = (xorshift32(&mut state) as i32) >> 28; // -8..7
            plane[y * w + x] = (grad + noise).clamp(0, 255) as u8;
        }
    }
    plane
}

fn bench_loop_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("loop_filter_8x8");
    group.throughput(Throughput::Elements(64));

    let block = build_block(0xF11_7E12);
    group.bench_function(BenchmarkId::new("apply_loop_filter", "u8_block"), |b| {
        b.iter(|| {
            let out = apply_loop_filter(black_box(&block));
            black_box(out);
        });
    });

    group.finish();
}

fn bench_motion_comp(c: &mut Criterion) {
    let mut group = c.benchmark_group("motion_comp_8x8");
    group.throughput(Throughput::Elements(64));

    // QCIF luma dimensions (176×144). Pick a block well inside the
    // plane so the "center" regime never clamps.
    let (rw, rh) = (176usize, 144usize);
    let plane = build_ref_plane(rw, rh, 0x0C7_0BE5);

    // center: block at (80, 64), MV (0,0) — fully interior.
    group.bench_function(BenchmarkId::new("copy_block_integer", "center"), |b| {
        let mut out = [0u8; 64];
        b.iter(|| {
            copy_block_integer(
                black_box(&plane),
                rw,
                rw as i32,
                rh as i32,
                black_box(80),
                black_box(64),
                black_box(0),
                black_box(0),
                &mut out,
            );
            black_box(&out);
        });
    });

    // mv_nonzero: interior block + small (+3,-3) MV — still no clamp.
    group.bench_function(BenchmarkId::new("copy_block_integer", "mv_nonzero"), |b| {
        let mut out = [0u8; 64];
        b.iter(|| {
            copy_block_integer(
                black_box(&plane),
                rw,
                rw as i32,
                rh as i32,
                black_box(80),
                black_box(64),
                black_box(3),
                black_box(-3),
                &mut out,
            );
            black_box(&out);
        });
    });

    // corner_clamp: block + MV push entirely off the top-left corner
    // so every sample's clamp branch fires (worst case).
    group.bench_function(
        BenchmarkId::new("copy_block_integer", "corner_clamp"),
        |b| {
            let mut out = [0u8; 64];
            b.iter(|| {
                copy_block_integer(
                    black_box(&plane),
                    rw,
                    rw as i32,
                    rh as i32,
                    black_box(0),
                    black_box(0),
                    black_box(-15),
                    black_box(-15),
                    &mut out,
                );
                black_box(&out);
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_loop_filter, bench_motion_comp);
criterion_main!(benches);
