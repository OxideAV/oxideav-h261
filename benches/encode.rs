//! Criterion benchmarks for the H.261 encoder hot paths.
//!
//! Round 175 (depth-mode benchmarks). The encoder pipeline runs
//! prediction (intra: copy raw pels; inter: spiral+diamond ME within
//! ±15 pel range per §3.2.2 / Annex A, then integer-pel MC subtract),
//! per-block FDCT + quantise (§4.2.4), per-MB mode decision (Table
//! 2/H.261 MTYPE selection: Intra / Inter{,+MQUANT} / Inter+MC{,+CBP,
//! +CBP+MQUANT} / Inter+MC+FIL{,+CBP,+CBP+MQUANT} via a bit-cost
//! estimator), zigzag + TCOEFF VLC emit (Table 5) including the
//! 20-bit `(escape | run | level)` escape, then GOB + picture
//! framing.
//!
//! These benches make the per-format / I-vs-P / quant cost visible
//! so future encoder rounds (e.g. a SIMD residual-SAD search or a
//! pre-computed CBP-prefix table) have a baseline to A/B against.
//!
//! Scenarios:
//!
//!   - **encode_qcif_intra_q8**: single 176×144 I-picture at quant=8
//!     (the canonical "balanced quality" point). Exercises the full
//!     intra mode-decision + zigzag + TCOEFF VLC across 33 GOBs of
//!     11 MBs each. No motion estimation runs.
//!   - **encode_qcif_inter_chain_4**: I + 3 P-pictures at quant=8.
//!     Adds the ±15 spiral+diamond ME, MC subtract, FIL mode-decision,
//!     and the per-GOB MQUANT rate controller cost on top of the
//!     intra baseline. Throughput is reported per frame in the group.
//!   - **encode_cif_intra_q8**: 352×288 (4× the area of QCIF) I-only.
//!     Exercises the same paths at the larger picture size so the
//!     per-MB constant factor is amortised correctly.
//!
//! Source frames are synthesised in-bench from a deterministic
//! striped pattern plus low-amplitude xorshift noise — high-
//! frequency horizontally + smooth vertically, the regime where
//! the FIL loop filter is most effective. No on-disk fixtures, no
//! third-party CLI, no `docs/` files are read at bench time.
//!
//! Run with:
//!     cargo bench -p oxideav-h261 --bench encode
//!
//! Sub-scenario selection:
//!     cargo bench -p oxideav-h261 --bench encode -- qcif_intra

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_h261::encoder::{encode_inter_picture, encode_intra_picture, H261Encoder};
use oxideav_h261::picture::SourceFormat;

fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Build one synthetic frame at `(w, h)` with a striped + diagonal
/// pattern shifted by `f` (frame index). Stripes 4 pels wide with a
/// 4× contrast step give the encoder both DC content and AC content
/// that survives the dead-zone; the diagonal term and the `shift`
/// dependency on `f` give MC something to find. Returns `(y, cb, cr)`
/// with chroma sub-sampled 4:2:0.
fn build_frame(w: usize, h: usize, f: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut state = 0x1234_5678u32.wrapping_add(f as u32);
    let shift = (f as i32) * 2;
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let xi = (i as i32 - shift).rem_euclid(w as i32);
            let stripe = if (xi / 4) % 2 == 0 { 60 } else { 196 };
            let diag = ((xi + j as i32) % 16) * 2;
            let grad = (j as i32 * 60) / h as i32;
            let noise = (xorshift32(&mut state) as i32) >> 30; // -2..1
            let v = (stripe + diag + grad + noise).clamp(0, 255);
            y[j * w + i] = v as u8;
        }
    }
    // Mid-grey chroma; the codec doesn't care for the bench shape,
    // and a non-128 mean would just bloat the AC residual.
    let cb = vec![128u8; (w / 2) * (h / 2)];
    let cr = vec![128u8; (w / 2) * (h / 2)];
    (y, cb, cr)
}

fn bench_intra_qcif(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_qcif_intra_q8");
    // 33 GOBs of 11 MBs of 6 blocks of 64 samples — one picture.
    group.throughput(Throughput::Elements((176 * 144) as u64));
    let (y, cb, cr) = build_frame(176, 144, 0);
    group.bench_function(BenchmarkId::new("intra", "176x144_q8"), |b| {
        b.iter(|| {
            let bytes = encode_intra_picture(
                SourceFormat::Qcif,
                black_box(&y),
                176,
                black_box(&cb),
                88,
                black_box(&cr),
                88,
                8,
                0,
            )
            .expect("intra");
            black_box(bytes);
        });
    });
    group.finish();
}

fn bench_inter_chain_qcif(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_qcif_inter_chain_4_q8");
    // Per-iteration we encode I + 3 P. Throughput is the total pel
    // count across the four frames so per-frame ns lands in criterion
    // naturally.
    group.throughput(Throughput::Elements((176 * 144 * 4) as u64));
    let frames: Vec<_> = (0..4).map(|f| build_frame(176, 144, f)).collect();
    group.bench_function(BenchmarkId::new("ipp", "176x144_q8_n4"), |b| {
        b.iter(|| {
            let mut enc = H261Encoder::new(SourceFormat::Qcif, 8);
            let mut total = 0usize;
            for (y, cb, cr) in &frames {
                let p = enc
                    .encode_frame(black_box(y), 176, black_box(cb), 88, black_box(cr), 88)
                    .expect("encode_frame");
                total += p.len();
            }
            black_box(total);
        });
    });
    group.finish();
}

fn bench_intra_cif(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_cif_intra_q8");
    group.throughput(Throughput::Elements((352 * 288) as u64));
    let (y, cb, cr) = build_frame(352, 288, 0);
    group.bench_function(BenchmarkId::new("intra", "352x288_q8"), |b| {
        b.iter(|| {
            let bytes = encode_intra_picture(
                SourceFormat::Cif,
                black_box(&y),
                352,
                black_box(&cb),
                176,
                black_box(&cr),
                176,
                8,
                0,
            )
            .expect("intra");
            black_box(bytes);
        });
    });
    group.finish();
}

fn bench_inter_one_qcif(c: &mut Criterion) {
    // Single I + single P measurement (no rate controller carryover)
    // so per-frame P cost is isolatable from the chain bench above.
    let mut group = c.benchmark_group("encode_qcif_inter_one_q8");
    group.throughput(Throughput::Elements((176 * 144) as u64));
    let (y0, cb, cr) = build_frame(176, 144, 0);
    let (y1, _, _) = build_frame(176, 144, 1);
    // Pre-encode the I-frame once and capture its local-recon, since
    // `encode_inter_picture` needs a `&Picture` reference. We import
    // the recon variant for that.
    use oxideav_h261::encoder::encode_intra_picture_with_recon;
    let (_i_bytes, recon_i) =
        encode_intra_picture_with_recon(SourceFormat::Qcif, &y0, 176, &cb, 88, &cr, 88, 8, 0)
            .expect("intra recon");
    group.bench_function(BenchmarkId::new("p_from_i", "176x144_q8"), |b| {
        b.iter(|| {
            let (bytes, _recon) = encode_inter_picture(
                SourceFormat::Qcif,
                black_box(&y1),
                176,
                black_box(&cb),
                88,
                black_box(&cr),
                88,
                8,
                1,
                black_box(&recon_i),
            )
            .expect("inter");
            black_box(bytes);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_intra_qcif,
    bench_inter_one_qcif,
    bench_inter_chain_qcif,
    bench_intra_cif
);
criterion_main!(benches);
