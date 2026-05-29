//! Criterion benchmarks for the H.261 decoder hot paths.
//!
//! Round 175 (depth-mode benchmarks). The decoder pipeline runs
//! start-code scan (§4.2.1.1 PSC, §4.2.2.1 GBSC), picture + GOB
//! header parsing, MBA-diff VLC (Table 1/H.261), MTYPE / MQUANT /
//! MVD / CBP VLCs (Tables 2-4), TCOEFF VLC + 20-bit `(esc | run |
//! level)` escape (Table 5), inverse zigzag (Figure 12),
//! dequantisation (§4.2.4), per-block 8×8 IDCT, INTRA prediction
//! (raw 8-bit pels) or INTER prediction (integer-pel MC plus
//! optional FIL loop filter), and 4:2:0 YUV recon assembly.
//!
//! These benches make the per-format / I-vs-P / quant cost visible
//! so future decoder rounds (e.g. a fixed-point IDCT or a SIMD VLC
//! decoder) have a baseline to A/B against.
//!
//! Scenarios:
//!
//!   - **decode_qcif_intra_q8**: 176×144 I-only picture at quant=8.
//!     The bytes are produced by the in-crate encoder once during
//!     bench setup; the timed loop only runs the decoder.
//!   - **decode_qcif_inter_chain_4**: I + 3 P-pictures at quant=8.
//!     Adds the per-MB MC + FIL paths on top of the intra baseline.
//!     Per iteration the decoder is rebuilt fresh so reference-
//!     picture state doesn't carry over.
//!   - **decode_cif_intra_q8**: 352×288 (4× the area of QCIF) I-only.
//!     Same paths at the larger picture size.
//!
//! Source bytes are produced by the production encoder over a
//! synthetic striped pattern (see `benches/encode.rs` for the
//! generator). No on-disk fixtures, no third-party CLI, no `docs/`
//! files are read at bench time.
//!
//! Run with:
//!     cargo bench -p oxideav-h261 --bench decode
//!
//! Sub-scenario selection:
//!     cargo bench -p oxideav-h261 --bench decode -- qcif_intra

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_core::{CodecId, Decoder, Packet, TimeBase};
use oxideav_h261::decoder::H261Decoder;
use oxideav_h261::encoder::{encode_intra_picture, H261Encoder};
use oxideav_h261::picture::SourceFormat;
use oxideav_h261::CODEC_ID_STR;

fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Mirror of `benches/encode.rs::build_frame` so the decoder benches
/// see the same source-content distribution the encoder benches do.
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
            let noise = (xorshift32(&mut state) as i32) >> 30;
            let v = (stripe + diag + grad + noise).clamp(0, 255);
            y[j * w + i] = v as u8;
        }
    }
    let cb = vec![128u8; (w / 2) * (h / 2)];
    let cr = vec![128u8; (w / 2) * (h / 2)];
    (y, cb, cr)
}

fn build_intra_stream(fmt: SourceFormat, w: usize, h: usize) -> Vec<u8> {
    let (y, cb, cr) = build_frame(w, h, 0);
    encode_intra_picture(fmt, &y, w, &cb, w / 2, &cr, w / 2, 8, 0).expect("intra")
}

fn build_ipp_stream(w: usize, h: usize, n: usize) -> Vec<u8> {
    let mut enc = H261Encoder::new(
        if w == 176 {
            SourceFormat::Qcif
        } else {
            SourceFormat::Cif
        },
        8,
    );
    let mut bytes = Vec::new();
    for f in 0..n {
        let (y, cb, cr) = build_frame(w, h, f);
        let p = enc
            .encode_frame(&y, w, &cb, w / 2, &cr, w / 2)
            .expect("encode_frame");
        bytes.extend_from_slice(&p);
    }
    bytes
}

fn drive_decoder(bytes: &[u8]) {
    let mut dec = H261Decoder::new(CodecId::new(CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes.to_vec());
    dec.send_packet(&pkt).expect("send_packet");
    // Drain every ready frame so the IDCT + MC paths actually run.
    while let Ok(f) = dec.receive_frame() {
        black_box(f);
    }
    dec.flush().expect("flush");
    while let Ok(f) = dec.receive_frame() {
        black_box(f);
    }
}

fn bench_intra_qcif(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_qcif_intra_q8");
    group.throughput(Throughput::Elements((176 * 144) as u64));
    let stream = build_intra_stream(SourceFormat::Qcif, 176, 144);
    group.bench_function(BenchmarkId::new("intra", "176x144_q8"), |b| {
        b.iter(|| drive_decoder(black_box(&stream)));
    });
    group.finish();
}

fn bench_inter_chain_qcif(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_qcif_inter_chain_4_q8");
    group.throughput(Throughput::Elements((176 * 144 * 4) as u64));
    let stream = build_ipp_stream(176, 144, 4);
    group.bench_function(BenchmarkId::new("ipp", "176x144_q8_n4"), |b| {
        b.iter(|| drive_decoder(black_box(&stream)));
    });
    group.finish();
}

fn bench_intra_cif(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_cif_intra_q8");
    group.throughput(Throughput::Elements((352 * 288) as u64));
    let stream = build_intra_stream(SourceFormat::Cif, 352, 288);
    group.bench_function(BenchmarkId::new("intra", "352x288_q8"), |b| {
        b.iter(|| drive_decoder(black_box(&stream)));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_intra_qcif,
    bench_inter_chain_qcif,
    bench_intra_cif
);
criterion_main!(benches);
