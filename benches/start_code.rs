//! Criterion benchmarks for the H.261 §4.1 / §4.2 start-code scanner.
//!
//! Round 238 (depth-mode benchmark). The start-code scanner sits on
//! the inner loop of every H.261 decode and every RTP depacketize:
//!
//! * `H261Decoder::send_packet` walks the incoming elementary stream
//!   one start code at a time (`iter_start_codes`) to chop the buffer
//!   into picture-aligned chunks before the picture-layer parser
//!   takes over.
//! * The §4.2 GOB-aligned RTP packetizer (`rtp::packetize_gob_aligned`)
//!   slices an elementary stream at every GBSC; both ends of the
//!   slice come from `iter_start_codes`.
//! * `rtp::depacketize` runs `iter_start_codes` on the reassembled
//!   payload as the spec-mandated sanity check (RFC 4587 §4: a
//!   depacketized H.261 stream must contain at least one start code).
//!
//! The scanner walks the bitstream bit-by-bit looking for the 16-bit
//! prefix `0000 0000 0000 0001` per §4.1 — the only start-code shape
//! the codec emits (PSC = GBSC with `GN == 0`, plain GBSC otherwise).
//! Per §4.1 the prefix is **not** byte-aligned in general (the GOB
//! layer ends on whatever bit-position the inner bitstream consumes),
//! so the scanner cannot take the usual byte-aligned shortcut every
//! H.263+ scanner can rely on. That makes this primitive a worthwhile
//! optimisation target: a single round of vectorised pre-scan over
//! byte-aligned `0x00 0x01` candidates followed by the bit-walk on
//! the few near-hit windows would shave most of the per-stream cost.
//! This bench gives that future change an A/B baseline.
//!
//! Scenarios:
//!
//!   - **iter_start_codes / qcif_intra_one_frame** — one full QCIF
//!     I-picture produced by the in-tree encoder. Expected
//!     start-code count: 1 PSC + 3 GBSCs (QCIF has 3 GOBs, numbered
//!     1, 3, 5 per §3.2.1).
//!   - **iter_start_codes / cif_intra_one_frame** — one full CIF
//!     I-picture. Expected: 1 PSC + 12 GBSCs (CIF has 12 GOBs).
//!   - **iter_start_codes / qcif_intra_three_frames** — three
//!     concatenated QCIF I-pictures (typical "show me start codes in
//!     a tiny GOP buffer" workload). Exercises the early-out path of
//!     the iterator after the last PSC.
//!   - **find_next_start_code / qcif_intra_first** — single
//!     byte-aligned scan to the very first PSC of a fresh QCIF
//!     elementary stream (the §4.2 GOB-aligned packetizer's first
//!     slice). Best-case cost (hit at bit 0).
//!   - **find_next_start_code_bits / qcif_intra_misaligned_start** —
//!     same QCIF stream but the bit-walker starts at bit 3 of the
//!     buffer, exercising the in-byte realignment cost the §4.2
//!     packetizer's slow path pays when a GOB does not happen to
//!     land on a byte boundary.
//!   - **find_next_start_code / no_start_code_in_buffer** — a
//!     pseudo-random 4-KiB buffer with no start code inside.
//!     Worst-case "scan the whole thing and return `None`" — the
//!     cost a misbehaving network endpoint can force on an RTP
//!     receiver per fuzzed payload.
//!
//! Throughput is reported in elements = bytes scanned, so per-byte
//! cost lands naturally next to the other benches (e.g. a future
//! SIMD pre-scan can be compared against the bit-by-bit walker on a
//! MiB/s basis without spreadsheet gymnastics).
//!
//! All inputs are synthesised in-bench from a deterministic
//! xorshift seed (for the no-start-code case) or from the in-crate
//! encoder (for the hit cases) — no committed fixtures, no on-disk
//! testsrc, no `docs/` files at bench time. Run with:
//!     cargo bench -p oxideav-h261 --bench start_code

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_h261::encoder::H261Encoder;
use oxideav_h261::picture::SourceFormat;
use oxideav_h261::start_code::{find_next_start_code, find_next_start_code_bits, iter_start_codes};

/// Cheap deterministic xorshift32 — same generator the other H.261
/// benches use. Produces a flat distribution suitable for the
/// "no start code anywhere" worst-case bench buffer.
fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Build a striped + low-amplitude noise YUV 4:2:0 source frame of
/// the given dimensions. Same shape used by `benches/encode.rs` so
/// the produced elementary stream is representative of what the
/// encoder typically sees from real content. Frame-rate / temporal
/// behaviour does not matter here — every bench frame is an
/// I-picture.
fn synth_yuv(width: usize, height: usize, seed: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut state = seed;
    let cw = width / 2;
    let ch = height / 2;
    let mut y = vec![0u8; width * height];
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for j in 0..height {
        for i in 0..width {
            let stripe = ((i + j) & 7) as i32 * 8; // 0..56
            let noise = (xorshift32(&mut state) as i32) >> 28; // -8..7
            y[j * width + i] = (64 + stripe + noise).clamp(0, 255) as u8;
        }
    }
    for j in 0..ch {
        for i in 0..cw {
            let n = (xorshift32(&mut state) as i32) >> 28;
            cb[j * cw + i] = (128 + n).clamp(0, 255) as u8;
            cr[j * cw + i] = (128 - n).clamp(0, 255) as u8;
        }
    }
    (y, cb, cr)
}

/// Encode `n_frames` I-pictures of the given format and return the
/// concatenated elementary stream. The first frame already carries a
/// PSC; subsequent intra-only frames each begin with their own PSC.
fn encode_intra_stream(fmt: SourceFormat, n_frames: usize) -> Vec<u8> {
    let (w, h) = match fmt {
        SourceFormat::Qcif => (176usize, 144usize),
        SourceFormat::Cif => (352usize, 288usize),
    };
    let cw = w / 2;
    let mut out = Vec::new();
    for k in 0..n_frames {
        // Force every frame to be an I-picture by recreating the
        // encoder each iteration. That keeps the bench input shape
        // independent of the encoder's GOP heuristics (which are
        // tuned for rate / quality, not for benching start-code
        // density).
        let mut enc_k = H261Encoder::new(fmt, /*quant*/ 12);
        let (y, cb, cr) = synth_yuv(w, h, 0xCAFE_F00D ^ (k as u32).wrapping_mul(0x9E37_79B1));
        let bits = enc_k.encode_frame(&y, w, &cb, cw, &cr, cw).unwrap();
        out.extend_from_slice(&bits);
    }
    out
}

/// Build a 4-KiB pseudo-random buffer that contains no `0x0001` 16-bit
/// prefix at any bit position. Used for the worst-case "scan to end,
/// return None" bench. We synth from a flat xorshift, then sweep the
/// buffer and overwrite any bit-window that happens to match the
/// PSC prefix so the scanner is guaranteed to walk every byte.
fn synth_no_start_code(seed: u32, n_bytes: usize) -> Vec<u8> {
    let mut state = seed;
    let mut buf = vec![0u8; n_bytes];
    for b in buf.iter_mut() {
        *b = (xorshift32(&mut state) >> 24) as u8;
    }
    // Sweep bit-by-bit and clobber any 16-bit `0x0001` matches by
    // setting the bit immediately before the trailing `1` to `1`
    // as well — that breaks the 15-zeros-then-1 prefix without
    // changing the buffer length. We repeat until no matches remain.
    loop {
        let mut found = false;
        let total_bits = (n_bytes as u64) * 8;
        let mut pos: u64 = 0;
        while pos + 16 <= total_bits {
            let mut window: u32 = 0;
            for i in 0..16 {
                let bit_pos = pos + i;
                let byte = (bit_pos / 8) as usize;
                let shift = 7 - (bit_pos % 8) as u8;
                let bit = (buf[byte] >> shift) & 1;
                window = (window << 1) | (bit as u32);
            }
            if window == 0x0001 {
                // Flip the bit at offset +14 from `pos` (= second-
                // to-last bit of the 16-window). That position is
                // the 14th zero; setting it to 1 breaks the prefix.
                let bit_pos = pos + 14;
                let byte = (bit_pos / 8) as usize;
                let shift = 7 - (bit_pos % 8) as u8;
                buf[byte] |= 1 << shift;
                found = true;
            }
            pos += 1;
        }
        if !found {
            break;
        }
    }
    buf
}

fn bench_iter_start_codes(c: &mut Criterion) {
    let mut group = c.benchmark_group("h261_start_code_iter");

    // QCIF I-frame — 1 PSC + 3 GBSCs expected.
    let qcif_one = encode_intra_stream(SourceFormat::Qcif, 1);
    group.throughput(Throughput::Bytes(qcif_one.len() as u64));
    group.bench_function(
        BenchmarkId::new("iter_start_codes", "qcif_intra_one_frame"),
        |b| {
            b.iter(|| {
                let n = iter_start_codes(black_box(&qcif_one)).count();
                black_box(n);
            });
        },
    );

    // CIF I-frame — 1 PSC + 12 GBSCs expected.
    let cif_one = encode_intra_stream(SourceFormat::Cif, 1);
    group.throughput(Throughput::Bytes(cif_one.len() as u64));
    group.bench_function(
        BenchmarkId::new("iter_start_codes", "cif_intra_one_frame"),
        |b| {
            b.iter(|| {
                let n = iter_start_codes(black_box(&cif_one)).count();
                black_box(n);
            });
        },
    );

    // Three concatenated QCIF I-frames — exercises iterator
    // advancement across multiple PSCs.
    let qcif_three = encode_intra_stream(SourceFormat::Qcif, 3);
    group.throughput(Throughput::Bytes(qcif_three.len() as u64));
    group.bench_function(
        BenchmarkId::new("iter_start_codes", "qcif_intra_three_frames"),
        |b| {
            b.iter(|| {
                let n = iter_start_codes(black_box(&qcif_three)).count();
                black_box(n);
            });
        },
    );

    group.finish();
}

fn bench_single_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("h261_start_code_single");

    // Byte-aligned first-PSC scan — best-case (hit at bit 0).
    let qcif_one = encode_intra_stream(SourceFormat::Qcif, 1);
    group.throughput(Throughput::Bytes(qcif_one.len() as u64));
    group.bench_function(
        BenchmarkId::new("find_next_start_code", "qcif_intra_first"),
        |b| {
            b.iter(|| {
                let sc = find_next_start_code(black_box(&qcif_one), 0);
                black_box(sc);
            });
        },
    );

    // Bit-misaligned start — scanner has to walk through the in-byte
    // prefix bits before hitting the PSC at bit 0. The §4.2
    // packetizer's slow path takes this when a GOB ends mid-byte.
    group.bench_function(
        BenchmarkId::new("find_next_start_code_bits", "qcif_intra_misaligned_start"),
        |b| {
            b.iter(|| {
                let sc = find_next_start_code_bits(black_box(&qcif_one), 3);
                black_box(sc);
            });
        },
    );

    // Worst case — 4 KiB buffer scanned end-to-end, no hit anywhere.
    let no_sc = synth_no_start_code(0xDEAD_BEEF, 4096);
    group.throughput(Throughput::Bytes(no_sc.len() as u64));
    group.bench_function(
        BenchmarkId::new("find_next_start_code", "no_start_code_in_buffer"),
        |b| {
            b.iter(|| {
                let sc = find_next_start_code(black_box(&no_sc), 0);
                black_box(sc);
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_iter_start_codes, bench_single_scan);
criterion_main!(benches);
