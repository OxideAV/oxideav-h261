//! Criterion benchmarks for the H.261 §5.4 BCH (511, 493) FEC layer.
//!
//! Round 233 (depth-mode benchmark). The §5.4 outer-coding layer is
//! a spec-mandated framing on top of every elementary H.261 stream
//! that travels over a synchronous serial link (ISDN B-channel, V.35,
//! H.221 PX64 multiplex). Three of its primitives sit in inner loops:
//!
//! * `parity18` — 511-step bit-serial GF(2) long division over the
//!   19-bit generator polynomial `0x495C9`. Called once per emitted
//!   §5.4.3 frame on the encoder side.
//! * `syndrome18` — same shift-register, called once per received
//!   frame on the decoder side.
//! * `locate_single_error` — §5.4.1 `t = 1` correction walk: marches
//!   `pow = x^i mod g(x)` for `i = 0..511` until it matches the
//!   non-zero syndrome. Called once per corrupted frame on the
//!   correcting decode path (`decode_multiframe_with_correction`,
//!   landed in round 230).
//!
//! The integrated `encode_multiframe` / `decode_multiframe` /
//! `decode_multiframe_with_correction` entry points wrap the above
//! primitives with the §5.4.4 24-bit lock-search and per-frame Fi /
//! data / parity extraction. Their per-multiframe cost is what an
//! actual H.261-over-ISDN endpoint pays per ~512 bytes of FEC frame.
//!
//! Scenarios:
//!
//!   - **parity18_one_frame** — one 493-bit message through the
//!     encoder long-division primitive. Throughput in elements is
//!     reported per data bit (493) so a per-bit cost lands naturally
//!     alongside the §5.4.3 frame-rate budget.
//!   - **syndrome18_one_frame** — one received 511-bit codeword
//!     through the decoder long-division primitive. Same throughput
//!     normalisation as `parity18_one_frame`.
//!   - **locate_single_error_worst_case** — the §5.4.1 correction
//!     walk on the worst-case syndrome value: one whose match
//!     position is the last of the 511 candidate `x^i`. This is the
//!     ceiling cost for a corrupted-but-correctable frame.
//!   - **locate_single_error_uncorrectable** — the same walk on a
//!     syndrome that matches no single-bit error pattern (weight ≥ 2
//!     error). Exercises the full 511 iterations followed by the
//!     `None` return.
//!   - **encode_multiframe_8frames** — one full §5.4.4 multiframe
//!     (8 frames × 511 bits = 4088 bits, packed to 512 bytes) with a
//!     realistic-density coded inner stream. Includes the per-frame
//!     `parity18` call.
//!   - **decode_multiframe_8frames_clean** — the detection-only
//!     decode path on a clean (zero-syndrome) multiframe. Exercises
//!     the §5.4.4 lock-search early-out, per-frame `syndrome18`, and
//!     the Fi / data-extraction loop. No correction work.
//!   - **decode_multiframe_8frames_one_bit_corrupted** — same as
//!     above but with one bit flipped in the middle of a data field;
//!     drives the syndrome-failure-without-correction path
//!     (`corrupted_frames > 0`, `corrected_frames = 0`).
//!   - **decode_multiframe_with_correction_8frames_one_bit** — the
//!     §5.4.1 correcting decoder on the same one-bit-corrupted
//!     stream. Adds the `locate_single_error` cost on the corrupted
//!     frame on top of the per-frame syndrome cost across the whole
//!     multiframe. This is the apples-to-apples A/B against
//!     `decode_multiframe_8frames_one_bit_corrupted` and makes the
//!     correction overhead measurable.
//!
//! All inputs are synthesised in-bench from a deterministic xorshift
//! seed — no committed fixtures, no `docs/` files, no third-party
//! tools. Run with:
//!     cargo bench -p oxideav-h261 --bench bch

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_h261::bch::{
    decode_multiframe, decode_multiframe_with_correction, encode_multiframe, locate_single_error,
    parity18, syndrome18, DATA_BITS, FRAME_BITS, MULTIFRAME_FRAMES, PARITY_BITS,
};

/// Cheap deterministic xorshift32 — synthesises a "realistic-density"
/// inner coded bitstream. Real H.261 inner data is a packed bitstream
/// of VLC codes whose bit-density is well-approximated by a fair coin;
/// xorshift is a fast PRNG that matches that statistic without a
/// committed fixture.
fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Build a 62-byte buffer (≥ 493 bits) suitable as input to
/// `parity18` / `syndrome18`. The first 493 bits are pseudo-random;
/// the trailing bits up to byte alignment are zero.
fn build_message_62b(seed: u32) -> [u8; 62] {
    let mut state = seed;
    let mut out = [0u8; 62];
    for b in out.iter_mut() {
        *b = (xorshift32(&mut state) >> 24) as u8;
    }
    // Mask off the bits beyond 493 (positions 493..496 in byte 61)
    // so the input represents exactly a 493-bit message zero-padded
    // to byte alignment. This matches `parity18`'s contract.
    let tail_bits_to_keep = (DATA_BITS as usize) - 8 * 61; // 5
    let mask = 0xFFu8 << (8 - tail_bits_to_keep);
    out[61] &= mask;
    out
}

/// Build an arbitrary-length pseudo-random byte buffer to feed
/// `encode_multiframe` as the inner coded stream.
fn build_inner_bits(seed: u32, n_bytes: usize) -> Vec<u8> {
    let mut state = seed;
    let mut out = vec![0u8; n_bytes];
    for b in out.iter_mut() {
        *b = (xorshift32(&mut state) >> 24) as u8;
    }
    out
}

/// Walk `pow = x^i mod g(x)` to step `i = target_i` exactly, then
/// return its value. Used to construct a syndrome that
/// `locate_single_error` finds at the last possible iteration
/// (worst-case input).
fn x_pow_i_mod_g(target_i: u32) -> u32 {
    // Mirrors the in-crate walk: same generator polynomial, same
    // shift-XOR step. Implemented locally so the bench input
    // construction doesn't reach back into the function under test.
    let gen_poly: u32 = 0x4_95C9;
    let mask: u32 = (1u32 << PARITY_BITS) - 1;
    let mut pow: u32 = 1;
    for _ in 0..target_i {
        pow <<= 1;
        if (pow >> PARITY_BITS) & 1 != 0 {
            pow ^= gen_poly;
        }
        pow &= mask;
    }
    pow
}

fn bench_primitives(c: &mut Criterion) {
    let mut group = c.benchmark_group("bch_primitives");
    // Throughput in bits across the 493-bit input message.
    group.throughput(Throughput::Elements(DATA_BITS as u64));

    let msg = build_message_62b(0xCAFE_F00D);

    group.bench_function(BenchmarkId::new("parity18", "one_frame"), |b| {
        b.iter(|| {
            let par = parity18(black_box(&msg));
            black_box(par);
        });
    });

    let parity = parity18(&msg);
    group.bench_function(BenchmarkId::new("syndrome18", "one_frame"), |b| {
        b.iter(|| {
            let s = syndrome18(black_box(&msg), black_box(parity));
            black_box(s);
        });
    });

    group.finish();

    // Locate-single-error is a fixed-cost walk; report in
    // (511 candidate positions / call), independent of input size.
    let mut group = c.benchmark_group("bch_correction");
    group.throughput(Throughput::Elements((FRAME_BITS - 1) as u64));

    // Worst-case match: the syndrome is x^510 mod g(x), the very last
    // candidate the walk inspects before returning Some.
    let worst_case = x_pow_i_mod_g(FRAME_BITS - 2);
    group.bench_function(BenchmarkId::new("locate_single_error", "worst_case"), |b| {
        b.iter(|| {
            let p = locate_single_error(black_box(worst_case));
            black_box(p);
        });
    });

    // Uncorrectable: a syndrome that doesn't match any of the 511
    // x^i mod g(x) values (i.e. weight ≥ 2). We use 0x3_FFFE — a
    // generic 18-bit value not on the orbit (verified by the
    // returned None below in debug; the benchmark relies on the
    // None return causing the full 511-step walk).
    let unc: u32 = 0x3_FFFE;
    debug_assert!(locate_single_error(unc).is_none());
    group.bench_function(
        BenchmarkId::new("locate_single_error", "uncorrectable"),
        |b| {
            b.iter(|| {
                let p = locate_single_error(black_box(unc));
                black_box(p);
            });
        },
    );

    group.finish();
}

fn bench_multiframe(c: &mut Criterion) {
    let mut group = c.benchmark_group("bch_multiframe");

    // One §5.4.4 multiframe = 8 frames × 492 inner data bits = 3936
    // inner bits. The xorshift inner stream is 492 bytes (3936 bits)
    // so the encoder packs it cleanly into one multiframe.
    let inner_bits = (MULTIFRAME_FRAMES * (DATA_BITS - 1)) as usize; // 3936
    let inner_bytes = inner_bits.div_ceil(8); // 492
    let inner = build_inner_bits(0xDEAD_BEEF, inner_bytes);

    group.throughput(Throughput::Elements(MULTIFRAME_FRAMES as u64));

    group.bench_function(BenchmarkId::new("encode_multiframe", "8frames"), |b| {
        b.iter(|| {
            let framed = encode_multiframe(black_box(&inner), black_box(inner_bits));
            black_box(framed);
        });
    });

    // For decode benches, pad the framed buffer with two extra
    // multiframes so the lock-search has the full §5.4.4 24-bit
    // window plus per-multiframe progress. `decode_multiframe`
    // needs at least three repetitions of the 8-bit alignment
    // pattern (`S1..S8` once per multiframe) to lock.
    let mut framed_clean = encode_multiframe(&inner, inner_bits);
    framed_clean.extend(encode_multiframe(&[], 0));
    framed_clean.extend(encode_multiframe(&[], 0));

    group.bench_function(BenchmarkId::new("decode_multiframe", "clean"), |b| {
        b.iter(|| {
            let dec = decode_multiframe(black_box(&framed_clean));
            black_box(dec);
        });
    });

    // One-bit corruption: flip the bit at the centre of the
    // first multiframe's first data field. Position picked so it
    // lands inside the §5.4.3 492 protected data bits (not on the
    // alignment pattern or in the parity tail), which is the
    // realistic deployment pattern for an error reaching the FEC.
    let mut framed_one_bit = framed_clean.clone();
    let bit_pos: usize = 2 + 246; // S + Fi (2) + ~half-way through 492 data bits
    framed_one_bit[bit_pos / 8] ^= 1 << (7 - (bit_pos & 7));

    group.bench_function(
        BenchmarkId::new("decode_multiframe", "one_bit_corrupted"),
        |b| {
            b.iter(|| {
                let dec = decode_multiframe(black_box(&framed_one_bit));
                black_box(dec);
            });
        },
    );

    group.bench_function(
        BenchmarkId::new("decode_multiframe_with_correction", "one_bit_corrupted"),
        |b| {
            b.iter(|| {
                let dec = decode_multiframe_with_correction(black_box(&framed_one_bit));
                black_box(dec);
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_primitives, bench_multiframe);
criterion_main!(benches);
