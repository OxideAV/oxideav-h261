#![no_main]

//! Drive arbitrary fuzz-supplied bytes through the H.261 §5.4 BCH
//! (511, 493) FEC multiframe parser surface — both the raw-bytes
//! lock-and-strip path and an "error-injection" path that first
//! frames a synthetic inner payload and then flips attacker-chosen
//! bits before re-decoding.
//!
//! The BCH framing layer (§5.4) is one of the two attacker-reachable
//! parser surfaces an H.261 endpoint exposes on the wire (the other
//! is the elementary-stream decoder, covered by `decode_h261`; the
//! RTCP control channel parser covered by `parse_rtcp_compound` is a
//! peer interface, not strictly a media path). The lock-search loop
//! sweeps `[0, FRAME_BITS)` candidate offsets and at each looks for
//! 24 framing bits matching `(00011011)^3` at a stride of
//! `FRAME_BITS = 511`. Once locked, every subsequent frame is
//! stripped of its parity, the syndrome is checked, and the inner
//! 492 data bits (or none, for `Fi=0` fill frames) are concatenated
//! into the recovered inner stream.
//!
//! The wire format under test:
//!
//! * **§5.4.4 lock search** — `decode_multiframe` walks 511 candidate
//!   bit offsets times 24 framing-bit reads each. A pathological
//!   input could in principle confuse the early-out / candidate-skip
//!   logic into running off the buffer end, looping forever, or
//!   accepting a false-positive lock; every path must return `None`
//!   or `Some(DecodedMultiframe)` and never panic.
//! * **§5.4.2 GF(2) division** — `parity18` and `syndrome18` are
//!   bit-serial long-division primitives over the 19-bit generator
//!   polynomial `0x495C9`. Both accept attacker-controlled bit
//!   counts; both must run in bounded time and never index out of
//!   the supplied buffer.
//! * **§5.4.3 frame stripping** — once locked, the decoder reads
//!   `Fi`, 492 data bits, and 18 parity bits per frame, then
//!   computes the syndrome. A non-zero syndrome bumps the
//!   `corrupted_frames` counter but never breaks lock; the data is
//!   emitted regardless (per the spec deployment pattern: dropping
//!   the frame creates a longer drop-out than letting the inner
//!   H.261 video VLC layer resync at the next GOB).
//!
//! The contract under test is purely that every parser / primitive
//! call *returns*: a malformed input yields `None` from
//! `decode_multiframe` or simply a junk syndrome from `parity18` /
//! `syndrome18`. No path may panic, abort, integer-overflow (in a
//! debug / ASAN build), index out of bounds, or OOM.
//!
//! ## Two-mode driver
//!
//! Mode A — **raw bytes** — feed `data` straight into
//! `decode_multiframe` / `parity18` / `syndrome18`. Most random
//! inputs fail to obtain frame lock, exercising the §5.4.4
//! lock-search early-out path; the small fraction that do lock
//! drive the per-frame syndrome and data-extraction loop.
//!
//! Mode B — **error injection** — frame a deterministic synthetic
//! payload with the in-crate `encode_multiframe`, then use `data`
//! as an attacker-supplied bit-flip vector to corrupt arbitrary
//! bits in the framed stream before re-decoding. This drives the
//! syndrome-failure path (`corrupted_frames > 0`) and the
//! occasional lock-loss path (a corrupted framing bit in the lock
//! window).
//!
//! Both modes are exercised on every fuzz iteration so neither path
//! starves the other for coverage budget.

use libfuzzer_sys::fuzz_target;
use oxideav_h261::bch::{
    decode_multiframe, decode_multiframe_with_correction, encode_multiframe, locate_single_error,
    parity18, syndrome18, FRAME_BITS, MULTIFRAME_FRAMES,
};

fuzz_target!(|data: &[u8]| {
    // ---- Mode A: raw attacker bytes through every public BCH entry. ----

    // §5.4.4 lock search + per-frame syndrome + data extraction. The
    // overwhelming majority of random inputs return `None` (no lock);
    // a small fraction will spuriously lock on a coincidental
    // `(00011011)^3` and drive the per-frame loop.
    let _ = decode_multiframe(data);

    // Exercise the §5.4.1 single-bit correction path on the same raw
    // attacker bytes. Most inputs won't lock; the small fraction that
    // do drive the per-frame correction loop including
    // `locate_single_error` for every non-zero syndrome encountered.
    let _ = decode_multiframe_with_correction(data);

    // Drive `locate_single_error` on attacker-controlled syndromes
    // directly: read the first 3 bytes as an 18-bit syndrome
    // candidate. The function must return for every input value in
    // 0..2^18 (no panic, no infinite loop). The 511 walked iterations
    // bound the run time independent of the syndrome value.
    let mut sb = [0u8; 3];
    let take = core::cmp::min(data.len(), sb.len());
    sb[..take].copy_from_slice(&data[..take]);
    let synd =
        (((sb[0] as u32) << 16) | ((sb[1] as u32) << 8) | (sb[2] as u32)) & 0x3_FFFF;
    let _ = locate_single_error(synd);

    // `parity18` requires at least 62 bytes (493 bits) of input. The
    // function's debug_assert guards that; in release we still want
    // coverage of the bit-walking loop, so feed it whatever slice we
    // have when long enough, and otherwise pad with zeros on the
    // stack to keep the fuzz iteration cheap.
    let mut padded = [0u8; 62];
    let take = core::cmp::min(data.len(), padded.len());
    padded[..take].copy_from_slice(&data[..take]);
    let par = parity18(&padded);

    // The syndrome of (padded || parity18(padded)) must be zero by
    // construction; we ignore the value but the call exercises the
    // 511-bit shift-register walk against attacker-controlled bytes.
    let _ = syndrome18(&padded, par);

    // Mutated-parity syndrome — exercises the non-zero-syndrome
    // path. Build an attacker-controlled 18-bit parity by reading the
    // first 3 bytes (or zero-padding) of `data`.
    let mut p = [0u8; 3];
    let take = core::cmp::min(data.len(), p.len());
    p[..take].copy_from_slice(&data[..take]);
    let attacker_parity = (((p[0] as u32) << 16) | ((p[1] as u32) << 8) | (p[2] as u32)) & 0x3_FFFF;
    let _ = syndrome18(&padded, attacker_parity);

    // ---- Mode B: error-injection on a well-formed framed stream. ----
    //
    // Frame a small deterministic payload, then use `data` as a
    // bit-flip vector to corrupt arbitrary bits of the framed
    // output. This exercises the §5.4.3 per-frame syndrome path
    // ("non-zero syndrome bumps `corrupted_frames`") and the
    // §5.4.4 lock-loss path (when the attacker flips bits inside the
    // 24-bit lock window).
    //
    // We frame three multiframes' worth of payload — the minimum to
    // obtain lock per §5.4.4. The payload itself is irrelevant for
    // panic-freedom; what matters is that the framed buffer is
    // well-formed before the bit-flips.
    let payload: [u8; 64] = {
        let mut p = [0u8; 64];
        let mut s: u32 = 0x1234_5678;
        for b in p.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (s >> 16) as u8;
        }
        p
    };
    let mut framed = encode_multiframe(&payload, payload.len() * 8);
    framed.extend(encode_multiframe(&[], 0));
    framed.extend(encode_multiframe(&[], 0));
    debug_assert_eq!(framed.len(), 3 * 512);

    // Use up to the first 32 bytes of `data` as bit-flip indices
    // into the framed buffer. Each pair of bytes encodes one bit
    // position mod `framed.len() * 8`, big-endian. Bounding by 32
    // bytes (16 flips) keeps the iteration cheap and matches the
    // §5.4.3 "≤ t = 1 error per 511-bit codeword" deployment
    // assumption (16 flips across 24 frames averages well under 1
    // per frame).
    let cap = core::cmp::min(data.len(), 32);
    let total_bits = framed.len() * 8;
    let mut i = 0;
    while i + 1 < cap {
        let pos = ((data[i] as usize) << 8) | (data[i + 1] as usize);
        let pos = pos % total_bits;
        framed[pos / 8] ^= 1 << (7 - (pos & 7));
        i += 2;
    }
    let _ = decode_multiframe(&framed);
    let _ = decode_multiframe_with_correction(&framed);

    // ---- Mode C: raw bytes truncated / extended across the lock window. ----
    //
    // The lock-search has an early-out `if bit0 + lock_span_bits >
    // total_bits { break; }`. A buffer sized **just** at the lock
    // boundary (3 multiframes = 1536 bytes = 12288 bits) is the edge
    // case where the comparison's strictness matters. Feed a fixed-
    // size truncation of the attacker buffer to cover that boundary
    // even if the raw input is larger or smaller.
    let lock_span_bits = (3 * MULTIFRAME_FRAMES * FRAME_BITS) as usize; // 12_264
    let edge_len = lock_span_bits.div_ceil(8);
    if data.len() >= edge_len {
        let _ = decode_multiframe(&data[..edge_len]);
        let _ = decode_multiframe_with_correction(&data[..edge_len]);
    } else {
        let mut padded_edge = vec![0u8; edge_len];
        padded_edge[..data.len()].copy_from_slice(data);
        let _ = decode_multiframe(&padded_edge);
        let _ = decode_multiframe_with_correction(&padded_edge);
    }
});
