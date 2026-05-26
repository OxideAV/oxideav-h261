#![no_main]

//! Decode arbitrary fuzz-supplied bytes through the H.261 decoder's
//! public surface (`send_packet` -> drain `receive_frame` -> `flush` ->
//! drain again).
//!
//! The contract under test is purely that every call *returns*: a
//! malformed stream yields `Err(oxideav_core::Error::…)`, a well-formed
//! one yields `Ok(())` / `Ok(Frame::Video(…))`, and no path may panic,
//! abort, integer-overflow (in a debug / ASAN build), index out of
//! bounds, or OOM — regardless of how hostile the bytes are.
//!
//! The H.261 attack surface this drives:
//! * PSC (20-bit `0000 0000 0000 0001 0000`) start-code scanner.
//! * Picture header (TR / PTYPE / source-format / PEI / PSPARE loop).
//! * GBSC (16-bit) GOB scanner, GN / GQUANT / GEI / GSPARE loop.
//! * Macroblock layer: MBA VLC (Table 1), MTYPE VLC (Table 2), MQUANT,
//!   MVD VLC (Table 3), CBP VLC (Table 4), §4.2.3.4 MV predictor.
//! * Block layer: TCOEFF VLC (Table 5) + EOB + 20-bit escape, zigzag,
//!   dequantisation, 8x8 IDCT, INTRA DC FLC (Table 6).
//! * Integer-pel motion compensation (range +-15 pels) and chroma
//!   subsampling via `mvluma / 2` truncated toward zero.
//! * Loop filter (§3.2.3) — separable `1/4, 1/2, 1/4`.
//!
//! The harness deliberately leaves [`DecoderLimits`] at default so the
//! crate's own DoS caps are exercised (`max_pixels_per_frame`,
//! `max_alloc_bytes_per_frame`, `max_arenas_in_flight`) — they should
//! surface as `Err(ResourceExhausted)` / `Err(InvalidData)` rather than
//! the fuzzer's allocator OOM-killing the process.
//!
//! The return values are intentionally discarded; a round-trip oracle
//! would need a trusted *decoder* of the *same* arbitrary bytes, which
//! does not exist in this codebase (the workspace's clean-room wall
//! bars any external H.261 codec source as a cross-decode oracle).

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, Decoder, Packet, TimeBase};
use oxideav_h261::decoder::H261Decoder;

fuzz_target!(|data: &[u8]| {
    let mut dec = H261Decoder::new(CodecId::new("h261"));

    // Single-packet path. Most real-world callers feed an entire
    // elementary stream as one packet (the §4.2.1 PSC scanner runs
    // over the whole buffer).
    let pkt = Packet::new(0, TimeBase::new(1, 30), data.to_vec());
    let _ = dec.send_packet(&pkt);
    // Bounded drain — `receive_frame` returns `Err(NeedMore)` /
    // `Err(Eof)` once the queue empties; the cap keeps a malicious
    // input from looping forever via an arena-pool ping-pong.
    for _ in 0..32 {
        if dec.receive_frame().is_err() {
            break;
        }
    }
    // Flush emits any tail-buffered picture (an H.261 stream has no
    // explicit end-of-sequence marker, so the decoder waits for either
    // the next PSC or EOF; flush is what tells it EOF arrived).
    let _ = dec.flush();
    for _ in 0..32 {
        if dec.receive_frame().is_err() {
            break;
        }
    }
});
