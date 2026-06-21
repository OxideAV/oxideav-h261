#![no_main]

//! Decode arbitrary fuzz-supplied bytes through the H.261 decoder's
//! public surface (`send_packet` -> drain (`receive_frame` /
//! `receive_arena_frame`) -> `flush` -> drain again).
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
//! # Beyond the single-packet path
//!
//! Earlier this target fed the whole input as one packet. Real callers
//! (an RTP depacketiser, a container demux) hand the elementary stream
//! to the decoder in *fragments*, so the §4.2.1 PSC scanner and the
//! picture-buffering state machine must survive a start code that
//! straddles a `send_packet` boundary, an empty packet, a packet that
//! completes a picture begun in a prior packet, and a `flush` arriving
//! mid-picture. This harness therefore lets the fuzzer:
//!
//! * choose tight [`DecoderLimits`] (so the crate's own DoS caps —
//!   `max_pixels_per_frame`, `max_alloc_bytes_per_frame`,
//!   `max_arenas_in_flight`, `max_alloc_count_per_frame` — are
//!   exercised on the *small* side, where off-by-one rejection bugs
//!   live, not just at the generous default);
//! * split the remaining bytes into an arbitrary number of packets at
//!   arbitrary offsets (driving the cross-packet buffer-accumulation
//!   path);
//! * drain via either `receive_frame` (heap `VideoFrame`
//!   materialisation) or `receive_arena_frame` (zero-copy arena lease),
//!   so both output surfaces — and the arena-pool lease/return cycle —
//!   are fuzzed.
//!
//! The default-limits whole-stream path is still covered: when the
//! fuzzer picks one packet and default limits, this reduces to the old
//! behaviour.
//!
//! The return values are intentionally discarded; a round-trip oracle
//! would need a trusted *decoder* of the *same* arbitrary bytes, which
//! does not exist in this codebase (the workspace's clean-room wall
//! bars any external H.261 codec source as a cross-decode oracle).

use libfuzzer_sys::arbitrary::{self, Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use oxideav_core::limits::DecoderLimits;
use oxideav_core::{CodecId, Decoder, Packet, TimeBase};
use oxideav_h261::decoder::H261Decoder;

/// How a single drain step pulls from the decoder. The fuzzer picks one
/// per drain so both output materialisation paths get exercised.
#[derive(Debug)]
enum DrainKind {
    /// Heap-backed `VideoFrame` (per-plane `to_vec`).
    Frame,
    /// Zero-copy arena lease (`arena::sync::Frame`), held briefly then
    /// dropped to return the buffer to the pool.
    Arena,
}

/// A structured fuzz input: a set of DoS caps, a packet-fragmentation
/// plan, a drain plan, and the raw elementary-stream bytes.
#[derive(Debug)]
struct Plan {
    limits: DecoderLimits,
    /// Byte counts for successive `send_packet` calls; the tail (any
    /// bytes left after these are consumed) goes in a final packet.
    splits: Vec<u16>,
    /// Drain kinds cycled through between/after packets.
    drains: Vec<DrainKind>,
    /// The elementary-stream payload.
    stream: Vec<u8>,
}

impl<'a> Arbitrary<'a> for Plan {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // Caps deliberately span the tight end of each range so the
        // decoder's rejection paths are reached, while never being so
        // large that a single picture OOMs the fuzzer host.
        // `DecoderLimits` is `#[non_exhaustive]`, so start from the
        // default and mutate the caps the decoder reads.
        let mut limits = DecoderLimits::default();
        limits.max_pixels_per_frame = u.int_in_range(0u64..=200_000)?;
        limits.max_alloc_bytes_per_frame = u.int_in_range(0u64..=512 * 1024)?;
        limits.max_alloc_count_per_frame = u.int_in_range(0u32..=64)?;
        limits.max_arenas_in_flight = u.int_in_range(1u8..=8)?;

        let n_splits = u.int_in_range(0usize..=8)?;
        let mut splits = Vec::with_capacity(n_splits);
        for _ in 0..n_splits {
            splits.push(u16::arbitrary(u)?);
        }

        let n_drains = u.int_in_range(1usize..=6)?;
        let mut drains = Vec::with_capacity(n_drains);
        for _ in 0..n_drains {
            drains.push(if bool::arbitrary(u)? {
                DrainKind::Arena
            } else {
                DrainKind::Frame
            });
        }

        // Whatever is left is the bitstream the decoder actually parses.
        let remaining = u.len();
        let stream = u.bytes(remaining)?.to_vec();

        Ok(Plan {
            limits,
            splits,
            drains,
            stream,
        })
    }
}

/// Drain at most `cap` frames, cycling through the planned drain kinds.
/// An arena frame is dropped immediately so its buffer returns to the
/// pool before the next lease.
fn drain(dec: &mut H261Decoder, drains: &[DrainKind], cap: usize) {
    for i in 0..cap {
        match &drains[i % drains.len()] {
            DrainKind::Frame => {
                if dec.receive_frame().is_err() {
                    break;
                }
            }
            DrainKind::Arena => match dec.receive_arena_frame() {
                Ok(frame) => drop(frame),
                Err(_) => break,
            },
        }
    }
}

fuzz_target!(|plan: Plan| {
    let Plan {
        limits,
        splits,
        drains,
        stream,
    } = plan;

    let mut dec = H261Decoder::with_limits(CodecId::new("h261"), limits);
    let tb = TimeBase::new(1, 30);

    // Feed the stream in fragments at the planned offsets. A start code
    // that straddles two of these packets exercises the cross-packet
    // buffer-accumulation path (the whole reason this is fragmented).
    let mut pos = 0usize;
    for &len in &splits {
        let end = (pos + len as usize).min(stream.len());
        let pkt = Packet::new(0, tb, stream[pos..end].to_vec());
        let _ = dec.send_packet(&pkt);
        // Bounded interleaved drain — a malicious input must not loop
        // forever via an arena-pool ping-pong.
        drain(&mut dec, &drains, 4);
        pos = end;
        if pos >= stream.len() {
            break;
        }
    }
    // Final packet carries whatever is left (and is the *only* packet
    // when `splits` is empty — the original whole-stream path).
    if pos < stream.len() || splits.is_empty() {
        let pkt = Packet::new(0, tb, stream[pos..].to_vec());
        let _ = dec.send_packet(&pkt);
    }
    drain(&mut dec, &drains, 32);

    // Flush emits any tail-buffered picture (an H.261 stream has no
    // explicit end-of-sequence marker, so the decoder waits for either
    // the next PSC or EOF; flush is what tells it EOF arrived).
    let _ = dec.flush();
    drain(&mut dec, &drains, 32);
});
