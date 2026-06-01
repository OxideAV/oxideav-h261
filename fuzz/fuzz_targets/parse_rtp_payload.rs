#![no_main]

//! Drive arbitrary fuzz-supplied bytes through the H.261 crate's RTP
//! payload parser surface — the network-receive path that an H.261
//! endpoint exposes to a remote sender.
//!
//! Unlike the H.261 elementary-stream decoder (covered by `decode_h261`)
//! and the RTCP control-channel parser (covered by `parse_rtcp_compound`)
//! and the BCH §5.4 FEC framer (covered by `decode_bch_multiframe`), this
//! target focuses on the **RTP data-path** parsers an endpoint runs on
//! every received UDP datagram before any of the H.261 bitstream layers
//! see a byte:
//!
//! * **RFC 3550 §5.1 fixed-header parser** — `parse_rtp_fixed_header`
//!   reads the 12-byte fixed header (V / P / X / CC / M / PT / seq / ts /
//!   SSRC), bounds-checks the CSRC list (4 bytes per CSRC, count drawn
//!   from the attacker-controlled CC field), and hands back a slice
//!   pointing past the header + CSRC list. CC ∈ 0..=15 means up to 60
//!   bytes of CSRC tail; a buffer shorter than `12 + 4*CC` must surface
//!   as `Err(ShortHeader)`, not a panic or out-of-bounds read.
//! * **RFC 4587 §4.1 H.261 payload-header parser** — `unpack_header`
//!   reads the 4-byte payload header (SBIT / EBIT / I / V / GOBN / MBAP
//!   / QUANT / HMVD / VMVD), sign-extends two 5-bit MV deltas, and
//!   returns a slice pointing at the inner H.261 bitstream bytes. A
//!   buffer shorter than 4 bytes must surface as `Err(ShortHeader)`.
//! * **Multi-packet depacketiser** — `depacketize` walks a sequence of
//!   `H261RtpPayload`s (each carrying its parsed 4-byte header plus a
//!   bytes buffer that holds header + inner data), honours per-packet
//!   SBIT/EBIT bit-alignment via a slow-path bit-walker, and finally
//!   asserts the recovered elementary stream contains at least one
//!   start code (PSC or GBSC). Attacker-supplied SBIT/EBIT pairs can
//!   push `sbit + ebit >= 8*data.len()` and must surface as
//!   `Err(EmptyPayload)`, never an integer underflow.
//!
//! The wire format under test:
//!
//! ```text
//!   [RFC 3550 §5.1 fixed header (12 B + 0..=60 B CSRC)]
//!   [RFC 4587 §4.1 H.261 payload header (4 B)]
//!   [H.261 elementary-stream slice (0..N B, MSB-aligned per SBIT/EBIT)]
//! ```
//!
//! The contract under test is purely that every parser call *returns*:
//! a malformed datagram yields `Err(RtpError::…)`; a well-formed one
//! yields `Ok((…, slice))`. No path may panic, abort, integer-overflow
//! (in a debug / ASAN build), index out of bounds, or OOM — regardless
//! of how hostile the bytes are.
//!
//! ## Three-mode driver
//!
//! Mode A — **raw datagram** — feed `data` straight into
//! `parse_rtp_fixed_header`, then if a valid fixed header was parsed,
//! feed the post-CSRC slice through `unpack_header`. The vast majority
//! of random inputs fail at the version-2 check; the small fraction
//! that pass exercise the full CSRC bounds-check and the H.261
//! payload-header parser.
//!
//! Mode B — **standalone H.261 payload header** — feed `data` straight
//! into `unpack_header` so the §4.1 header parser is driven against
//! attacker bytes even when the §5.1 RTP wrapper is malformed.
//!
//! Mode C — **multi-packet depacketise** — split `data` into multiple
//! synthetic `H261RtpPayload`s (each carrying a header parsed from
//! attacker bytes plus an attacker-controlled inner slice), then run
//! the `depacketize` bit-walker. This drives the §4.1 SBIT/EBIT
//! bit-alignment path against attacker-chosen offset combinations and
//! exercises the final `iter_start_codes` sanity check.
//!
//! All three modes run on every iteration so neither path starves the
//! other for coverage budget.

use libfuzzer_sys::fuzz_target;
use oxideav_h261::rtp::{
    depacketize, parse_rtp_fixed_header, unpack_header, H261RtpHeader, H261RtpPayload, HEADER_LEN,
    RTP_FIXED_HEADER_LEN,
};

fuzz_target!(|data: &[u8]| {
    // ---- Mode A: full RFC 3550 §5.1 + RFC 4587 §4.1 walk on raw bytes. ----

    if let Ok((_, post_fixed)) = parse_rtp_fixed_header(data) {
        // Post-fixed slice already accounts for the CSRC list per
        // `parse_rtp_fixed_header`'s contract.
        let _ = unpack_header(post_fixed);
    }

    // ---- Mode B: §4.1 H.261 payload header parser in isolation. ----

    // The standalone path covers the slice of inputs whose §5.1 wrapper
    // doesn't parse (e.g. version != 2) but whose tail happens to look
    // like a §4.1 H.261 header.
    let _ = unpack_header(data);

    // ---- Mode C: multi-packet depacketise with attacker-chosen splits. ----
    //
    // The depacketiser walks a sequence of `H261RtpPayload`s, each
    // bringing its own (sbit, ebit, hmvd, vmvd, …) header and a bytes
    // buffer whose first 4 bytes are the §4.1 header and whose tail is
    // the inner H.261 data. The slow-path bit-walker activates whenever
    // (sbit != 0 || ebit != 0 || mid-packet bit boundary); driving it
    // exhaustively means we need a few packets and a non-zero sbit/ebit
    // mix.
    //
    // We carve `data` into up to four payload slices. The carve uses
    // attacker bytes for both the split positions and the per-packet
    // header bits, so the fuzzer drives the entire shape of the
    // synthetic payload sequence.
    //
    // We cap at 4 packets and 8 KiB per packet to keep each iteration
    // bounded. `depacketize` rejects sequences shorter than HEADER_LEN
    // per packet and rejects depacketised streams that contain no
    // start codes — neither rejection is a panic.

    if data.len() >= 8 {
        let split_seed = data[0];
        let n_packets = ((split_seed >> 6) & 0x3) as usize + 1; // 1..=4
        let mut payloads: Vec<H261RtpPayload> = Vec::with_capacity(n_packets);
        // After the seed byte the rest of `data` carries the per-packet
        // (header-seed, data-len-seed, data-bytes...) tuples.
        let mut cursor = 1usize;
        for pkt_idx in 0..n_packets {
            if cursor + 3 >= data.len() {
                break;
            }
            // Per-packet header seed: 3 attacker bytes mapped onto the
            // legal sub-fields of `H261RtpHeader`. This keeps `pack_header`
            // happy without removing fuzzer reach into the bit-walker.
            let s0 = data[cursor];
            let s1 = data[cursor + 1];
            let s2 = data[cursor + 2];
            cursor += 3;

            let sbit = s0 & 0x07;
            let ebit = (s0 >> 3) & 0x07;
            let intra_only = (s0 & 0x40) != 0;
            let motion_vectors = (s0 & 0x80) != 0;
            let gobn = s1 & 0x0F;
            let mbap = (s1 >> 4) & 0x1F;
            // Cap MBAP at the 5-bit field max (bits stradle byte boundary
            // above; mask down explicitly).
            let mbap = mbap & 0x1F;
            let quant = s2 & 0x1F;
            // Per RFC 4587 §4.1, HMVD/VMVD are 5-bit two's-complement and
            // -16 is forbidden. We use the safe legal range [-15, 15]
            // sourced from the seed byte so `pack_header` doesn't reject
            // the synthesized header (`pack_header` returning an error
            // would short-circuit the bit-walker we're trying to fuzz).
            let hmvd = ((s2 >> 5) as i8 & 0x07) - 4; // -4..=3
            let vmvd = ((s1 >> 1) as i8 & 0x07) - 4; // -4..=3

            let hdr = H261RtpHeader {
                sbit,
                ebit,
                intra_only,
                motion_vectors,
                gobn,
                mbap,
                quant,
                hmvd,
                vmvd,
            };

            // Data-length seed (1 byte) → 0..=255 bytes of inner data;
            // capped against the remaining buffer.
            if cursor >= data.len() {
                break;
            }
            let want_data = data[cursor] as usize;
            cursor += 1;
            let take = core::cmp::min(want_data, data.len().saturating_sub(cursor));
            let data_slice = &data[cursor..cursor + take];
            cursor += take;

            // Pack the §4.1 header so the depacketiser's `unpack_header`
            // can parse it back. We then carry the same (header, bytes)
            // pair through `H261RtpPayload`.
            let header_bytes = match oxideav_h261::rtp::pack_header(&hdr) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut bytes = Vec::with_capacity(HEADER_LEN + data_slice.len());
            bytes.extend_from_slice(&header_bytes);
            bytes.extend_from_slice(data_slice);

            payloads.push(H261RtpPayload {
                header: hdr,
                bytes,
                marker: pkt_idx + 1 == n_packets,
            });
        }
        // Drive the depacketiser. Empty sequence is allowed; the slow
        // bit-walker activates as soon as any payload sets sbit != 0.
        let _ = depacketize(&payloads);
    }

    // ---- Mode D: edge slice on the §5.1 minimum-buffer boundary. ----
    //
    // `parse_rtp_fixed_header` requires `buf.len() >= 12`, then bumps
    // the requirement to `12 + 4*CC`. A buffer sized exactly at one of
    // those boundaries (12, 13, 16, 28, …) is the canonical place a
    // version-2 + CC=0..=15 walk can mis-bound-check. Drive a fixed
    // 12-byte truncation when the input is long enough so the boundary
    // is always exercised.

    if data.len() >= RTP_FIXED_HEADER_LEN {
        let _ = parse_rtp_fixed_header(&data[..RTP_FIXED_HEADER_LEN]);
    }
});
