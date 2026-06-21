#![no_main]

//! Drive arbitrary fuzz-supplied bytes through the H.261 crate's RTP
//! *packetiser* surface — the send/forward path that parses an H.261
//! elementary stream's VLC layer to split it into RTP payloads.
//!
//! The four other surfaces are covered elsewhere: the elementary-stream
//! decoder (`decode_h261`), the RTCP control channel
//! (`parse_rtcp_compound`), the BCH §5.4 FEC framer
//! (`decode_bch_multiframe`), and the RTP *receive* path —
//! `parse_rtp_fixed_header` / `unpack_header` / `depacketize`
//! (`parse_rtp_payload`). This target is the complementary RTP *send*
//! path:
//!
//! * **RFC 4587 §4.2 GOB-aligned packetiser** — `packetize_gob_aligned`
//!   scans the stream for byte-aligned start codes (PSC / GBSC) and
//!   emits one payload per GOB-sized run. It is total over arbitrary
//!   bytes (returns an empty vector when no start code is found).
//! * **RFC 4587 §4.2 RECOMMENDED MB-level fragmentation** —
//!   `packetize_mb_fragmented` walks the *Huffman / VLC macroblock
//!   layer* (`walk_mb_split_points`: picture header, GOB header, MBA /
//!   MTYPE / MQUANT / MVD / CBP VLCs, TCOEFF VLC + escape) to find
//!   macroblock-boundary split points, so an oversize GOB can be
//!   fragmented mid-GOB with full §4.1 GOBN/MBAP/QUANT/HMVD/VMVD
//!   continuation context. This is the richest attacker-reachable parse
//!   on the send side: an MCU / forwarder re-fragmenting an *untrusted*
//!   sender's bitstream runs this VLC walk over bytes it did not
//!   produce. It returns `Result`; a malformed stream must surface as
//!   `Err(RtpError::…)`, never a panic / OOB / overflow.
//! * **`RtpPacketizer::pack_frame`** — the session-stateful glue that
//!   routes a frame through one of the two packetisers (selected by
//!   `with_mb_fragmentation`) and wraps each payload in a full RFC 3550
//!   §5.1 fixed header, tracking running packet / octet counts.
//!
//! ## Round-trip oracle
//!
//! Unlike `decode_h261` (no trusted external decoder exists for a
//! cross-check), the packetiser *does* have an in-crate inverse: the
//! `depacketize` receive-path walker. When the GOB-aligned packetiser
//! produces a non-empty payload sequence from a stream that itself
//! parses, re-running `depacketize` over those payloads must also just
//! *return* — exercising the pack → unpack seam end-to-end without
//! asserting byte-equality (GOB-aligned packetisation is lossy w.r.t.
//! the exact stuffing bits, so an equality oracle would be unsound).
//!
//! The contract under test is purely that every call *returns*: no
//! path may panic, abort, integer-overflow (debug / ASAN), index out
//! of bounds, or OOM — regardless of how hostile the bytes are. The
//! documented `assert!` preconditions on `max_payload` /
//! `max_rtp_packet_size` (the packet must hold the headers + ≥ 1 data
//! byte) are *not* fuzzed: those are caller-contract violations, so the
//! harness only ever feeds budgets above the documented floor.

use libfuzzer_sys::arbitrary::{self, Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use oxideav_h261::rtp::{
    depacketize, packetize_gob_aligned, packetize_mb_fragmented, RtpPacketizer, HEADER_LEN,
    RTP_FIXED_HEADER_LEN,
};

/// A structured input: packetiser configuration + the elementary-stream
/// bytes to fragment.
#[derive(Debug)]
struct Plan {
    /// Inner-payload budget for the two bare packetisers. Kept strictly
    /// above `HEADER_LEN` (the documented `assert!` floor) so we fuzz
    /// behaviour, not the precondition. Small values force the maximum
    /// number of fragments / the deepest VLC-walk split search.
    inner_budget: usize,
    /// Whole-packet budget for `RtpPacketizer`. Kept strictly above
    /// `RTP_FIXED_HEADER_LEN + HEADER_LEN` for the same reason.
    rtp_budget: usize,
    intra_only: bool,
    motion_vectors: bool,
    mb_fragmentation: bool,
    pt: u8,
    ssrc: u32,
    seq: u16,
    ts: u32,
    /// The H.261 elementary stream to packetise.
    stream: Vec<u8>,
}

impl<'a> Arbitrary<'a> for Plan {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // Budgets span from the smallest legal value (maximal
        // fragmentation pressure) up to a few KiB. `+ 1` keeps them
        // strictly above the documented `assert!` floor.
        let inner_budget = HEADER_LEN + 1 + u.int_in_range(0usize..=4096)?;
        let rtp_budget = RTP_FIXED_HEADER_LEN + HEADER_LEN + 1 + u.int_in_range(0usize..=4096)?;
        let intra_only = bool::arbitrary(u)?;
        let motion_vectors = bool::arbitrary(u)?;
        let mb_fragmentation = bool::arbitrary(u)?;
        let pt = u8::arbitrary(u)?;
        let ssrc = u32::arbitrary(u)?;
        let seq = u16::arbitrary(u)?;
        let ts = u32::arbitrary(u)?;

        let remaining = u.len();
        let stream = u.bytes(remaining)?.to_vec();

        Ok(Plan {
            inner_budget,
            rtp_budget,
            intra_only,
            motion_vectors,
            mb_fragmentation,
            pt,
            ssrc,
            seq,
            ts,
            stream,
        })
    }
}

fuzz_target!(|plan: Plan| {
    let Plan {
        inner_budget,
        rtp_budget,
        intra_only,
        motion_vectors,
        mb_fragmentation,
        pt,
        ssrc,
        seq,
        ts,
        stream,
    } = plan;

    // ---- GOB-aligned packetiser (total over arbitrary bytes). ----
    let gob_payloads = packetize_gob_aligned(&stream, inner_budget, intra_only, motion_vectors);

    // ---- Round-trip seam: re-run the receive-path walker over the
    // payloads the GOB packetiser just produced. Must also just return;
    // no byte-equality assertion (GOB packetisation is lossy w.r.t.
    // exact stuffing bits, so an equality oracle would be unsound). ----
    if !gob_payloads.is_empty() {
        let _ = depacketize(&gob_payloads);
    }

    // ---- MB-level fragmentation: the VLC-layer walk over attacker
    // bytes. Returns Result; a malformed stream is `Err`, not a panic. ----
    let _ = packetize_mb_fragmented(&stream, inner_budget, intra_only, motion_vectors);

    // ---- Session-stateful packetiser, both fragmentation modes. ----
    let mut pktzr = RtpPacketizer::new(pt, ssrc, seq, rtp_budget)
        .with_intra_only(intra_only)
        .with_motion_vectors(motion_vectors)
        .with_mb_fragmentation(mb_fragmentation);
    // Pack the same stream twice so the running packet / octet / seq
    // counters advance across a frame boundary (wraparound arithmetic
    // on `packet_count` / `octet_count` / `next_seq` is part of the
    // surface under test).
    let _ = pktzr.pack_frame(&stream, ts);
    let _ = pktzr.pack_frame(&stream, ts.wrapping_add(3000));
});
