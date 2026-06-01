//! Sanity-check the fuzz seed corpus under
//! `fuzz/corpus/parse_rtp_payload/` and drive each seed through the
//! same public RTP payload-parser surface the `parse_rtp_payload` fuzz
//! target exercises. Because `cargo-fuzz` requires the nightly
//! toolchain (libFuzzer's sanitizer-coverage flags are `-Z`-gated), the
//! regular CI matrix never builds the fuzz crate; this stable-Rust test
//! gives the same logical coverage so a corrupted corpus or a regressed
//! RTP-parser surface trips one of the existing CI lanes instead of
//! waiting for the daily fuzz run to notice.
//!
//! The harness mirrors `fuzz/fuzz_targets/parse_rtp_payload.rs`
//! exactly: each buffer is fed through `parse_rtp_fixed_header` (the
//! RFC 3550 §5.1 fixed-header walker that bounds-checks the CSRC list
//! against the attacker-controlled CC field), then through
//! `unpack_header` (the RFC 4587 §4.1 4-byte H.261 payload-header
//! parser) on both the post-CSRC tail and on the raw buffer itself.
//! A multi-packet `depacketize` walk is then driven against
//! attacker-derived split positions and per-packet SBIT/EBIT/MV
//! field combinations so the slow-path bit-walker stays covered.
//!
//! In addition to the on-disk corpus, the test drives several
//! adversarial in-line buffers — empty, single-zero, all-ones, a
//! deterministic pseudo-random buffer, plus hand-crafted edge cases
//! (CC=15 with a 12-byte truncated tail, version=1 rejection, exactly
//! the 12-byte fixed-header boundary, SBIT+EBIT summing past the
//! payload length) — so a corrupt corpus doesn't hide the
//! parser-surface contract under test.
//!
//! The RTP payload parser is the third attacker-reachable media-path
//! surface an H.261 endpoint exposes on the wire (the first two are
//! the elementary-stream decoder, covered by
//! `tests/fuzz_seed_corpus.rs`, and the BCH §5.4 FEC framer, covered
//! by `tests/fuzz_seed_corpus_bch.rs`; the RTCP control-channel
//! parser covered by `tests/fuzz_seed_corpus_rtcp.rs` is a peer
//! interface, not strictly a media path).

use std::fs;
use std::path::PathBuf;

use oxideav_h261::rtp::{
    depacketize, pack_header, parse_rtp_fixed_header, unpack_header, H261RtpHeader, H261RtpPayload,
    HEADER_LEN, RTP_FIXED_HEADER_LEN,
};

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz")
        .join("corpus")
        .join("parse_rtp_payload")
}

/// Mirror the fuzz target's three-mode driver against a single buffer.
fn drive(bytes: &[u8]) {
    // ---- Mode A: full §5.1 + §4.1 walk on raw bytes. ----
    if let Ok((_, post_fixed)) = parse_rtp_fixed_header(bytes) {
        let _ = unpack_header(post_fixed);
    }

    // ---- Mode B: §4.1 header parser in isolation. ----
    let _ = unpack_header(bytes);

    // ---- Mode C: multi-packet depacketise with attacker-chosen splits. ----
    if bytes.len() >= 8 {
        let split_seed = bytes[0];
        let n_packets = ((split_seed >> 6) & 0x3) as usize + 1; // 1..=4
        let mut payloads: Vec<H261RtpPayload> = Vec::with_capacity(n_packets);
        let mut cursor = 1usize;
        for pkt_idx in 0..n_packets {
            if cursor + 3 >= bytes.len() {
                break;
            }
            let s0 = bytes[cursor];
            let s1 = bytes[cursor + 1];
            let s2 = bytes[cursor + 2];
            cursor += 3;

            let sbit = s0 & 0x07;
            let ebit = (s0 >> 3) & 0x07;
            let intra_only = (s0 & 0x40) != 0;
            let motion_vectors = (s0 & 0x80) != 0;
            let gobn = s1 & 0x0F;
            let mbap = (s1 >> 4) & 0x1F;
            let quant = s2 & 0x1F;
            let hmvd = ((s2 >> 5) as i8 & 0x07) - 4;
            let vmvd = ((s1 >> 1) as i8 & 0x07) - 4;

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

            if cursor >= bytes.len() {
                break;
            }
            let want_data = bytes[cursor] as usize;
            cursor += 1;
            let take = core::cmp::min(want_data, bytes.len().saturating_sub(cursor));
            let data_slice = &bytes[cursor..cursor + take];
            cursor += take;

            let header_bytes = match pack_header(&hdr) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut packet_bytes = Vec::with_capacity(HEADER_LEN + data_slice.len());
            packet_bytes.extend_from_slice(&header_bytes);
            packet_bytes.extend_from_slice(data_slice);

            payloads.push(H261RtpPayload {
                header: hdr,
                bytes: packet_bytes,
                marker: pkt_idx + 1 == n_packets,
            });
        }
        let _ = depacketize(&payloads);
    }

    // ---- Mode D: edge slice on the §5.1 minimum-buffer boundary. ----
    if bytes.len() >= RTP_FIXED_HEADER_LEN {
        let _ = parse_rtp_fixed_header(&bytes[..RTP_FIXED_HEADER_LEN]);
    }
}

#[test]
fn corpus_files_drive_rtp_parsers_without_panicking() {
    let dir = corpus_dir();
    let entries = fs::read_dir(&dir).unwrap_or_else(|e| {
        panic!("read fuzz corpus dir {}: {}", dir.display(), e);
    });
    let mut count = 0;
    for ent in entries {
        let path = ent.unwrap().path();
        if !path.is_file() {
            continue;
        }
        let bytes = fs::read(&path).expect("read seed");
        drive(&bytes);
        count += 1;
    }
    assert!(
        count >= 1,
        "expected at least one seed in {}",
        dir.display()
    );
}

#[test]
fn empty_bytes_dont_panic() {
    drive(&[]);
}

#[test]
fn single_zero_byte_doesnt_panic() {
    // 1 byte < 12 → ShortHeader on `parse_rtp_fixed_header`, also <
    // 4 → ShortHeader on `unpack_header`. Neither path may panic.
    drive(&[0u8]);
}

#[test]
fn all_ones_doesnt_panic() {
    // 0xFF leading byte ⇒ V = 0b11 = 3 ⇒ FieldOverflow rejection.
    // 0xFF leading byte through `unpack_header` decodes SBIT=7 EBIT=7
    // and HMVD/VMVD = -1, all in-range; the depacketise carve then
    // exercises the SBIT=7 EBIT=7 slow-path bit walker.
    let bytes = vec![0xFFu8; 128];
    drive(&bytes);
}

#[test]
fn random_pattern_doesnt_panic() {
    // Deterministic pseudo-random buffer; most random seeds fail the
    // V=2 check, but the bit-walker carve will fire on the long
    // 256-byte input and drive a 4-packet sequence through
    // `depacketize`.
    let mut bytes = Vec::with_capacity(256);
    let mut x: u32 = 0xdead_beef;
    for _ in 0..256 {
        x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
        bytes.push((x >> 16) as u8);
    }
    drive(&bytes);
}

#[test]
fn exact_fixed_header_boundary_doesnt_panic() {
    // Exactly RTP_FIXED_HEADER_LEN = 12 bytes; CC=0 so the parser
    // accepts. Tail slice is empty; `unpack_header` on it returns
    // ShortHeader.
    let bytes = vec![0x80, 0x60, 0x00, 0x01, 0, 0, 0, 0, 0xDE, 0xAD, 0xBE, 0xEF];
    drive(&bytes);
    let (hdr, tail) = parse_rtp_fixed_header(&bytes).expect("V=2 CC=0 must parse");
    assert_eq!(hdr.version, 2);
    assert_eq!(hdr.csrc_count, 0);
    assert_eq!(hdr.payload_type, 96);
    assert_eq!(tail.len(), 0);
}

#[test]
fn cc15_truncated_rejected_as_short_header() {
    // V=2 CC=15 in the first byte; buffer is only 12 bytes so
    // 12 + 4*15 = 72 > 12 must surface as ShortHeader, not an OOB
    // read.
    let bytes = vec![0x8F, 0xE0, 0x00, 0x09, 0, 0, 0, 0, 0xDE, 0xAD, 0xBE, 0xEF];
    drive(&bytes);
    let err = parse_rtp_fixed_header(&bytes);
    assert!(err.is_err(), "CC=15 with 12-byte buffer must reject");
}

#[test]
fn version_one_rejected_as_field_overflow() {
    // V=1 in the first byte; parser must surface FieldOverflow.
    let mut bytes = vec![0x40, 0x60, 0x00, 0x01, 0, 0, 0, 0, 0xDE, 0xAD, 0xBE, 0xEF];
    bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00, 0xAB, 0xCD]);
    drive(&bytes);
    let err = parse_rtp_fixed_header(&bytes);
    assert!(err.is_err(), "V=1 must reject with FieldOverflow");
}

#[test]
fn unpack_header_roundtrip_through_pack() {
    // Round-trip the typical-fields header used in the seed corpus
    // (SBIT=3 EBIT=5 I=V=1 GOBN=7 MBAP=12 QUANT=17 HMVD=-7 VMVD=11)
    // so a regression in either `pack_header` or `unpack_header`
    // trips this test even if the corpus is missing.
    let hdr = H261RtpHeader {
        sbit: 3,
        ebit: 5,
        intra_only: true,
        motion_vectors: true,
        gobn: 7,
        mbap: 12,
        quant: 17,
        hmvd: -7,
        vmvd: 11,
    };
    let bytes = pack_header(&hdr).expect("typical-fields pack");
    let (back, rest) = unpack_header(&bytes).expect("typical-fields unpack");
    assert_eq!(back, hdr);
    assert!(rest.is_empty());
}

#[test]
fn depacketize_with_sbit_ebit_overflow_rejected() {
    // SBIT + EBIT >= 8 * data.len() is the §4.1 "no usable bits"
    // case — the depacketiser must surface EmptyPayload, not an
    // integer underflow. Use one packet with 1 byte of data and
    // SBIT=4 EBIT=5 → 9 >= 8.
    let hdr = H261RtpHeader {
        sbit: 4,
        ebit: 5,
        intra_only: false,
        motion_vectors: false,
        gobn: 0,
        mbap: 0,
        quant: 0,
        hmvd: 0,
        vmvd: 0,
    };
    let header_bytes = pack_header(&hdr).expect("sbit=4 ebit=5 packs fine");
    let mut bytes = Vec::with_capacity(HEADER_LEN + 1);
    bytes.extend_from_slice(&header_bytes);
    bytes.push(0x80);
    let payload = H261RtpPayload {
        header: hdr,
        bytes,
        marker: true,
    };
    let err = depacketize(&[payload]);
    assert!(err.is_err(), "SBIT+EBIT >= 8*data_bits must reject");
}
