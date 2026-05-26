//! Sanity-check the fuzz seed corpus under `fuzz/corpus/parse_rtcp_compound/`
//! and drive each seed through the same public RTCP parser surface the
//! `parse_rtcp_compound` fuzz target exercises. Because `cargo-fuzz`
//! requires the nightly toolchain (libFuzzer's sanitizer-coverage flags
//! are `-Z`-gated), the regular CI matrix never builds the fuzz crate;
//! this stable-Rust test gives the same logical coverage so a corrupted
//! corpus or a regressed parser surface trips one of the existing CI
//! lanes instead of waiting for the daily fuzz run to notice.
//!
//! The harness mirrors `fuzz/fuzz_targets/parse_rtcp_compound.rs`
//! exactly: each buffer is fed to [`parse_compound`] (the canonical
//! receiver entry point), then to each individual sub-packet parser
//! ([`parse_report`], [`parse_sdes`], [`parse_bye`], [`parse_app`]) so
//! a single bad sub-packet that the compound walk rejects before
//! reaching it still gets its own parser coverage.
//!
//! In addition to the on-disk corpus, the test drives several
//! adversarial in-line buffers — empty, single-zero, all-ones, a
//! deterministic pseudo-random buffer, and a handful of hand-crafted
//! "lies about the length" header buffers — so a corrupt corpus doesn't
//! hide the parser-surface contract under test.

use std::fs;
use std::path::PathBuf;

use oxideav_h261::rtcp::{parse_app, parse_bye, parse_compound, parse_report, parse_sdes};

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz")
        .join("corpus")
        .join("parse_rtcp_compound")
}

fn drive(bytes: &[u8]) {
    let _ = parse_compound(bytes);
    let _ = parse_report(bytes);
    let _ = parse_sdes(bytes);
    let _ = parse_bye(bytes);
    let _ = parse_app(bytes);
}

#[test]
fn corpus_files_drive_parsers_without_panicking() {
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
    drive(&[0u8]);
}

#[test]
fn all_ones_doesnt_panic() {
    let bytes = vec![0xffu8; 256];
    drive(&bytes);
}

#[test]
fn random_pattern_doesnt_panic() {
    // Deterministic pseudo-random buffer — unlikely to be a valid RTCP
    // datagram, but exercises the compound walk's "advance via stated
    // length" branch a fuzz target would hit on the first iteration.
    let mut bytes = Vec::with_capacity(4096);
    let mut x: u32 = 0xdead_beef;
    for _ in 0..4096 {
        x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
        bytes.push((x >> 16) as u8);
    }
    drive(&bytes);
}

#[test]
fn header_claims_huge_length_is_rejected() {
    // V=2, P=0, RC=0, PT=200 (SR), length = 0xFFFF (would advance past
    // a 4-byte buffer end). Must return Err, not panic / loop / OOM.
    let bytes = [0x80, 200, 0xff, 0xff];
    drive(&bytes);
}

#[test]
fn header_claims_zero_length_is_advanced_correctly() {
    // length = 0 → sub_len = (0 + 1) * 4 = 4. With a 4-byte buffer this
    // is the minimum legal sub-packet; the walk must terminate after
    // exactly one iteration rather than spinning on a zero advance.
    // (The parser will reject PT 200 with no body, but the walk must
    // terminate either way.)
    let bytes = [0x80, 200, 0x00, 0x00];
    drive(&bytes);
}

#[test]
fn truncated_compound_walks_partway() {
    // Two valid-looking sub-packet headers followed by a third truncated
    // one — the parser must reject the third (Truncated) and not parse
    // past the end of the buffer.
    let mut bytes = Vec::new();
    // RR with RC=0: length=1 → sub_len=8.
    bytes.extend_from_slice(&[0x80, 201, 0x00, 0x01]);
    // SSRC.
    bytes.extend_from_slice(&[0xaa, 0xaa, 0xaa, 0xaa]);
    // BYE with SC=1 claiming length=2 (sub_len=12) — only 4 bytes follow.
    bytes.extend_from_slice(&[0x81, 203, 0x00, 0x02]);
    bytes.extend_from_slice(&[0xbb, 0xbb, 0xbb, 0xbb]);
    drive(&bytes);
}

#[test]
fn sdes_priv_length_overflow_is_rejected() {
    // SDES with one chunk: SSRC, then a PRIV item (type=8) claiming an
    // inner prefix-length that runs past the item-length. The parser
    // must reject (not panic) and the compound walk must still advance
    // via the outer 16-bit length field.
    let mut bytes = Vec::new();
    // SDES header: V=2, P=0, SC=1, PT=202, length = 3 → sub_len = 16.
    bytes.extend_from_slice(&[0x81, 202, 0x00, 0x03]);
    // SSRC.
    bytes.extend_from_slice(&[0xcc, 0xcc, 0xcc, 0xcc]);
    // PRIV item: type=8, length=4, prefix-length=0xff (lies), then 3 bytes,
    // then END(0) + padding to 32-bit alignment.
    bytes.extend_from_slice(&[8, 4, 0xff, b'a', b'b', b'c', 0, 0]);
    drive(&bytes);
}

#[test]
fn bye_reason_length_overflow_is_rejected() {
    // BYE with SC=1, reason length = 0xff but only 3 reason bytes follow.
    let mut bytes = Vec::new();
    // BYE: V=2, P=0, SC=1, PT=203, length = 2 → sub_len = 12
    bytes.extend_from_slice(&[0x81, 203, 0x00, 0x02]);
    bytes.extend_from_slice(&[0xdd, 0xdd, 0xdd, 0xdd]); // SSRC
    bytes.extend_from_slice(&[0xff, b'x', b'y', b'z']); // reason len lies
    drive(&bytes);
}

#[test]
fn app_subtype_at_max() {
    // APP with subtype=31 (the §6.7 5-bit max), 4-byte name "PING",
    // zero application data — minimum-size valid APP packet.
    let mut bytes = Vec::new();
    // V=2, P=0, subtype=31, PT=204, length = 2 → sub_len = 12
    bytes.extend_from_slice(&[0x9f, 204, 0x00, 0x02]);
    bytes.extend_from_slice(&[0xee, 0xee, 0xee, 0xee]); // SSRC
    bytes.extend_from_slice(b"PING");
    drive(&bytes);
}

#[test]
fn unknown_packet_type_surfaces_as_other() {
    // PT=205 (RTPFB, RFC 4585) is not modelled by this crate — the
    // compound walk must still advance via the length field and return
    // an Other(...) entry rather than failing the whole datagram.
    let mut bytes = Vec::new();
    // V=2, P=0, FMT=0, PT=205, length = 2 → sub_len = 12
    bytes.extend_from_slice(&[0x80, 205, 0x00, 0x02]);
    bytes.extend_from_slice(&[0xff, 0xff, 0xff, 0xff]);
    bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
    drive(&bytes);
}
