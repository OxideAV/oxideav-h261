//! Sanity-check the fuzz seed corpus under
//! `fuzz/corpus/packetize_h261/` and drive each seed through the same
//! public RTP *packetiser* (send / forward) surface the `packetize_h261`
//! fuzz target exercises. Because `cargo-fuzz` requires the nightly
//! toolchain (libFuzzer's sanitizer-coverage flags are `-Z`-gated), the
//! regular CI matrix never builds the fuzz crate; this stable-Rust test
//! gives the same logical coverage so a corrupted corpus or a regressed
//! packetiser surface trips one of the existing CI lanes instead of
//! waiting for the daily fuzz run to notice.
//!
//! The packetiser is the complement of the RTP *receive* path covered
//! by `tests/fuzz_seed_corpus_rtp.rs`: it parses an H.261 elementary
//! stream's VLC macroblock layer to split it into RTP payloads. The
//! richest attacker-reachable parse here is `packetize_mb_fragmented`
//! (RFC 4587 §4.2 RECOMMENDED MB-level fragmentation) — an MCU /
//! forwarder re-fragmenting an *untrusted* sender's bitstream runs that
//! Huffman/VLC walk over bytes it did not produce.
//!
//! The harness mirrors `fuzz/fuzz_targets/packetize_h261.rs`: each
//! buffer is fed through `packetize_gob_aligned` (total over arbitrary
//! bytes), through `packetize_mb_fragmented` (Result; a malformed stream
//! is `Err`, never a panic), and through `RtpPacketizer::pack_frame` in
//! both fragmentation modes; the GOB-aligned packetiser's output is then
//! re-run through the receive-path `depacketize` to exercise the
//! pack → unpack seam.
//!
//! The contract under test is that every call *returns* — no panic, no
//! abort, no integer overflow, no out-of-bounds index, no OOM. The
//! documented `assert!` preconditions on the payload budgets (each
//! packet must hold the headers + ≥ 1 data byte) are caller-contract
//! violations, so the harness only feeds budgets above the documented
//! floor.

use std::fs;
use std::path::PathBuf;

use oxideav_h261::rtp::{
    depacketize, packetize_gob_aligned, packetize_mb_fragmented, RtpError, RtpPacketizer,
    HEADER_LEN, RTP_FIXED_HEADER_LEN,
};

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz")
        .join("corpus")
        .join("packetize_h261")
}

/// Mirror the fuzz target's driver against a single buffer across a
/// spread of payload budgets and both fragmentation modes.
fn drive(stream: &[u8]) {
    for &inner in &[
        HEADER_LEN + 1,
        HEADER_LEN + 7,
        HEADER_LEN + 64,
        HEADER_LEN + 512,
    ] {
        for io in [false, true] {
            for mv in [false, true] {
                // GOB-aligned packetiser is total over arbitrary bytes.
                let gob = packetize_gob_aligned(stream, inner, io, mv);
                if !gob.is_empty() {
                    // Round-trip seam — must also just return; no
                    // byte-equality assertion (GOB packetisation is lossy
                    // w.r.t. exact stuffing bits).
                    let _ = depacketize(&gob);
                }
                // MB-level VLC walk over attacker bytes; Result, never panic.
                let _ = packetize_mb_fragmented(stream, inner, io, mv);
            }
        }
    }

    // Session-stateful packetiser, both fragmentation modes, across two
    // frames so the running packet / octet / sequence counters advance.
    for &rtp in &[
        RTP_FIXED_HEADER_LEN + HEADER_LEN + 1,
        RTP_FIXED_HEADER_LEN + HEADER_LEN + 256,
    ] {
        for frag in [false, true] {
            let mut p = RtpPacketizer::new(96, 0xdead_beef, 7, rtp)
                .with_intra_only(false)
                .with_motion_vectors(true)
                .with_mb_fragmentation(frag);
            let _ = p.pack_frame(stream, 0);
            let _ = p.pack_frame(stream, 3000);
        }
    }
}

#[test]
fn corpus_files_drive_packetiser_without_panicking() {
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
    drive(&[0xFFu8; 128]);
}

#[test]
fn random_pattern_doesnt_panic() {
    let mut bytes = Vec::with_capacity(256);
    let mut x: u32 = 0xdead_beef;
    for _ in 0..256 {
        x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
        bytes.push((x >> 16) as u8);
    }
    drive(&bytes);
}

#[test]
fn mb_fragment_returns_err_not_panic_on_desynced_mv() {
    // Regression for a packetiser panic surfaced by the `packetize_h261`
    // fuzz target. On a malformed INTER stream the §4.2.3.4 MV predictor
    // can desync, producing a reconstructed MV outside the RFC 4587 §4.1
    // 5-bit `-15..=15` HMVD/VMVD field. The MB-level fragmenter copied
    // that MV into a mid-GOB continuation header; `pack_header` then
    // rejected it with `ForbiddenMvd` / `FieldOverflow`. The fragmenter
    // used to `.expect("hdr packs")` that pack — turning a recoverable
    // malformed-input error into a process abort, fatal for an MCU /
    // forwarder re-fragmenting an untrusted sender's bitstream. It must
    // return `Err`. This 34-byte stream + budget=11 is a fuzz-minimized
    // reproducer.
    let stream: &[u8] = &[
        0, 101, 0, 33, 0, 0, 0, 0, 0, 12, 39, 0, 0, 0, 0, 155, 0, 0, 0, 0, 0, 52, 51, 51, 51, 51,
        51, 51, 3, 39, 37, 83, 83, 11,
    ];
    match packetize_mb_fragmented(stream, HEADER_LEN + 7, false, false) {
        Err(RtpError::ForbiddenMvd { .. }) | Err(RtpError::FieldOverflow { .. }) => {}
        other => panic!("expected ForbiddenMvd/FieldOverflow Err, got {other:?}"),
    }
    // The full fuzz driver must also stay panic-free on this input.
    drive(stream);
}
