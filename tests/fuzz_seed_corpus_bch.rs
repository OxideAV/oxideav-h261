//! Sanity-check the fuzz seed corpus under
//! `fuzz/corpus/decode_bch_multiframe/` and drive each seed through the
//! same public BCH parser surface the `decode_bch_multiframe` fuzz
//! target exercises. Because `cargo-fuzz` requires the nightly
//! toolchain (libFuzzer's sanitizer-coverage flags are `-Z`-gated), the
//! regular CI matrix never builds the fuzz crate; this stable-Rust test
//! gives the same logical coverage so a corrupted corpus or a regressed
//! BCH framing/syndrome path trips one of the existing CI lanes instead
//! of waiting for the daily fuzz run to notice.
//!
//! The harness mirrors `fuzz/fuzz_targets/decode_bch_multiframe.rs`
//! exactly: each buffer is fed through `decode_multiframe` (the §5.4.4
//! lock + per-frame syndrome path), through `parity18` on a 62-byte
//! zero-padded slice (the §5.4.2 generator-polynomial long-division
//! primitive), and through `syndrome18` with both the matching parity
//! and an attacker-controlled 18-bit value (driving the non-zero-
//! syndrome branch).
//!
//! In addition to the on-disk corpus, the test drives several
//! adversarial in-line buffers — empty, single-zero, all-ones, a
//! deterministic pseudo-random buffer, and a handful of hand-crafted
//! edge-case inputs (one-byte-short-of-lock, exactly-three-multiframes,
//! and a deterministic bit-flip injection on a well-formed framed
//! stream) — so a corrupt corpus doesn't hide the parser-surface
//! contract under test.
//!
//! The BCH framing layer (§5.4) is one of the two attacker-reachable
//! media parsers an H.261 endpoint exposes on the wire (the other is
//! the elementary-stream decoder, covered by
//! `tests/fuzz_seed_corpus.rs`).

use std::fs;
use std::path::PathBuf;

use oxideav_h261::bch::{
    decode_multiframe, decode_multiframe_with_correction, encode_multiframe, locate_single_error,
    parity18, syndrome18, FRAME_BITS, MULTIFRAME_FRAMES,
};

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz")
        .join("corpus")
        .join("decode_bch_multiframe")
}

fn drive(bytes: &[u8]) {
    // §5.4.4 lock + per-frame syndrome + data extraction.
    let _ = decode_multiframe(bytes);

    // §5.4.1 lock + per-frame syndrome + t = 1 single-bit correction +
    // data extraction.
    let _ = decode_multiframe_with_correction(bytes);

    // §5.4.2 GF(2) long division on a 62-byte zero-padded slice.
    let mut padded = [0u8; 62];
    let take = core::cmp::min(bytes.len(), padded.len());
    padded[..take].copy_from_slice(&bytes[..take]);
    let par = parity18(&padded);

    // Self-syndrome must be zero by construction; the call exercises
    // the 511-bit shift-register walk against the input bytes.
    let _ = syndrome18(&padded, par);

    // Attacker-controlled 18-bit parity sourced from the first 3 input
    // bytes (zero-padded if shorter) — drives the non-zero-syndrome
    // branch.
    let mut p = [0u8; 3];
    let take = core::cmp::min(bytes.len(), p.len());
    p[..take].copy_from_slice(&bytes[..take]);
    let attacker_parity = (((p[0] as u32) << 16) | ((p[1] as u32) << 8) | (p[2] as u32)) & 0x3_FFFF;
    let _ = syndrome18(&padded, attacker_parity);

    // Drive the §5.4.1 single-bit-error position lookup against an
    // attacker-controlled 18-bit syndrome candidate. The function
    // bounds its walk at 511 iterations, so the run time is constant
    // regardless of the value of `attacker_parity`.
    let _ = locate_single_error(attacker_parity);

    // Edge-of-lock-window slice.
    let lock_span_bits = (3 * MULTIFRAME_FRAMES * FRAME_BITS) as usize;
    let edge_len = lock_span_bits.div_ceil(8);
    if bytes.len() >= edge_len {
        let _ = decode_multiframe(&bytes[..edge_len]);
        let _ = decode_multiframe_with_correction(&bytes[..edge_len]);
    }
}

#[test]
fn corpus_files_drive_bch_without_panicking() {
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
    // All-ones can never satisfy the §5.4.4 alignment pattern (which
    // requires three leading 0-bits in `(00011011)`), so the
    // lock-search must exhaust every candidate offset and return None
    // without panicking.
    let bytes = vec![0xffu8; 3 * 512];
    drive(&bytes);
}

#[test]
fn random_pattern_doesnt_panic() {
    // Deterministic pseudo-random buffer of three multiframes — large
    // enough for the lock-search to run its full candidate sweep.
    let mut bytes = Vec::with_capacity(3 * 512);
    let mut x: u32 = 0xdead_beef;
    for _ in 0..(3 * 512) {
        x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
        bytes.push((x >> 16) as u8);
    }
    drive(&bytes);
}

#[test]
fn one_byte_short_of_lock_doesnt_panic() {
    // 3 multiframes = 1536 bytes. One byte short is *not* enough for
    // a complete 24-frame lock window — the early-out branch must fire.
    let bytes = vec![0u8; 3 * 512 - 1];
    drive(&bytes);
}

#[test]
fn exactly_three_multiframes_locks_cleanly() {
    // The minimum input that satisfies §5.4.4. Round-trip a small
    // payload and verify the decoded stream comes back without panic.
    let payload = vec![0x42u8; 11_808 / 8];
    let framed = encode_multiframe(&payload, payload.len() * 8);
    assert_eq!(framed.len(), 3 * 512);
    let dec = decode_multiframe(&framed).expect("lock on minimum input");
    assert_eq!(dec.frames_consumed, 24);
    assert_eq!(dec.corrupted_frames, 0);
}

#[test]
fn deterministic_bit_flips_drive_syndrome_branch() {
    // Frame three multiframes' worth of payload, then deterministically
    // flip 4 bits — at positions that fall inside the data portion of
    // four distinct frames. Each flip should bump `corrupted_frames`
    // exactly once; lock must be preserved because none of the flips
    // land on a framing bit.
    let payload = vec![0xA5u8; 11_808 / 8];
    let mut framed = encode_multiframe(&payload, payload.len() * 8);
    assert_eq!(framed.len(), 3 * 512);
    // Flips at byte 100, 400, 700, 1000 — each falls in a different
    // 512-byte multiframe / sub-multiframe region.
    framed[100] ^= 0b0000_1000;
    framed[400] ^= 0b0010_0000;
    framed[700] ^= 0b0001_0000;
    framed[1000] ^= 0b0100_0000;
    let dec = decode_multiframe(&framed).expect("lock survives data flips");
    assert!(
        dec.corrupted_frames >= 1,
        "syndrome should flag ≥ 1 corrupted frame"
    );
    assert_eq!(dec.frames_consumed, 24);
}

#[test]
fn header_with_attacker_high_parity_value_is_handled() {
    // syndrome18 accepts a u32 `parity`; values > 18 bits are not
    // forbidden by the API. The implementation masks them down via
    // the same long-division loop. A value with bits set above
    // position 17 still produces a well-defined (but non-zero)
    // syndrome; the parser must never panic on the path.
    let msg = [0u8; 62];
    let _ = syndrome18(&msg, 0xFFFF_FFFF);
    let _ = syndrome18(&msg, 0x8000_0000);
}

#[test]
fn correction_path_round_trips_clean_input() {
    // The minimum lock input — three multiframes of payload, no
    // corruption — through the correction path: no flagged frames,
    // no corrections fired, payload bit-exact.
    let payload = vec![0x42u8; 11_808 / 8];
    let framed = encode_multiframe(&payload, payload.len() * 8);
    let dec = decode_multiframe_with_correction(&framed).expect("lock on minimum input");
    assert_eq!(dec.frames_consumed, 24);
    assert_eq!(dec.corrupted_frames, 0);
    assert_eq!(dec.corrected_frames, 0);
    assert_eq!(dec.uncorrectable_frames, 0);
    assert_eq!(&dec.data[..payload.len()], &payload[..]);
}

#[test]
fn correction_path_recovers_single_bit_data_error_byte_exact() {
    // Frame a payload, flip a single bit, verify the correction path
    // restores the payload bit-exact and the counts are
    // (corrupted=1, corrected=1, uncorrectable=0).
    let payload = vec![0xC3u8; 11_808 / 8];
    let mut framed = encode_multiframe(&payload, payload.len() * 8);
    framed[700] ^= 0b0001_0000;
    let dec = decode_multiframe_with_correction(&framed).expect("lock survives single-bit error");
    assert_eq!(dec.corrupted_frames, 1);
    assert_eq!(dec.corrected_frames, 1);
    assert_eq!(dec.uncorrectable_frames, 0);
    assert_eq!(&dec.data[..payload.len()], &payload[..]);
}

#[test]
fn locate_single_error_attacker_syndromes_dont_loop() {
    // Drive `locate_single_error` against a hand-rolled adversarial
    // set: zero (must report no error), all-ones (a syndrome value
    // that almost certainly doesn't correspond to any single-bit
    // error pattern), an 18-bit-saturated value, and the syndrome of
    // each of a handful of known single-bit errors (must round-trip).
    assert_eq!(locate_single_error(0), None);
    let _ = locate_single_error(0xFFFF_FFFF);
    let _ = locate_single_error(0x3_FFFF);

    for &p in &[0usize, 1, 5, 100, 492, 493, 510] {
        let mut scratch = [0u8; 62];
        let mut par: u32 = 0;
        if p < 493 {
            scratch[p / 8] |= 1 << (7 - (p & 7));
        } else {
            par |= 1 << (17 - (p - 493));
        }
        let s = syndrome18(&scratch, par);
        assert_eq!(locate_single_error(s), Some(p as u32));
    }
}
