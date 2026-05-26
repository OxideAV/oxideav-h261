//! Sanity-check the fuzz seed corpus under `fuzz/corpus/decode_h261/`:
//! every file there must be decodable through the same public
//! decoder surface the fuzz harness drives (`send_packet` -> drain
//! `receive_frame` -> `flush` -> drain again). Because cargo-fuzz
//! requires the nightly toolchain (libFuzzer's sanitizer-coverage
//! flags are `-Z`-gated), the regular CI matrix never builds the
//! fuzz crate; this test gives the same logical coverage on stable
//! so a corrupted corpus or a regressed decoder surface fails one
//! of the existing CI lanes instead of waiting for the daily fuzz
//! run to notice.
//!
//! The harness mirrors `fuzz/fuzz_targets/decode_h261.rs` exactly:
//! drive each seed through one `send_packet`, drain bounded frames,
//! then `flush` + drain again. The test merely asserts the calls
//! return — it doesn't pin a frame count, because the corpus is
//! curated to grow over time.

use std::fs;
use std::path::PathBuf;

use oxideav_core::{CodecId, Decoder, Packet, TimeBase};
use oxideav_h261::decoder::H261Decoder;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz")
        .join("corpus")
        .join("decode_h261")
}

fn drive(bytes: &[u8]) {
    let mut dec = H261Decoder::new(CodecId::new("h261"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes.to_vec());
    let _ = dec.send_packet(&pkt);
    for _ in 0..32 {
        if dec.receive_frame().is_err() {
            break;
        }
    }
    let _ = dec.flush();
    for _ in 0..32 {
        if dec.receive_frame().is_err() {
            break;
        }
    }
}

#[test]
fn corpus_files_drive_decoder_without_panicking() {
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
    // A deterministic pseudo-random buffer that's unlikely to be a
    // valid H.261 stream but should still drive the start-code scanner
    // and bail cleanly with `Err(...)`.
    let mut bytes = Vec::with_capacity(4096);
    let mut x: u32 = 0xdead_beef;
    for _ in 0..4096 {
        x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
        bytes.push((x >> 16) as u8);
    }
    drive(&bytes);
}
