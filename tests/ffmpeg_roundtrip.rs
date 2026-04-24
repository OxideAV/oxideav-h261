//! Integration test — encode a synthetic I-picture, then use the
//! installed `ffmpeg` binary (if any) to decode it and compare against
//! the source YUV. Skipped gracefully when ffmpeg is not on PATH.
//!
//! This complements the in-crate decoder round-trip in
//! `src/encoder.rs::tests` — ffmpeg is the true conformance target.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use oxideav_h261::encoder::encode_intra_picture;
use oxideav_h261::picture::SourceFormat;

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn gradient_qcif_source() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = 176usize;
    let h = 144usize;
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            // Smooth diagonal gradient.
            let v = 32 + (i * 150) / w + (j * 50) / h;
            y[j * w + i] = v.clamp(0, 255) as u8;
        }
    }
    let cb = vec![128u8; (w / 2) * (h / 2)];
    let cr = vec![128u8; (w / 2) * (h / 2)];
    (y, cb, cr)
}

fn tmpdir() -> PathBuf {
    let d = std::env::temp_dir().join("oxideav-h261-enc-roundtrip");
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn ffmpeg_decodes_our_qcif_intra_picture() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not on PATH — skipping");
        return;
    }
    let (y, cb, cr) = gradient_qcif_source();
    let bytes = encode_intra_picture(
        SourceFormat::Qcif,
        &y,
        176,
        &cb,
        88,
        &cr,
        88,
        8,
        0,
    )
    .expect("encode");

    let dir = tmpdir();
    let es_path = dir.join("q.h261");
    let yuv_path = dir.join("q.yuv");
    std::fs::write(&es_path, &bytes).unwrap();

    // Run ffmpeg to decode.
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "h261",
            "-i",
        ])
        .arg(&es_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .expect("spawn ffmpeg");

    // ffmpeg might refuse to probe a bare QCIF intra-only picture (needs at
    // least a PSC + some trailing data). We don't hard-fail the test on
    // non-zero exit — we check the output file instead.
    if !status.success() {
        // Write diagnostic stderr for postmortem.
        let err = Command::new("ffmpeg")
            .args(["-y", "-hide_banner", "-f", "h261", "-i"])
            .arg(&es_path)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv_path)
            .stderr(Stdio::piped())
            .output()
            .expect("spawn ffmpeg diag");
        let msg = String::from_utf8_lossy(&err.stderr);
        // Write our bitstream to stdout for debugging.
        let mut h = std::io::stderr();
        writeln!(h, "encoded bytes = {} long", bytes.len()).ok();
        writeln!(h, "first 32 bytes: {:02x?}", &bytes[..32.min(bytes.len())]).ok();
        writeln!(h, "ffmpeg stderr: {msg}").ok();
        panic!("ffmpeg failed to decode our H.261 QCIF intra picture");
    }

    let decoded = std::fs::read(&yuv_path).expect("read yuv");
    let expected_size = 176 * 144 * 3 / 2;
    assert_eq!(
        decoded.len(),
        expected_size,
        "ffmpeg produced wrong-size YUV420 ({})",
        decoded.len()
    );

    // Compare Y plane against the source — we expect large tolerance
    // because ffmpeg's IDCT + our quantisation can diverge a few levels,
    // but the block-mean reconstruction should be close.
    let y_out = &decoded[..176 * 144];
    let mut err_sum: u64 = 0;
    for (a, b) in y.iter().zip(y_out.iter()) {
        err_sum += (*a as i32 - *b as i32).unsigned_abs() as u64;
    }
    let mean_err = err_sum as f64 / (176.0 * 144.0);
    assert!(
        mean_err < 15.0,
        "mean Y error {mean_err:.2} too large (should be <15 for QUANT=8)"
    );
}
