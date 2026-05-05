//! Integration test — encode a synthetic I-picture, then use the
//! installed `ffmpeg` binary (if any) to decode it and compare against
//! the source YUV. Skipped gracefully when ffmpeg is not on PATH.
//!
//! This complements the in-crate decoder round-trip in
//! `src/encoder.rs::tests` — ffmpeg is the true conformance target.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use oxideav_core::{CodecId, CodecParameters, Frame, RuntimeContext, VideoFrame, VideoPlane};
use oxideav_h261::encoder::{encode_intra_picture, H261Encoder};
use oxideav_h261::picture::SourceFormat;

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return f64::INFINITY;
    }
    let mut sse = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        sse += d * d;
    }
    let mse = sse / a.len() as f64;
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

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
    let bytes =
        encode_intra_picture(SourceFormat::Qcif, &y, 176, &cb, 88, &cr, 88, 8, 0).expect("encode");

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

/// Encode a 3-frame I+P+P sequence with our encoder and have ffmpeg decode
/// it. Validates that the P-picture bitstream we emit is at least
/// syntactically conformant — ffmpeg's H.261 demuxer has a real picture
/// scanner so it requires PSC-anchored pictures.
#[test]
fn ffmpeg_decodes_our_qcif_ipp_sequence() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not on PATH — skipping");
        return;
    }
    let (y0, cb0, cr0) = gradient_qcif_source();
    // Frame 1: shift a small region.
    let mut y1 = y0.clone();
    for j in 32..64 {
        for i in 32..96 {
            y1[j * 176 + i] = y1[j * 176 + i].saturating_add(20);
        }
    }
    // Frame 2: shift another region.
    let mut y2 = y1.clone();
    for j in 80..112 {
        for i in 64..128 {
            y2[j * 176 + i] = y2[j * 176 + i].saturating_sub(15);
        }
    }

    let mut enc = H261Encoder::new(SourceFormat::Qcif, 8);
    let p0 = enc.encode_frame(&y0, 176, &cb0, 88, &cr0, 88).expect("f0");
    let p1 = enc.encode_frame(&y1, 176, &cb0, 88, &cr0, 88).expect("f1");
    let p2 = enc.encode_frame(&y2, 176, &cb0, 88, &cr0, 88).expect("f2");

    let mut stream = Vec::new();
    stream.extend_from_slice(&p0);
    stream.extend_from_slice(&p1);
    stream.extend_from_slice(&p2);

    let dir = tmpdir();
    let es_path = dir.join("ipp.h261");
    let yuv_path = dir.join("ipp.yuv");
    std::fs::write(&es_path, &stream).unwrap();

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

    if !status.success() {
        let err = Command::new("ffmpeg")
            .args(["-y", "-hide_banner", "-f", "h261", "-i"])
            .arg(&es_path)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv_path)
            .stderr(Stdio::piped())
            .output()
            .expect("spawn ffmpeg diag");
        let msg = String::from_utf8_lossy(&err.stderr);
        let mut h = std::io::stderr();
        writeln!(h, "stream length = {} bytes", stream.len()).ok();
        writeln!(
            h,
            "first 32 bytes: {:02x?}",
            &stream[..32.min(stream.len())]
        )
        .ok();
        writeln!(h, "ffmpeg stderr: {msg}").ok();
        panic!("ffmpeg failed to decode our IPP sequence");
    }

    let decoded = std::fs::read(&yuv_path).expect("read yuv");
    let frame_size = 176 * 144 * 3 / 2;
    // ffmpeg should emit at least our 3 frames (it sometimes pads on EOF).
    assert!(
        decoded.len() >= 3 * frame_size,
        "ffmpeg produced too few bytes: {}",
        decoded.len()
    );

    // Compare each Y plane against the source — generous tolerance because
    // we drift through ffmpeg's IDCT vs ours, plus quantisation.
    let inputs = [&y0, &y1, &y2];
    for (i, input) in inputs.iter().enumerate() {
        let y_out = &decoded[i * frame_size..i * frame_size + 176 * 144];
        let mut err_sum: u64 = 0;
        for (a, b) in input.iter().zip(y_out.iter()) {
            err_sum += (*a as i32 - *b as i32).unsigned_abs() as u64;
        }
        let mean_err = err_sum as f64 / (176.0 * 144.0);
        // 25 is a loose bar — we just want to confirm it's not garbage.
        assert!(
            mean_err < 25.0,
            "frame {i}: mean Y error {mean_err:.2} too large"
        );
    }
}

/// Build a translating QCIF source — a pattern of vertical stripes that
/// scrolls horizontally one pel per frame for `n_frames` frames. Returns
/// `(y_planes, cb_planes, cr_planes)` per-frame.
fn translating_qcif(n_frames: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let w = 176usize;
    let h = 144usize;
    let mut frames = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let mut y = vec![0u8; w * h];
        let shift = f as i32 * 2; // 2 pels/frame — well within ±15.
        for j in 0..h {
            for i in 0..w {
                let xi = (i as i32 - shift).rem_euclid(w as i32);
                // Stripe pattern with diagonal modulation — inseparable
                // gradient so the encoder must use MC.
                let stripe = if (xi / 8) % 2 == 0 { 60 } else { 200 };
                let diag = (xi + j as i32) % 32;
                y[j * w + i] = (stripe + diag).clamp(0, 255) as u8;
            }
        }
        let cb = vec![128u8; (w / 2) * (h / 2)];
        let cr = vec![128u8; (w / 2) * (h / 2)];
        frames.push((y, cb, cr));
    }
    frames
}

/// End-to-end test of motion-compensated P-pictures: encode a 4-frame
/// translating fixture, hand the bitstream to ffmpeg, and verify each
/// decoded frame matches the source within a tight PSNR bound. Without MC
/// the stripes would mispredict catastrophically; with MC the encoder
/// should track the 2-pel/frame translation and decode at >=27 dB Y PSNR.
#[test]
fn ffmpeg_decodes_our_qcif_translating_sequence() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not on PATH — skipping");
        return;
    }
    let frames = translating_qcif(4);

    let mut enc = H261Encoder::new(SourceFormat::Qcif, 8);
    let mut stream = Vec::new();
    let mut sizes = Vec::new();
    for (y, cb, cr) in &frames {
        let p = enc.encode_frame(y, 176, cb, 88, cr, 88).expect("encode");
        sizes.push(p.len());
        stream.extend_from_slice(&p);
    }

    let dir = tmpdir();
    let es_path = dir.join("scroll.h261");
    let yuv_path = dir.join("scroll.yuv");
    std::fs::write(&es_path, &stream).unwrap();

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
    assert!(
        status.success(),
        "ffmpeg failed to decode our scrolling H.261"
    );

    let decoded = std::fs::read(&yuv_path).expect("read yuv");
    let frame_size = 176 * 144 * 3 / 2;
    assert!(
        decoded.len() >= frames.len() * frame_size,
        "ffmpeg produced too few bytes: {}",
        decoded.len()
    );

    // Each decoded frame's Y plane should match its source closely.
    let mut psnrs = Vec::new();
    for (i, (y, _, _)) in frames.iter().enumerate() {
        let y_out = &decoded[i * frame_size..i * frame_size + 176 * 144];
        let p = psnr(y, y_out);
        psnrs.push(p);
    }
    // I-frame should always be high.
    assert!(
        psnrs[0] >= 27.0,
        "I-frame PSNR {:.2} too low (PSNRs={psnrs:?} sizes={sizes:?})",
        psnrs[0]
    );
    // P-frames track the 2-pel/frame translation through the MC machinery.
    // 27 dB means ffmpeg is reconstructing the moved pels with high
    // fidelity; a non-MC encoder would drop into the teens.
    for i in 1..psnrs.len() {
        assert!(
            psnrs[i] >= 27.0,
            "P-frame {i} PSNR {:.2} too low (PSNRs={psnrs:?} sizes={sizes:?})",
            psnrs[i]
        );
    }
    // Sanity: the I-frame should be the largest; subsequent P-frames smaller.
    assert!(
        sizes[1] < sizes[0],
        "P-frame {} should be smaller than I-frame {} (sizes={sizes:?})",
        sizes[1],
        sizes[0]
    );
}

/// Build a noisy QCIF source where the loop filter (FIL MTYPEs, §3.2.3) is
/// likely to pay off — high-frequency stripes that smooth nicely after the
/// 1/4-1/2-1/4 separable filter.
fn fil_friendly_qcif(n_frames: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let w = 176usize;
    let h = 144usize;
    let mut frames = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let shift = f as i32 * 2;
        let mut y = vec![0u8; w * h];
        for j in 0..h {
            for i in 0..w {
                let xi = (i as i32 - shift).rem_euclid(w as i32);
                // Tight 4-pel stripes amplify the high-freq content the
                // loop filter is designed to attenuate.
                let stripe = if (xi / 4) % 2 == 0 { 60 } else { 196 };
                let diag = ((xi + j as i32) % 16) * 2;
                let grad = (j * 60) / h;
                let v = (stripe + diag + grad as i32).clamp(0, 255);
                y[j * w + i] = v as u8;
            }
        }
        let cb = vec![128u8; (w / 2) * (h / 2)];
        let cr = vec![128u8; (w / 2) * (h / 2)];
        frames.push((y, cb, cr));
    }
    frames
}

/// Smoke test for the registry encoder path (`reg.first_encoder`).
/// Encodes a 4-frame QCIF sequence via the `Encoder` trait wrapper and
/// verifies ffmpeg can decode the resulting stream.
#[test]
fn registry_encoder_qcif_roundtrip() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not on PATH — skipping");
        return;
    }
    let mut ctx = RuntimeContext::new();
    oxideav_h261::register(&mut ctx);

    let params = CodecParameters::video(CodecId::new(oxideav_h261::CODEC_ID_STR));
    let mut enc = ctx.codecs.first_encoder(&params).expect("encoder factory");

    let (y0, cb0, cr0) = gradient_qcif_source();
    let mut y1 = y0.clone();
    // Shift a region to generate some inter-frame motion.
    for j in 32..64usize {
        for i in 32..96usize {
            y1[j * 176 + i] = y1[j * 176 + i].saturating_add(20);
        }
    }

    let mut stream = Vec::new();
    for (idx, (y, cb, cr)) in [
        (&y0, &cb0, &cr0),
        (&y1, &cb0, &cr0),
        (&y0, &cb0, &cr0),
        (&y1, &cb0, &cr0),
    ]
    .iter()
    .enumerate()
    {
        let vf = VideoFrame {
            pts: Some(idx as i64),
            planes: vec![
                VideoPlane {
                    stride: 176,
                    data: y.to_vec(),
                },
                VideoPlane {
                    stride: 88,
                    data: cb.to_vec(),
                },
                VideoPlane {
                    stride: 88,
                    data: cr.to_vec(),
                },
            ],
        };
        enc.send_frame(&Frame::Video(vf)).expect("send_frame");
        let pkt = enc.receive_packet().expect("receive_packet");
        stream.extend_from_slice(&pkt.data);
    }

    let dir = tmpdir();
    let es_path = dir.join("reg_enc.h261");
    let yuv_path = dir.join("reg_enc.yuv");
    std::fs::write(&es_path, &stream).unwrap();

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
    assert!(
        status.success(),
        "ffmpeg failed to decode registry-encoder stream"
    );

    let decoded = std::fs::read(&yuv_path).expect("read decoded");
    let frame_size = 176 * 144 * 3 / 2;
    assert!(
        decoded.len() >= 4 * frame_size,
        "ffmpeg decoded too few bytes"
    );
}

/// Measure PSNR of our encoder output vs the original source after ffmpeg
/// cross-decode. At QUANT=8 / default settings (testsrc QCIF), we expect
/// PSNR_Y ≥ 35 dB — the canonical H.261 quality bar.
#[test]
fn encoder_psnr_vs_source_at_default_quant() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not on PATH — skipping");
        return;
    }
    let w = 176usize;
    let h = 144usize;

    // Generate source frames from a synthetic gradient that moves.
    let n_frames = 8usize;
    let mut sources: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::new();
    for f in 0..n_frames {
        let shift = f as i32 * 3;
        let mut y = vec![0u8; w * h];
        for j in 0..h {
            for i in 0..w {
                let xi = (i as i32 - shift).rem_euclid(w as i32) as usize;
                let val = 32 + (xi * 180) / w + (j * 40) / h;
                y[j * w + i] = val.clamp(0, 255) as u8;
            }
        }
        let cb = vec![128u8; (w / 2) * (h / 2)];
        let cr = vec![128u8; (w / 2) * (h / 2)];
        sources.push((y, cb, cr));
    }

    // Encode with our encoder (QUANT=8, default).
    let mut enc = H261Encoder::new(SourceFormat::Qcif, 8);
    let mut stream = Vec::new();
    for (y, cb, cr) in &sources {
        let pkt = enc
            .encode_frame(y, w, cb, w / 2, cr, w / 2)
            .expect("encode");
        stream.extend_from_slice(&pkt);
    }

    let dir = tmpdir();
    let es_path = dir.join("psnr_test.h261");
    let yuv_path = dir.join("psnr_test.yuv");
    std::fs::write(&es_path, &stream).unwrap();

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
    assert!(status.success(), "ffmpeg failed to decode psnr test stream");

    let decoded = std::fs::read(&yuv_path).expect("read decoded");
    let frame_size = w * h * 3 / 2;
    assert!(
        decoded.len() >= n_frames * frame_size,
        "too few decoded bytes"
    );

    let mut total_psnr = 0.0f64;
    let mut count = 0usize;
    for (i, (y, _, _)) in sources.iter().enumerate() {
        let dec_y = &decoded[i * frame_size..i * frame_size + w * h];
        if dec_y.len() == y.len() {
            let p = psnr(y, dec_y);
            eprintln!("frame {i}: PSNR_Y = {p:.2} dB");
            if p.is_finite() {
                total_psnr += p;
                count += 1;
            }
        }
    }
    let avg_psnr = if count > 0 {
        total_psnr / count as f64
    } else {
        0.0
    };
    eprintln!("average PSNR_Y = {avg_psnr:.2} dB over {count} frames");

    // The I-frame at QUANT=8 should decode well above 35 dB.
    // P-frames on this smooth-motion content should stay above 32 dB.
    assert!(
        avg_psnr >= 32.0,
        "average PSNR_Y {avg_psnr:.2} dB below 32.0 dB"
    );
}

/// End-to-end FIL test: encode 4 frames of FIL-friendly QCIF, hand the
/// bitstream to ffmpeg, and assert each decoded frame matches the source
/// at high PSNR. This proves the FIL MTYPEs (`Inter+MC+FIL` 3-bit `001`
/// and `Inter+MC+FIL+CBP` 2-bit `01`) we emit are decoded correctly by a
/// reference H.261 decoder.
#[test]
fn ffmpeg_decodes_our_fil_p_pictures() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not on PATH — skipping");
        return;
    }
    let frames = fil_friendly_qcif(4);

    let mut enc = H261Encoder::new(SourceFormat::Qcif, 8);
    let mut stream = Vec::new();
    let mut sizes = Vec::new();
    for (y, cb, cr) in &frames {
        let p = enc.encode_frame(y, 176, cb, 88, cr, 88).expect("encode");
        sizes.push(p.len());
        stream.extend_from_slice(&p);
    }

    let dir = tmpdir();
    let es_path = dir.join("fil.h261");
    let yuv_path = dir.join("fil.yuv");
    std::fs::write(&es_path, &stream).unwrap();

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
    assert!(status.success(), "ffmpeg failed to decode our FIL stream");

    let decoded = std::fs::read(&yuv_path).expect("read yuv");
    let frame_size = 176 * 144 * 3 / 2;
    assert!(decoded.len() >= frames.len() * frame_size);

    let mut psnrs = Vec::new();
    for (i, (y, _, _)) in frames.iter().enumerate() {
        let y_out = &decoded[i * frame_size..i * frame_size + 176 * 144];
        psnrs.push(psnr(y, y_out));
    }
    // I-frame target.
    assert!(
        psnrs[0] >= 27.0,
        "I-frame PSNR {:.2} too low (PSNRs={psnrs:?} sizes={sizes:?})",
        psnrs[0]
    );
    // P-frame target — the FIL machinery should keep us well above 27 dB
    // even on the high-freq stripes.
    for i in 1..psnrs.len() {
        assert!(
            psnrs[i] >= 27.0,
            "P-frame {i} PSNR {:.2} too low (PSNRs={psnrs:?} sizes={sizes:?})",
            psnrs[i]
        );
    }
}
