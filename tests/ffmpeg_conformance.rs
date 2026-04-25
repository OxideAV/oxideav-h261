//! PSNR-based conformance test against ffmpeg.
//!
//! Strategy (all in one process, leaves no lingering state):
//!
//! 1. Generate a short YUV420P source clip with `ffmpeg -f lavfi -i testsrc`.
//! 2. Encode it with `ffmpeg -c:v h261 -f h261` at QCIF and CIF.
//! 3. Decode the resulting H.261 elementary stream two ways:
//!    * with the pure-Rust `H261Decoder`
//!    * with `ffmpeg -f h261 -i ... -f rawvideo -pix_fmt yuv420p`
//! 4. Compute Y-plane PSNR between the two decodes and assert it is above
//!    a generous threshold (35 dB is the usual H.261 bar — IDCT rounding
//!    tolerances, §4.2.4.1, make bit-exact impossible).
//!
//! Skipped gracefully when `ffmpeg` isn't on PATH.
//!
//! Spec refs:
//! * ITU-T Rec. H.261 (03/93) §3, §4.2 — bitstream syntax.
//! * Annex A of H.261 — IDCT accuracy tolerances (why PSNR, not bit-exact).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use oxideav_core::packet::PacketFlags;
use oxideav_core::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h261::decoder::H261Decoder;

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("oxideav-h261-conf-{tag}"));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Run `ffmpeg -f lavfi -i <source> -t <duration> <extra> -c:v h261 -f h261 <out>`.
///
/// `source` is a lavfi filter spec like `testsrc=size=176x144:rate=10`. The
/// `-t <duration>` goes outside the filter so that it also works with filters
/// (e.g. `mandelbrot`) that don't accept a `duration=` option internally.
fn ffmpeg_encode_cmd(
    dir: &Path,
    source: &str,
    duration: f32,
    extra_encode_args: &[&str],
) -> Option<PathBuf> {
    let es_path = dir.join("clip.h261");
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
    ])
    .arg(source)
    .args(["-t", &format!("{duration}")]);
    for a in extra_encode_args {
        cmd.arg(a);
    }
    cmd.args(["-c:v", "h261", "-f", "h261"]).arg(&es_path);
    let status = cmd.status().ok()?;
    if !status.success() {
        return None;
    }
    Some(es_path)
}

fn ffmpeg_encode(dir: &Path, source: &str, duration: f32) -> Option<PathBuf> {
    ffmpeg_encode_cmd(dir, source, duration, &[])
}

fn ffmpeg_encode_with_q(dir: &Path, source: &str, duration: f32, qscale: u32) -> Option<PathBuf> {
    let q = qscale.to_string();
    ffmpeg_encode_cmd(dir, source, duration, &["-qscale:v", &q])
}

/// Force every frame to be an I-picture (`-g 1`). Eliminates P-frame IDCT
/// mismatch drift so we can test pure INTRA reconstruction fidelity.
fn ffmpeg_encode_intra_only(dir: &Path, source: &str, duration: f32) -> Option<PathBuf> {
    ffmpeg_encode_cmd(dir, source, duration, &["-g", "1"])
}

fn ffmpeg_decode(es_path: &Path, dir: &Path) -> Option<PathBuf> {
    let out = dir.join("ref.yuv");
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
        .arg(es_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&out)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    Some(out)
}

fn oxideav_decode(es: &[u8]) -> Vec<Vec<u8>> {
    // Return a flat per-frame YUV420P blob (Y then Cb then Cr, tightly packed).
    let codec_id = CodecId::new(oxideav_h261::CODEC_ID_STR);
    let mut decoder = H261Decoder::new(codec_id);
    let pkt = Packet {
        stream_index: 0,
        data: es.to_vec(),
        pts: Some(0),
        dts: Some(0),
        duration: None,
        time_base: TimeBase::new(1, 30_000),
        flags: PacketFlags {
            keyframe: true,
            ..Default::default()
        },
    };
    decoder.send_packet(&pkt).expect("send_packet");
    decoder.flush().ok();

    let mut frames = Vec::new();
    loop {
        match decoder.receive_frame() {
            Ok(Frame::Video(vf)) => {
                let mut buf = Vec::new();
                // Packed planes without stride padding — dims are always MB-multiples for H.261.
                for plane in &vf.planes {
                    buf.extend_from_slice(&plane.data);
                }
                frames.push(buf);
            }
            Ok(_) => panic!("unexpected non-video frame"),
            Err(_) => break,
        }
    }
    frames
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return f64::INFINITY;
    }
    let mut sse: f64 = 0.0;
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

fn run_case_with_encoder(
    tag: &str,
    w: u32,
    h: u32,
    _duration: f32,
    min_psnr_db: f64,
    encode: impl FnOnce(&Path) -> Option<PathBuf>,
) {
    if !have_ffmpeg() {
        eprintln!("{tag}: ffmpeg not on PATH — skipping");
        return;
    }
    let dir = tmpdir(tag);
    let Some(es_path) = encode(&dir) else {
        eprintln!("{tag}: ffmpeg encode failed — skipping");
        return;
    };
    let Some(ref_yuv) = ffmpeg_decode(&es_path, &dir) else {
        eprintln!("{tag}: ffmpeg decode failed — skipping");
        return;
    };

    let es = std::fs::read(&es_path).expect("read es");
    let our_frames = oxideav_decode(&es);
    assert!(
        !our_frames.is_empty(),
        "{tag}: oxideav decoded zero frames from {} byte stream",
        es.len()
    );

    let ref_data = std::fs::read(&ref_yuv).expect("read ref yuv");
    let frame_size = (w as usize) * (h as usize) * 3 / 2;
    assert_eq!(
        ref_data.len() % frame_size,
        0,
        "{tag}: ffmpeg yuv file isn't a multiple of frame size"
    );
    let ref_frame_count = ref_data.len() / frame_size;

    // Require the frame counts match — H.261 has no explicit EOF signal
    // but both decoders should agree when the whole stream is fed in one go.
    assert_eq!(
        our_frames.len(),
        ref_frame_count,
        "{tag}: frame count mismatch (our={}, ref={})",
        our_frames.len(),
        ref_frame_count
    );

    // Compute per-frame Y-plane PSNR and take the minimum.
    let y_size = (w as usize) * (h as usize);
    let c_size = y_size / 4;
    let mut min_y_psnr = f64::INFINITY;
    let mut min_cb_psnr = f64::INFINITY;
    let mut min_cr_psnr = f64::INFINITY;
    for (i, our) in our_frames.iter().enumerate() {
        assert_eq!(our.len(), frame_size, "{tag}: frame {i} wrong size");
        let ref_frame = &ref_data[i * frame_size..(i + 1) * frame_size];
        let y_psnr = psnr(&our[..y_size], &ref_frame[..y_size]);
        let cb_psnr = psnr(
            &our[y_size..y_size + c_size],
            &ref_frame[y_size..y_size + c_size],
        );
        let cr_psnr = psnr(&our[y_size + c_size..], &ref_frame[y_size + c_size..]);
        eprintln!("{tag}: frame {i}: Y={y_psnr:.2} dB, Cb={cb_psnr:.2} dB, Cr={cr_psnr:.2} dB");
        min_y_psnr = min_y_psnr.min(y_psnr);
        min_cb_psnr = min_cb_psnr.min(cb_psnr);
        min_cr_psnr = min_cr_psnr.min(cr_psnr);
    }

    // IDCT Annex A tolerances mean a handful of +-1 pel drifts between
    // our decoder and ffmpeg's are expected. 35 dB on a testsrc pattern
    // means ~4.5 / 255 RMS per pel, comfortably inside spec. The chroma
    // planes get more headroom (smaller) so we use a slightly lower bar.
    assert!(
        min_y_psnr >= min_psnr_db,
        "{tag}: worst-frame Y PSNR {min_y_psnr:.2} dB < {min_psnr_db} dB"
    );
    assert!(
        min_cb_psnr >= min_psnr_db - 3.0,
        "{tag}: worst-frame Cb PSNR {min_cb_psnr:.2} dB < {:.1} dB",
        min_psnr_db - 3.0
    );
    assert!(
        min_cr_psnr >= min_psnr_db - 3.0,
        "{tag}: worst-frame Cr PSNR {min_cr_psnr:.2} dB < {:.1} dB",
        min_psnr_db - 3.0
    );
}

fn run_case(tag: &str, w: u32, h: u32, duration: f32, min_psnr_db: f64) {
    run_case_with_encoder(tag, w, h, duration, min_psnr_db, |dir| {
        ffmpeg_encode(dir, &format!("testsrc=size={w}x{h}:rate=10"), duration)
    });
}

#[test]
fn ffmpeg_conformance_qcif() {
    // 1.2s @ 10fps -> 12 frames of I + P.
    run_case("qcif", 176, 144, 1.2, 35.0);
}

#[test]
fn ffmpeg_conformance_cif() {
    run_case("cif", 352, 288, 1.2, 35.0);
}

#[test]
fn ffmpeg_conformance_qcif_long() {
    // Longer run exercises MV predictor / loop-filter paths more thoroughly.
    run_case("qcif_long", 176, 144, 3.0, 35.0);
}

#[test]
fn ffmpeg_conformance_qcif_mandelbrot_intra() {
    // Mandelbrot has heavy spatial detail — stresses TCOEFF VLC + escape.
    // Intra-only so there's no P-frame IDCT-mismatch drift to confuse the
    // PSNR signal (H.261 Annex A explicitly tolerates IDCT divergence
    // between decoders, which otherwise accumulates across P-frames).
    run_case_with_encoder("qcif_mbrot_intra", 176, 144, 1.2, 35.0, |dir| {
        ffmpeg_encode_intra_only(dir, "mandelbrot=size=176x144:rate=10", 1.2)
    });
}

#[test]
fn ffmpeg_conformance_cif_mandelbrot_intra() {
    run_case_with_encoder("cif_mbrot_intra", 352, 288, 1.2, 35.0, |dir| {
        ffmpeg_encode_intra_only(dir, "mandelbrot=size=352x288:rate=10", 1.2)
    });
}

#[test]
fn ffmpeg_conformance_qcif_mandelbrot_ipp() {
    // Same mandelbrot but with the default GOP (only 1 INTRA refresh),
    // so drift from the per-decoder IDCT rounding accumulates across the
    // P-chain. H.261 Annex A permits this — we expect some decay but
    // not catastrophic. 25 dB is a safe floor for this adversarial case
    // and still proves that motion compensation + residual addition are
    // structurally correct (vs. a real bug, which would blow up PSNR).
    run_case_with_encoder("qcif_mbrot_ipp", 176, 144, 1.2, 25.0, |dir| {
        ffmpeg_encode(dir, "mandelbrot=size=176x144:rate=10", 1.2)
    });
}

#[test]
fn ffmpeg_conformance_qcif_high_qscale() {
    // QP near the top (quant = 30/31 is the H.261 max) stresses the
    // dequant path — any off-by-one in Table 5 quant mapping shows up
    // as banding. We keep the PSNR bar the same since the *difference*
    // between our decoder and ffmpeg's should still be tiny, even if
    // absolute reconstruction quality versus source is poor.
    run_case_with_encoder("qcif_q28", 176, 144, 1.0, 35.0, |dir| {
        ffmpeg_encode_with_q(dir, "testsrc=size=176x144:rate=10", 1.0, 28)
    });
}

#[test]
fn ffmpeg_conformance_qcif_low_qscale() {
    // Low QP (high quality) = many transmitted MBs and lots of VLC escapes
    // with large coefficients; exercises the TCOEFF FLC escape code path.
    run_case_with_encoder("qcif_q2", 176, 144, 1.0, 35.0, |dir| {
        ffmpeg_encode_with_q(dir, "testsrc=size=176x144:rate=10", 1.0, 2)
    });
}
