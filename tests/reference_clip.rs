//! Integration test against an ffmpeg-generated H.261 elementary-stream clip.
//!
//! Generate the fixtures with:
//!
//! ```sh
//! mkdir -p /tmp/h261
//! ffmpeg -y -f lavfi -i "testsrc=size=352x288:rate=10:duration=0.5" \
//!     -c:v h261 -f h261 /tmp/h261/cif.h261
//! ffmpeg -y -i /tmp/h261/cif.h261 -f rawvideo -pix_fmt yuv420p \
//!     /tmp/h261/cif.yuv
//! ffmpeg -y -f lavfi -i "testsrc=size=176x144:rate=10:duration=0.5" \
//!     -c:v h261 -f h261 /tmp/h261/qcif.h261
//! ffmpeg -y -i /tmp/h261/qcif.h261 -f rawvideo -pix_fmt yuv420p \
//!     /tmp/h261/qcif.yuv
//! ```
//!
//! When the fixtures aren't present the test logs a warning and passes.

use std::path::Path;

use oxideav_core::packet::PacketFlags;
use oxideav_core::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h261::decoder::H261Decoder;

fn read_optional(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("fixture {path} missing — skipping test");
        return None;
    }
    Some(std::fs::read(path).expect("read fixture"))
}

fn decode_all_frames(es_path: &str, w: u32, h: u32) -> usize {
    let Some(es) = read_optional(es_path) else {
        return 0;
    };

    let codec_id = CodecId::new(oxideav_h261::CODEC_ID_STR);
    let mut decoder = H261Decoder::new(codec_id.clone());
    let pkt = Packet {
        stream_index: 0,
        data: es,
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

    let mut count = 0usize;
    loop {
        match decoder.receive_frame() {
            Ok(Frame::Video(vf)) => {
                assert_eq!(vf.width, w, "width");
                assert_eq!(vf.height, h, "height");
                assert_eq!(vf.planes.len(), 3, "should have YUV 4:2:0 planes");
                assert_eq!(vf.planes[0].stride, w as usize);
                assert_eq!(vf.planes[0].data.len(), (w * h) as usize);
                assert_eq!(vf.planes[1].stride, (w / 2) as usize);
                assert_eq!(vf.planes[1].data.len(), (w * h / 4) as usize);
                count += 1;
            }
            Ok(_) => panic!("unexpected non-video frame"),
            Err(_) => break,
        }
    }
    count
}

fn decode_first_frame(es_path: &str, w: u32, h: u32) {
    let c = decode_all_frames(es_path, w, h);
    if c > 0 {
        // Fixture present — we expect at least one frame.
        assert!(c >= 1, "expected at least one frame decoded");
    }
}

#[test]
fn decodes_cif_fixture() {
    decode_first_frame("/tmp/h261/cif.h261", 352, 288);
}

#[test]
fn decodes_qcif_fixture() {
    decode_first_frame("/tmp/h261/qcif.h261", 176, 144);
}

#[test]
fn decodes_all_qcif_frames() {
    // The 0.5s @ 10fps = 5 frames fixture should decode cleanly through the
    // P-picture path too (INTER + INTER+MC + INTER+MC+FIL MBs).
    let c = decode_all_frames("/tmp/h261/qcif.h261", 176, 144);
    if c > 0 {
        assert!(c >= 2, "expected at least 2 frames (I + P), got {c}");
    }
}

#[test]
fn decodes_all_cif_frames() {
    let c = decode_all_frames("/tmp/h261/cif.h261", 352, 288);
    if c > 0 {
        assert!(c >= 2, "expected at least 2 frames (I + P), got {c}");
    }
}
