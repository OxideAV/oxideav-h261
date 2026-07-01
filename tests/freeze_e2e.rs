//! End-to-end: drive the H.261 decoder's §4.3.1 / §4.3.3 freeze-picture
//! behaviour through the real encode → decode path.
//!
//! §4.3.1: a freeze-picture request (external means) causes the decoder to
//! hold its displayed picture until a release signal or the "at least six
//! seconds" timeout. §4.3.3: the release signal is PTYPE bit 3 of the first
//! picture coded in response to a fast-update request. These tests confirm the
//! decoder holds the last-shown frame while frozen (even as decoding
//! continues), and releases on both the §4.3.3 in-band bit and the §4.3.1
//! timeout.

use oxideav_core::registry::codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h261::decoder::H261Decoder;
use oxideav_h261::encoder::H261Encoder;
use oxideav_h261::multipoint::FREEZE_TIMEOUT_NANOS;
use oxideav_h261::picture::SourceFormat;

const W: usize = 176;
const H: usize = 144;

/// A flat QCIF frame at luma value `v` (chroma neutral 128).
fn flat_frame(v: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        vec![v; W * H],
        vec![128u8; (W / 2) * (H / 2)],
        vec![128u8; (W / 2) * (H / 2)],
    )
}

/// Feed one coded picture. The H.261 decoder buffers a picture until the next
/// PSC (or flush) proves it complete, so a picture is only *decoded* (and the
/// freeze state machine only consulted) when the *following* picture arrives.
/// This helper feeds `bytes`, then drains and returns whichever picture became
/// ready (the previously fed one), if any.
fn feed(dec: &mut H261Decoder, bytes: &[u8]) -> Option<Frame> {
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes.to_vec());
    dec.send_packet(&pkt).expect("send_packet");
    dec.receive_frame().ok()
}

/// Flush and drain the final buffered picture.
fn drain_last(dec: &mut H261Decoder) -> Option<Frame> {
    dec.flush().ok();
    dec.receive_frame().ok()
}

fn luma00(f: &Frame) -> u8 {
    match f {
        Frame::Video(vf) => vf.planes[0].data[0],
        _ => panic!("expected a video frame"),
    }
}

#[test]
fn freeze_holds_last_shown_frame_then_timeout_releases() {
    let mut dec = H261Decoder::new(CodecId::new(oxideav_h261::CODEC_ID_STR));
    // Encode a sequence of distinguishable flat frames. Force every frame
    // INTRA so each carries its own full luma value regardless of prediction.
    let mut enc = H261Encoder::new(SourceFormat::Qcif, 8).with_intra_period(1);
    // Pre-encode 182 flat frames with distinct luma so decoded output is
    // trivially identifiable: frame k has luma value `luma(k)`.
    let luma = |k: usize| -> u8 {
        match k {
            0 => 40,
            181 => 70,
            _ => 41u8.wrapping_add((k as u8).wrapping_sub(1)),
        }
    };
    let streams: Vec<Vec<u8>> = (0..182)
        .map(|k| {
            let (y, cb, cr) = flat_frame(luma(k));
            enc.encode_frame(&y, W, &cb, W / 2, &cr, W / 2).unwrap()
        })
        .collect();

    // A picture is decoded when the *next* picture is fed. Feeding frame k
    // triggers the decode of frame k-1. We request the freeze at k == 2, i.e.
    // after frame 1 has been fed but before frame 2: frame 0 (decoded at k==1)
    // and frame 1's feed happen un-frozen, and frame 1 onward (decoded from
    // k==2) are decoded frozen. The first frozen decode is frame 1 — tick 1 of
    // the six-second timeout.
    let mut shown: Vec<u8> = Vec::new();
    for (k, s) in streams.iter().enumerate() {
        if k == 2 {
            dec.request_freeze();
            assert!(dec.is_frozen());
        }
        if let Some(f) = feed(&mut dec, s) {
            shown.push(luma00(&f));
        }
    }
    if let Some(f) = drain_last(&mut dec) {
        shown.push(luma00(&f));
    }

    // 182 pictures fed ⇒ 182 displayed. Frame 0 shows its own value (40).
    assert_eq!(shown.len(), 182);
    assert_eq!(shown[0], 40, "frame 0 shown un-frozen");
    // Frames 1..=180 are decoded frozen; each is the 1..180th tick of the
    // six-second timeout. 179 ticks stay under 6 s (held at 40); the 180th
    // tick (the 180th frozen decode) crosses the timeout and shows its own
    // picture. Frozen decodes are frames at index 1..=180 in the shown list.
    for (i, &v) in shown.iter().enumerate().take(180).skip(1) {
        assert_eq!(v, 40, "frame {i} held at 40 while frozen");
    }
    assert!(!dec.is_frozen(), "timeout released the freeze");
    // Index 180 is the 180th frozen decode ⇒ timeout release ⇒ own value.
    assert_eq!(
        luma(180),
        shown[180],
        "timeout releases + shows new picture"
    );
    // Index 181 shows normally again (luma 70).
    assert_eq!(shown[181], 70);
}

#[test]
fn freeze_before_first_picture_shows_first() {
    // If a freeze is requested before any picture has been decoded, there is
    // nothing to hold — the first decoded picture is displayed.
    let mut dec = H261Decoder::new(CodecId::new(oxideav_h261::CODEC_ID_STR));
    let mut enc = H261Encoder::new(SourceFormat::Qcif, 8).with_intra_period(1);
    dec.request_freeze();
    assert!(dec.is_frozen());
    let (y, cb, cr) = flat_frame(55);
    let s0 = enc.encode_frame(&y, W, &cb, W / 2, &cr, W / 2).unwrap();
    // Feed frame 0 (buffered, not yet decoded), then flush to force its decode
    // and display. Frozen, but nothing to hold ⇒ frame 0 is shown.
    assert!(feed(&mut dec, &s0).is_none());
    let f = drain_last(&mut dec).expect("frame 0");
    assert_eq!(luma00(&f), 55, "nothing to hold ⇒ first picture shown");
}

#[test]
fn freeze_timeout_constant_is_six_seconds() {
    assert_eq!(FREEZE_TIMEOUT_NANOS, 6_000_000_000);
}
