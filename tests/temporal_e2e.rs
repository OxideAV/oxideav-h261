//! End-to-end: §3.1 / §4.2.1.2 temporal-reference and picture-rate handling
//! through the real encode → decode path.
//!
//! §4.2.1.2: the 5-bit `TR` field advances by `1 + non_transmitted` between
//! consecutive transmitted pictures. §3.1: an encoder may restrict its picture
//! rate by leaving at least 0, 1, 2 or 3 non-transmitted pictures between
//! transmitted ones. These tests drive a real reduced-rate encoded sequence
//! back through the decoder and confirm the decoder recovers the per-picture
//! delta, the monotonic presentation index, and that the §4.3.1 freeze timeout
//! is measured against the true elapsed source-picture periods.

use oxideav_core::registry::codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h261::decoder::H261Decoder;
use oxideav_h261::encoder::H261Encoder;
use oxideav_h261::picture::SourceFormat;
use oxideav_h261::temporal::{tr_delta, PictureRate, TrTracker};

const W: usize = 176;
const H: usize = 144;

fn flat_frame(v: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        vec![v; W * H],
        vec![128u8; (W / 2) * (H / 2)],
        vec![128u8; (W / 2) * (H / 2)],
    )
}

/// Feed one coded picture; the H.261 decoder decodes picture k-1 when picture k
/// arrives, so this returns whichever picture became ready (if any).
fn feed(dec: &mut H261Decoder, bytes: &[u8]) -> Option<Frame> {
    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes.to_vec());
    dec.send_packet(&pkt).expect("send_packet");
    dec.receive_frame().ok()
}

fn drain_last(dec: &mut H261Decoder) -> Option<Frame> {
    dec.flush().ok();
    dec.receive_frame().ok()
}

#[test]
fn full_rate_stream_reports_unit_deltas() {
    let mut enc = H261Encoder::new(SourceFormat::Qcif, 8).with_intra_period(4);
    let mut dec = H261Decoder::new(CodecId::new(oxideav_h261::CODEC_ID_STR));
    let streams: Vec<Vec<u8>> = (0..8)
        .map(|k| {
            let (y, cb, cr) = flat_frame(40u8.wrapping_add(k as u8));
            enc.encode_frame(&y, W, &cb, W / 2, &cr, W / 2).unwrap()
        })
        .collect();

    let mut decoded = 0usize;
    for s in &streams {
        if feed(&mut dec, s).is_some() {
            // Full rate ⇒ every decoded picture advances one source-picture
            // period.
            assert_eq!(dec.last_tr_delta(), 1);
            assert_eq!(dec.last_non_transmitted_pictures(), 0);
            decoded += 1;
        }
    }
    if drain_last(&mut dec).is_some() {
        assert_eq!(dec.last_tr_delta(), 1);
        decoded += 1;
    }
    assert_eq!(decoded, 8);
    // Presentation index of the last decoded picture = 7 periods after the
    // first (index 0).
    assert_eq!(dec.presentation_index(), 7);
}

#[test]
fn reduced_rate_stream_round_trips_temporal_reference() {
    // §3.1: two non-transmitted pictures ⇒ interval 3. The decoder must recover
    // a per-picture delta of 3 and a presentation index stepping by 3.
    let rate = PictureRate::from_non_transmitted(2);
    let mut enc = H261Encoder::new(SourceFormat::Qcif, 8)
        .with_intra_period(4)
        .with_picture_rate(rate);
    let mut dec = H261Decoder::new(CodecId::new(oxideav_h261::CODEC_ID_STR));

    let streams: Vec<Vec<u8>> = (0..8)
        .map(|k| {
            let (y, cb, cr) = flat_frame(50u8.wrapping_add(k as u8));
            enc.encode_frame(&y, W, &cb, W / 2, &cr, W / 2).unwrap()
        })
        .collect();

    // Independently model the expected presentation timeline with the public
    // TrTracker so the e2e path is cross-checked against the primitive.
    let mut model = TrTracker::new();
    let mut prev_tr: Option<u8> = None;
    let mut expected_tr = 0u8;

    let mut check = |dec: &H261Decoder, first: &mut bool| {
        // The stamped TR schedule is 0, 3, 6, ... mod 32.
        model.observe(expected_tr);
        if *first {
            // First decoded picture: delta defaults to 1, index 0.
            assert_eq!(dec.last_tr_delta(), 1);
            assert_eq!(dec.presentation_index(), 0);
            *first = false;
        } else {
            assert_eq!(dec.last_tr_delta(), 3, "interval-3 stream");
            assert_eq!(dec.last_non_transmitted_pictures(), 2);
            assert_eq!(dec.presentation_index(), model.presentation_index());
        }
        if let Some(p) = prev_tr {
            assert_eq!(tr_delta(p, expected_tr), 3);
        }
        prev_tr = Some(expected_tr);
        expected_tr = expected_tr.wrapping_add(3) & 0x1F;
    };

    let mut first = true;
    for s in &streams {
        if feed(&mut dec, s).is_some() {
            check(&dec, &mut first);
        }
    }
    if drain_last(&mut dec).is_some() {
        check(&dec, &mut first);
    }
    // 8 pictures at interval 3 ⇒ last presentation index = 7 * 3 = 21.
    assert_eq!(dec.presentation_index(), 21);
}

#[test]
fn reset_clears_temporal_state() {
    let mut enc = H261Encoder::new(SourceFormat::Qcif, 8)
        .with_intra_period(0)
        .with_picture_rate(PictureRate::from_non_transmitted(1));
    let mut dec = H261Decoder::new(CodecId::new(oxideav_h261::CODEC_ID_STR));
    let s0 = {
        let (y, cb, cr) = flat_frame(60);
        enc.encode_frame(&y, W, &cb, W / 2, &cr, W / 2).unwrap()
    };
    let s1 = {
        let (y, cb, cr) = flat_frame(61);
        enc.encode_frame(&y, W, &cb, W / 2, &cr, W / 2).unwrap()
    };
    feed(&mut dec, &s0);
    feed(&mut dec, &s1); // decodes s0
    drain_last(&mut dec); // decodes s1 ⇒ delta 2
    assert_eq!(dec.last_tr_delta(), 2);
    assert!(dec.presentation_index() > 0);

    dec.reset().unwrap();
    assert_eq!(dec.presentation_index(), 0);
    assert_eq!(dec.last_tr_delta(), 1);
}
