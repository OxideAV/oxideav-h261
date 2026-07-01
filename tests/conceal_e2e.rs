//! End-to-end: §2.7 / §2.8 error concealment.
//!
//! §2.7 makes the outer BCH FEC layer optional; the GOB start-code structure
//! (§2.8) is the recovery mechanism. A decoder in concealment mode resyncs at
//! the next GBSC when a GOB fails to decode, concealing the damaged GOB from
//! the reference instead of discarding the whole picture. These tests corrupt
//! a coded P-picture and confirm concealment recovers a full frame while a
//! strict decode rejects it.

use oxideav_core::registry::codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h261::decoder::H261Decoder;
use oxideav_h261::encoder::H261Encoder;
use oxideav_h261::picture::SourceFormat;

const W: usize = 176;
const H: usize = 144;

fn flat_frame(v: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        vec![v; W * H],
        vec![128u8; (W / 2) * (H / 2)],
        vec![128u8; (W / 2) * (H / 2)],
    )
}

/// Concatenate two coded pictures (I then P) into one elementary stream.
fn make_i_then_p() -> (Vec<u8>, Vec<u8>) {
    let mut enc = H261Encoder::new(SourceFormat::Qcif, 8).with_intra_period(0);
    // I-picture: flat 40.
    let (y0, cb0, cr0) = flat_frame(40);
    let s_i = enc.encode_frame(&y0, W, &cb0, W / 2, &cr0, W / 2).unwrap();
    // P-picture: a visibly different image (gradient) so a coded (non-skipped)
    // GOB carries real residual, making a corruption land on decodable data.
    let mut y1 = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            y1[j * W + i] = (20 + (i * 200) / W) as u8;
        }
    }
    let cb1 = vec![128u8; (W / 2) * (H / 2)];
    let cr1 = vec![128u8; (W / 2) * (H / 2)];
    let s_p = enc.encode_frame(&y1, W, &cb1, W / 2, &cr1, W / 2).unwrap();
    (s_i, s_p)
}

fn decode_all(dec: &mut H261Decoder, stream: &[u8]) -> Vec<Frame> {
    let pkt = Packet::new(0, TimeBase::new(1, 30), stream.to_vec());
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().ok();
    let mut out = Vec::new();
    while let Ok(f) = dec.receive_frame() {
        out.push(f);
    }
    out
}

/// Corrupt a byte in the middle of the P-picture's coded body (past its
/// picture + first GOB header) so a GOB fails to decode.
fn corrupt_mid(p: &[u8]) -> Vec<u8> {
    let mut c = p.to_vec();
    // Flip several bytes near the middle to reliably derail a VLC decode
    // inside a GOB without hitting a start code.
    let mid = c.len() / 2;
    for k in 0..3 {
        let idx = mid + k;
        if idx < c.len() {
            c[idx] ^= 0xFF;
        }
    }
    c
}

#[test]
fn strict_decode_rejects_corrupted_gob_but_conceal_recovers() {
    let (s_i, s_p) = make_i_then_p();
    let corrupted_p = corrupt_mid(&s_p);

    // Build I + corrupted-P + a trailing I (its PSC bounds the P for the
    // decoder, and gives us a clean picture after the damaged one).
    let mut enc2 = H261Encoder::new(SourceFormat::Qcif, 8).with_intra_period(1);
    let (yt, cbt, crt) = flat_frame(60);
    let s_tail = enc2.encode_frame(&yt, W, &cbt, W / 2, &crt, W / 2).unwrap();

    let mut stream = Vec::new();
    stream.extend_from_slice(&s_i);
    stream.extend_from_slice(&corrupted_p);
    stream.extend_from_slice(&s_tail);

    // Strict decoder: the corrupted P must surface an error. The decoder
    // decodes eagerly inside `send_packet`, so the leading I is produced but
    // the corrupted P aborts the batch — the error surfaces at `send_packet`
    // (or `flush`), and the corrupted picture never yields a frame.
    let mut strict = H261Decoder::new(CodecId::new(oxideav_h261::CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 30), stream.clone());
    let send_res = strict.send_packet(&pkt);
    let flush_res = strict.flush();
    assert!(
        send_res.is_err() || flush_res.is_err(),
        "strict decode must reject the corrupted P-picture"
    );

    // Concealing decoder: recovers a full frame for the corrupted P and
    // reports at least one concealed GOB. Feed the I and corrupted-P
    // separately so we can read the concealed-GOB count for the P picture
    // itself (the I decodes when the P's PSC arrives, the P when the tail's
    // PSC arrives).
    let mut conceal =
        H261Decoder::new(CodecId::new(oxideav_h261::CODEC_ID_STR)).with_error_concealment(true);
    // Feed I (buffered), then corrupted-P (decodes the I, buffers P), then the
    // tail-I (decodes the corrupted P — concealing its damaged GOB(s)).
    let feed = |dec: &mut H261Decoder, b: &[u8]| {
        let pkt = Packet::new(0, TimeBase::new(1, 30), b.to_vec());
        dec.send_packet(&pkt)
            .expect("send_packet must not error in conceal mode");
    };
    feed(&mut conceal, &s_i);
    feed(&mut conceal, &corrupted_p);
    // I is now decodable.
    let f_i = conceal.receive_frame().expect("I");
    assert!(matches!(f_i, Frame::Video(_)));
    assert_eq!(conceal.last_concealed_gobs(), 0, "clean I conceals nothing");
    // Feed the tail I ⇒ the corrupted P is decoded, concealing ≥ 1 GOB.
    feed(&mut conceal, &s_tail);
    let f_p = conceal.receive_frame().expect("concealed P");
    let concealed_for_p = conceal.last_concealed_gobs();
    assert!(
        concealed_for_p >= 1,
        "the corrupted P concealed at least one GOB (got {concealed_for_p})"
    );
    match f_p {
        Frame::Video(vf) => {
            assert_eq!(vf.planes[0].data.len(), W * H, "full QCIF luma plane");
            assert_eq!(vf.planes.len(), 3);
        }
        _ => panic!("expected a video frame"),
    }
    // Drain the tail-I; it decodes cleanly (0 concealed).
    conceal.flush().ok();
    let f_tail = conceal.receive_frame().expect("tail I");
    assert!(matches!(f_tail, Frame::Video(_)));
    assert_eq!(
        conceal.last_concealed_gobs(),
        0,
        "clean tail I conceals nothing"
    );
}

#[test]
fn clean_stream_conceals_nothing() {
    // A well-formed stream decodes identically with concealment on, and the
    // concealed-GOB count stays zero.
    let (s_i, s_p) = make_i_then_p();
    let mut stream = Vec::new();
    stream.extend_from_slice(&s_i);
    stream.extend_from_slice(&s_p);

    let mut strict = H261Decoder::new(CodecId::new(oxideav_h261::CODEC_ID_STR));
    let strict_frames = decode_all(&mut strict, &stream);

    let mut conceal =
        H261Decoder::new(CodecId::new(oxideav_h261::CODEC_ID_STR)).with_error_concealment(true);
    let conceal_frames = decode_all(&mut conceal, &stream);
    assert_eq!(
        conceal.last_concealed_gobs(),
        0,
        "clean stream, no concealment"
    );

    // Same number of frames, byte-identical planes.
    assert_eq!(strict_frames.len(), conceal_frames.len());
    for (a, b) in strict_frames.iter().zip(conceal_frames.iter()) {
        match (a, b) {
            (Frame::Video(va), Frame::Video(vb)) => {
                assert_eq!(va.planes[0].data, vb.planes[0].data, "Y identical");
                assert_eq!(va.planes[1].data, vb.planes[1].data, "Cb identical");
                assert_eq!(va.planes[2].data, vb.planes[2].data, "Cr identical");
            }
            _ => panic!("expected video frames"),
        }
    }
}
