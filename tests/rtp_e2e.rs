//! End-to-end: encode H.261 I-pictures, packetize per RFC 4587 §4.1, and
//! verify that depacketization recovers the original elementary stream.
//!
//! The GOB-aligned cheap packetizer (RFC 4587 §4.2) splits at start
//! codes and produces SBIT=EBIT=0 payloads, so the recovered stream
//! must match byte-for-byte after a round trip — and the H.261 decoder
//! must still accept it.

use oxideav_h261::decoder::H261Decoder;
use oxideav_h261::encoder::encode_intra_picture;
use oxideav_h261::picture::SourceFormat;
use oxideav_h261::rtp::{depacketize, packetize_gob_aligned, unpack_header, HEADER_LEN};

use oxideav_core::registry::codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};

fn gradient_qcif() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = 176usize;
    let h = 144usize;
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let v = 32 + (i * 150) / w + (j * 50) / h;
            y[j * w + i] = v.clamp(0, 255) as u8;
        }
    }
    let cb = vec![128u8; (w / 2) * (h / 2)];
    let cr = vec![128u8; (w / 2) * (h / 2)];
    (y, cb, cr)
}

fn encode_qcif_intra(y: &[u8], cb: &[u8], cr: &[u8], quant: u32, tr: u8) -> Vec<u8> {
    encode_intra_picture(SourceFormat::Qcif, y, 176, cb, 88, cr, 88, quant, tr)
        .expect("encode_intra_picture")
}

#[test]
fn rtp_round_trip_recovers_encoded_qcif_intra_picture() {
    let (y, cb, cr) = gradient_qcif();
    let bits = encode_qcif_intra(&y, &cb, &cr, 8, 0);
    assert!(!bits.is_empty(), "encoder should emit at least one byte");

    // Generous max_payload (well above one GOB at QCIF + canonical quants).
    let pkts = packetize_gob_aligned(&bits, 1500, true, false);
    assert!(
        !pkts.is_empty(),
        "encoder output should contain at least one start code"
    );

    // Every payload must be ≤ max_payload, ≥ HEADER_LEN + 1 bytes,
    // and unpack to a syntactically valid header.
    for p in &pkts {
        assert!(p.bytes.len() <= 1500);
        assert!(p.bytes.len() > HEADER_LEN);
        let (h, _data) = unpack_header(&p.bytes).expect("unpack");
        // GOB-aligned packetizer always sets these to 0.
        assert_eq!(h.sbit, 0);
        assert_eq!(h.ebit, 0);
        assert_eq!(h.gobn, 0);
        assert_eq!(h.mbap, 0);
        assert_eq!(h.quant, 0);
        assert_eq!(h.hmvd, 0);
        assert_eq!(h.vmvd, 0);
        // Intra-only hint should match what we asked for.
        assert!(h.intra_only);
    }

    // Exactly one packet should carry the marker bit (last of frame).
    let marker_count = pkts.iter().filter(|p| p.marker).count();
    assert_eq!(marker_count, 1, "exactly one packet should be marker=true");

    let recovered = depacketize(&pkts).expect("depacketize");
    assert_eq!(
        recovered, bits,
        "GOB-aligned RTP round trip should recover the elementary stream byte-for-byte"
    );
}

#[test]
fn rtp_round_trip_then_decode_produces_a_frame() {
    // Verifies that after a packetize/depacketize cycle the recovered
    // stream still decodes through the regular decoder API.
    let (y, cb, cr) = gradient_qcif();
    let bits = encode_qcif_intra(&y, &cb, &cr, 12, 0);
    let pkts = packetize_gob_aligned(&bits, 800, true, false);
    let recovered = depacketize(&pkts).expect("depacketize");

    let mut dec = H261Decoder::new(CodecId::new(oxideav_h261::CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 30), recovered);
    dec.send_packet(&pkt).expect("send_packet");
    // The H.261 decoder buffers until it sees the next picture's PSC
    // (so it can know the current picture has ended). Flush to draw out
    // the trailing picture in a single-shot decode.
    dec.flush().expect("flush");
    match dec.receive_frame() {
        Ok(Frame::Video(vf)) => {
            // QCIF luma plane should be 176 wide × 144 tall — i.e. 176 stride.
            assert_eq!(vf.planes.len(), 3, "expected Y/Cb/Cr planes");
            assert_eq!(vf.planes[0].stride, 176);
            assert_eq!(vf.planes[0].data.len(), 176 * 144);
        }
        Ok(other) => panic!("expected video frame, got {other:?}"),
        Err(e) => panic!("decode failed after RTP round trip: {e}"),
    }
}

#[test]
fn rtp_round_trip_with_small_mtu_round_trip_still_recovers() {
    // Force fragmentation: small MTU well below a single QCIF GOB.
    let (y, cb, cr) = gradient_qcif();
    let bits = encode_qcif_intra(&y, &cb, &cr, 8, 0);
    let pkts = packetize_gob_aligned(&bits, 64, true, false);
    assert!(pkts.len() > 3, "expected several fragments at MTU=64");

    let recovered = depacketize(&pkts).expect("depacketize");
    assert_eq!(
        recovered, bits,
        "fragmented RTP round trip should still be byte-exact"
    );
}

#[test]
fn rtp_payload_header_packs_to_4_bytes_uniformly() {
    // Drift check: every payload in a typical encode must have a 4-byte
    // header followed by ≥ 1 byte of H.261 stream.
    let (y, cb, cr) = gradient_qcif();
    let bits = encode_qcif_intra(&y, &cb, &cr, 16, 3);
    let pkts = packetize_gob_aligned(&bits, 512, true, false);
    for p in &pkts {
        assert!(p.bytes.len() > HEADER_LEN);
        assert_eq!(HEADER_LEN, 4);
    }
}
