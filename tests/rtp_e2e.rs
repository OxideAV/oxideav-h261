//! End-to-end: encode H.261 I-pictures, packetize per RFC 4587 §4.1, and
//! verify that depacketization recovers the original elementary stream.
//!
//! The GOB-aligned cheap packetizer (RFC 4587 §4.2) splits at start
//! codes and produces SBIT=EBIT=0 payloads, so the recovered stream
//! must match byte-for-byte after a round trip — and the H.261 decoder
//! must still accept it.

use oxideav_h261::decoder::H261Decoder;
use oxideav_h261::encoder::{encode_intra_picture, H261Encoder};
use oxideav_h261::picture::SourceFormat;
use oxideav_h261::rtp::{
    depacketize, packetize_gob_aligned, packetize_mb_fragmented, parse_rtp_fixed_header,
    unpack_header, H261RtpPayload, RtpPacketizer, HEADER_LEN, RTP_FIXED_HEADER_LEN,
};

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

// --------------------------------------------------------------------
// MB-level fragmentation (RFC 4587 §4.2 RECOMMENDED packetization)
// --------------------------------------------------------------------

#[test]
fn mb_fragmented_round_trip_decodes_end_to_end() {
    // The §4.2 RECOMMENDED packetizer splits oversize GOBs on macroblock
    // boundaries; continuation packets carry the §4.1 GOBN / MBAP /
    // QUANT / HMVD / VMVD context and non-zero SBIT/EBIT. After the
    // round trip the recovered stream must be byte-exact and still
    // decode through the regular decoder API.
    let (y, cb, cr) = gradient_qcif();
    let bits = encode_qcif_intra(&y, &cb, &cr, 8, 0);
    let pkts = packetize_mb_fragmented(&bits, 64, true, false).expect("mb fragment");
    assert!(pkts.len() > 3, "expected several MB-boundary fragments");

    let mut saw_continuation = false;
    for p in &pkts {
        assert!(p.bytes.len() <= 64, "payload exceeds budget");
        let (h, _data) = unpack_header(&p.bytes).expect("unpack");
        assert!(h.intra_only);
        if h.gobn != 0 {
            saw_continuation = true;
            // Mid-GOB context invariants per RFC 4587 §4.1.
            assert!((1..=12).contains(&h.gobn));
            assert!((1..=31).contains(&h.quant));
            assert!(h.mbap <= 31);
        }
    }
    assert!(
        saw_continuation,
        "a 64-byte budget must force mid-GOB continuation packets"
    );
    let marker_count = pkts.iter().filter(|p| p.marker).count();
    assert_eq!(marker_count, 1, "exactly one marker per picture");

    let recovered = depacketize(&pkts).expect("depacketize");
    assert_eq!(recovered, bits, "MB-fragmented round trip is byte-exact");

    let mut dec = H261Decoder::new(CodecId::new(oxideav_h261::CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 30), recovered);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    match dec.receive_frame() {
        Ok(Frame::Video(vf)) => {
            assert_eq!(vf.planes[0].stride, 176);
            assert_eq!(vf.planes[0].data.len(), 176 * 144);
        }
        Ok(other) => panic!("expected video frame, got {other:?}"),
        Err(e) => panic!("decode failed after MB-fragmented round trip: {e}"),
    }
}

#[test]
fn rtp_packetizer_mb_fragmentation_end_to_end() {
    // Full RTP wrap with MB fragmentation enabled: every packet fits the
    // MTU, the receiver path (fixed-header parse → H.261 header unpack →
    // depacketize) recovers the elementary stream byte-exactly, and it
    // still decodes.
    let (y, cb, cr) = gradient_qcif();
    let bits = encode_qcif_intra(&y, &cb, &cr, 8, 0);
    let mut pk = RtpPacketizer::new(96, 0x0BAD_F00D, 7, 96)
        .with_intra_only(true)
        .with_mb_fragmentation(true);
    let packets = pk.pack_frame(&bits, 0);
    assert!(!packets.is_empty());

    let mut inner = Vec::new();
    let mut saw_continuation = false;
    for p in &packets {
        assert!(p.bytes.len() <= 96, "RTP packet exceeds MTU");
        let (rtp_hdr, rest) = parse_rtp_fixed_header(&p.bytes).expect("rtp parse");
        assert_eq!(rtp_hdr.payload_type, 96);
        let (h261_hdr, _payload) = unpack_header(rest).expect("h261 unpack");
        if h261_hdr.gobn != 0 {
            saw_continuation = true;
        }
        inner.push(H261RtpPayload {
            header: h261_hdr,
            bytes: rest.to_vec(),
            marker: p.marker,
        });
    }
    assert!(saw_continuation, "MTU 96 must force mid-GOB packets");
    assert!(packets.last().unwrap().marker);
    assert_eq!(packets.iter().filter(|p| p.marker).count(), 1);

    let recovered = depacketize(&inner).expect("depacketize");
    assert_eq!(recovered, bits);

    let mut dec = H261Decoder::new(CodecId::new(oxideav_h261::CODEC_ID_STR));
    let pkt = Packet::new(0, TimeBase::new(1, 30), recovered);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    assert!(
        matches!(dec.receive_frame(), Ok(Frame::Video(_))),
        "MB-fragmented RTP session output must decode"
    );
}

// --------------------------------------------------------------------
// RtpPacketizer (encoder-side full RFC 3550 packets)
// --------------------------------------------------------------------

#[test]
fn rtp_packetizer_emits_full_rtp_packets_for_a_real_intra_frame() {
    let (y, cb, cr) = gradient_qcif();
    let bits = encode_qcif_intra(&y, &cb, &cr, 8, 0);
    let mut pk = RtpPacketizer::new(96, 0xCAFEBABE, 100, 1400).with_intra_only(true);
    let packets = pk.pack_frame(&bits, 90_000);
    assert!(
        !packets.is_empty(),
        "RtpPacketizer should emit at least one packet for a real frame"
    );

    // Every packet carries the same SSRC + timestamp, and consecutive
    // sequence numbers.
    let mut expected_seq = 100u16;
    for p in &packets {
        assert_eq!(p.ssrc, 0xCAFEBABE);
        assert_eq!(p.timestamp, 90_000);
        assert_eq!(p.sequence_number, expected_seq);
        expected_seq = expected_seq.wrapping_add(1);
        // Wire bytes match struct.
        assert_eq!(
            u32::from_be_bytes(p.bytes[8..12].try_into().unwrap()),
            p.ssrc
        );
        assert_eq!(
            u32::from_be_bytes(p.bytes[4..8].try_into().unwrap()),
            p.timestamp
        );
        assert_eq!(
            u16::from_be_bytes(p.bytes[2..4].try_into().unwrap()),
            p.sequence_number
        );
        // V=2, P=0, X=0, CC=0 ⇒ byte 0 == 0x80.
        assert_eq!(p.bytes[0], 0x80);
        assert!(p.bytes.len() <= 1400);
        assert!(p.bytes.len() > RTP_FIXED_HEADER_LEN + HEADER_LEN);
    }
    // Exactly one packet — the last — carries M=1.
    let markers: Vec<bool> = packets.iter().map(|p| p.marker).collect();
    assert!(markers.last().copied().unwrap());
    assert_eq!(markers.iter().filter(|m| **m).count(), 1);
}

#[test]
fn rtp_packetizer_drives_h261_encoder_end_to_end() {
    // Encode two frames (I then P) through the public H261Encoder API
    // and packetise each one independently. Verify timestamps are
    // independent, sequence numbers are dense, every frame's last
    // packet carries the marker, and reassembled bytes decode through
    // the H261Decoder.
    let (y, cb, cr) = gradient_qcif();
    let mut enc = H261Encoder::new(SourceFormat::Qcif, 12);
    let mut pk = RtpPacketizer::new(96, 0xFEEDFACE, 9000, 1024).with_intra_only(false);

    let bits_a = enc
        .encode_frame(&y, 176, &cb, 88, &cr, 88)
        .expect("encode frame A");
    let bits_b = enc
        .encode_frame(&y, 176, &cb, 88, &cr, 88)
        .expect("encode frame B");

    let packets_a = pk.pack_frame(&bits_a, 0);
    let packets_b = pk.pack_frame(&bits_b, 3000); // +3000 ticks at 90 kHz = 1/30s
    assert!(!packets_a.is_empty());
    assert!(!packets_b.is_empty());

    // All packets in A share timestamp 0, all in B share timestamp 3000.
    for p in &packets_a {
        assert_eq!(p.timestamp, 0);
    }
    for p in &packets_b {
        assert_eq!(p.timestamp, 3000);
    }
    // Sequence numbers run dense across the frame boundary.
    assert_eq!(
        packets_b[0].sequence_number,
        packets_a.last().unwrap().sequence_number.wrapping_add(1)
    );
    // Marker bit set exactly on the last packet of each frame.
    assert!(packets_a.last().unwrap().marker);
    assert!(packets_b.last().unwrap().marker);
    assert_eq!(packets_a.iter().filter(|p| p.marker).count(), 1);
    assert_eq!(packets_b.iter().filter(|p| p.marker).count(), 1);

    // Receiver path: parse the RTP fixed headers, hand the inner H.261
    // payloads to depacketize, decode the result.
    let mut dec = H261Decoder::new(CodecId::new(oxideav_h261::CODEC_ID_STR));
    for (frame_pkts, frame_bits) in [(&packets_a, &bits_a), (&packets_b, &bits_b)] {
        let mut inner = Vec::new();
        for rp in frame_pkts {
            let (rtp_hdr, rest) = parse_rtp_fixed_header(&rp.bytes).expect("rtp parse");
            assert_eq!(rtp_hdr.payload_type, 96);
            assert_eq!(rtp_hdr.ssrc, 0xFEEDFACE);
            let (h261_hdr, _payload) = unpack_header(rest).expect("h261 unpack");
            inner.push(H261RtpPayload {
                header: h261_hdr,
                bytes: rest.to_vec(),
                marker: rp.marker,
            });
        }
        let recovered = depacketize(&inner).expect("depacketize");
        assert_eq!(
            recovered, *frame_bits,
            "RtpPacketizer round trip must be byte-exact"
        );
        let pkt = Packet::new(0, TimeBase::new(1, 30), recovered);
        dec.send_packet(&pkt).expect("send_packet");
    }
    dec.flush().expect("flush");
    let mut frames = 0;
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => {
                assert_eq!(vf.planes[0].stride, 176);
                frames += 1;
            }
            Ok(other) => panic!("expected video frame, got {other:?}"),
            Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("decode failed after RtpPacketizer: {e}"),
        }
    }
    assert!(frames >= 1, "expected at least one decoded frame");
}

#[test]
fn rtp_packetizer_payload_size_fits_under_mtu() {
    // Every emitted packet must be ≤ max_rtp_packet_size, never above.
    let (y, cb, cr) = gradient_qcif();
    let bits = encode_qcif_intra(&y, &cb, &cr, 16, 0);
    for &mtu in &[64usize, 200, 512, 1500] {
        let mut pk = RtpPacketizer::new(96, 1, 0, mtu);
        let packets = pk.pack_frame(&bits, 0);
        for p in &packets {
            assert!(
                p.bytes.len() <= mtu,
                "packet len {} exceeds mtu {}",
                p.bytes.len(),
                mtu
            );
        }
    }
}

#[test]
fn rtcp_sender_report_reflects_packetizer_session_state() {
    use oxideav_h261::rtcp::{parse_report, ReceptionReportBlock, PT_SR};

    // Encode two QCIF I-pictures and packetize each, then build an RTCP SR
    // from the packetizer's running counters (RFC 3550 §6.4.1).
    let (y, cb, cr) = gradient_qcif();
    let f0 = encode_qcif_intra(&y, &cb, &cr, 12, 0);
    let f1 = encode_qcif_intra(&y, &cb, &cr, 12, 1);

    let mut pk = RtpPacketizer::new(96, 0x1357_9BDF, 0, 1500);
    let p0 = pk.pack_frame(&f0, 0);
    let p1 = pk.pack_frame(&f1, 3003);

    let expected_packets = (p0.len() + p1.len()) as u32;
    let expected_octets: u32 = p0
        .iter()
        .chain(p1.iter())
        .map(|p| (p.bytes.len() - RTP_FIXED_HEADER_LEN) as u32)
        .sum();
    assert_eq!(pk.packet_count(), expected_packets);
    assert_eq!(pk.octet_count(), expected_octets);

    // SR with one reception report block describing a peer.
    let block = ReceptionReportBlock {
        ssrc: 0x2468_ACE0,
        fraction_lost: 26, // ~10%
        cumulative_lost: 5,
        extended_highest_seq: 0x0001_0042,
        jitter: 144,
        last_sr: 0xB705_2000,
        delay_since_last_sr: 0x0005_4000,
    };
    let ntp = 0xB44D_B705_2000_0000u64;
    let sr = pk.sender_report(ntp, std::slice::from_ref(&block)).unwrap();

    let parsed = parse_report(&sr).unwrap();
    assert_eq!(parsed.packet_type, PT_SR);
    assert_eq!(parsed.ssrc, 0x1357_9BDF);
    let info = parsed.sender_info.expect("SR carries sender info");
    assert_eq!(info.ntp_timestamp, ntp);
    assert_eq!(info.rtp_timestamp, 3003); // last frame's RTP timestamp
    assert_eq!(info.packet_count, expected_packets);
    assert_eq!(info.octet_count, expected_octets);
    assert_eq!(parsed.report_blocks, vec![block]);
}

#[test]
fn rtcp_receiver_report_round_trips_pure_receiver_path() {
    use oxideav_h261::rtcp::{build_receiver_report, parse_report, ReceptionReportBlock, PT_RR};

    // A pure receiver (no media transmitted) emits an RR. The canonical
    // empty RR (RC = 0) heads a compound packet when nothing is heard;
    // a populated RR carries one block per source.
    let empty = build_receiver_report(0xCAFE, &[]).unwrap();
    let parsed_empty = parse_report(&empty).unwrap();
    assert_eq!(parsed_empty.packet_type, PT_RR);
    assert!(parsed_empty.sender_info.is_none());
    assert!(parsed_empty.report_blocks.is_empty());

    let block = ReceptionReportBlock {
        ssrc: 0x1357_9BDF, // the sender from the previous test
        fraction_lost: 0,
        cumulative_lost: -2, // a duplicate arrived ⇒ negative
        extended_highest_seq: 0x0000_00FF,
        jitter: 7,
        last_sr: 0x1122_3344,
        delay_since_last_sr: 0x0000_8000,
    };
    let rr = build_receiver_report(0xCAFE, std::slice::from_ref(&block)).unwrap();
    let parsed = parse_report(&rr).unwrap();
    assert_eq!(parsed.ssrc, 0xCAFE);
    assert_eq!(parsed.report_blocks, vec![block]);
    assert_eq!(parsed.report_blocks[0].cumulative_lost, -2);
}
