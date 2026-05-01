//! DoS-protection coverage for the H.261 decoder.
//!
//! Two fixtures, each demonstrating one layer of the
//! [`oxideav_core::DecoderLimits`] framework:
//!
//! 1. `picture_header_too_large_returns_invalid_data` —
//!    The header-parse layer rejects a picture whose declared
//!    dimensions exceed `limits.max_pixels_per_frame`. H.261 only
//!    allows two picture formats (QCIF 176x144 and CIF 352x288), neither
//!    of which is *caller-declared* — so the right error is
//!    [`Error::InvalidData`] (the codec's intrinsic frame size doesn't
//!    fit in the caller's caps), NOT
//!    [`Error::ResourceExhausted`] (which is reserved for
//!    pool-exhaustion events at the arena layer).
//!
//! 2. `pool_exhaustion_returns_resource_exhausted` —
//!    Lease N+1 arenas from the decoder's pool (pool sized at
//!    `limits.max_arenas_in_flight = 2` here); the third lease must
//!    surface `Error::ResourceExhausted`. This proves the pool
//!    backpressure path is wired.

use oxideav_core::packet::PacketFlags;
use oxideav_core::{CodecId, Decoder, DecoderLimits, Error, Packet, TimeBase};

use oxideav_h261::decoder::H261Decoder;

/// Build a minimal QCIF I-picture header with an empty body. The body
/// is intentionally empty — for the header-parse fuzz fixture the
/// decoder must reject the picture *before* attempting to decode it.
fn minimal_qcif_picture_header() -> Vec<u8> {
    // Same construction as `picture::tests::minimal_qcif_header` —
    // PSC + TR + PTYPE (QCIF) + PEI=0, padded to a byte boundary.
    let mut bits: Vec<u8> = Vec::new();
    let append = |v: &mut Vec<u8>, val: u32, n: u32| {
        for i in (0..n).rev() {
            v.push(((val >> i) & 1) as u8);
        }
    };
    append(&mut bits, 0x00010, 20); // PSC
    append(&mut bits, 1, 5); // TR = 1
    append(&mut bits, 1, 1); // split
    append(&mut bits, 0, 1); // doc-cam
    append(&mut bits, 0, 1); // freeze
    append(&mut bits, 0, 1); // fmt = QCIF
    append(&mut bits, 1, 1); // HI_RES off
    append(&mut bits, 0, 1); // spare
    append(&mut bits, 0, 1); // PEI = 0
    while bits.len() % 8 != 0 {
        bits.push(0);
    }
    let mut out = Vec::new();
    for chunk in bits.chunks(8) {
        let mut b = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            b |= bit << (7 - i);
        }
        out.push(b);
    }
    out
}

#[test]
fn picture_header_too_large_returns_invalid_data() {
    // Cap pixels at 99x99 = 9801 < QCIF (176*144 = 25344).
    let limits = DecoderLimits::default().with_max_pixels_per_frame(99 * 99);
    let codec_id = CodecId::new(oxideav_h261::CODEC_ID_STR);
    let mut dec = H261Decoder::with_limits(codec_id, limits);

    let pkt = Packet {
        stream_index: 0,
        data: minimal_qcif_picture_header(),
        pts: Some(0),
        dts: Some(0),
        duration: None,
        time_base: TimeBase::new(1, 30_000),
        flags: PacketFlags {
            keyframe: true,
            ..Default::default()
        },
    };
    // The decoder won't fail until `process` runs and tries to consume
    // the picture — which happens on `flush` (single-PSC packet, EOF
    // boundary triggers the last-picture decode).
    let send_res = dec.send_packet(&pkt);
    let flush_res = dec.flush();
    let any_err = match (send_res, flush_res) {
        (Err(e), _) | (_, Err(e)) => Some(e),
        (Ok(()), Ok(())) => None,
    };
    let err =
        any_err.expect("expected decoder to reject the QCIF picture against a 99x99 pixel cap");
    assert!(
        matches!(err, Error::InvalidData(_)),
        "expected InvalidData (caller's cap is tighter than H.261's intrinsic format), got {err:?}"
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("max_pixels_per_frame"),
        "error message should mention the cap that fired: {msg}"
    );
}

#[test]
fn picture_header_within_cap_decodes_normally() {
    // Sanity counterpoint: a generous cap admits the same QCIF header.
    let limits = DecoderLimits::default().with_max_pixels_per_frame(1024 * 1024);
    let codec_id = CodecId::new(oxideav_h261::CODEC_ID_STR);
    let mut dec = H261Decoder::with_limits(codec_id, limits);

    // The header alone won't decode (no GOB body, no MBs) — but it must
    // get *past* the dimension check before failing on body parse. We
    // assert the failure (if any) is NOT the dimension-cap diagnostic.
    let pkt = Packet {
        stream_index: 0,
        data: minimal_qcif_picture_header(),
        pts: Some(0),
        dts: Some(0),
        duration: None,
        time_base: TimeBase::new(1, 30_000),
        flags: PacketFlags {
            keyframe: true,
            ..Default::default()
        },
    };
    let _ = dec.send_packet(&pkt);
    let r = dec.flush();
    if let Err(e) = r {
        let msg = format!("{e}");
        assert!(
            !msg.contains("max_pixels_per_frame"),
            "should not have tripped the dimension cap, but did: {msg}"
        );
    }
}

#[test]
fn pool_exhaustion_returns_resource_exhausted() {
    // Pool size 2 — two arenas may be checked out concurrently, the
    // third lease must error with ResourceExhausted.
    let limits = DecoderLimits::default().with_max_arenas_in_flight(2);
    let codec_id = CodecId::new(oxideav_h261::CODEC_ID_STR);
    let dec = H261Decoder::with_limits(codec_id, limits);
    let pool = dec.arena_pool().clone();
    assert_eq!(pool.max_arenas(), 2);
    let a = pool.lease().expect("first lease");
    let b = pool.lease().expect("second lease");
    let third = pool.lease();
    let third_err_msg = match &third {
        Err(e) => format!("Err({e})"),
        Ok(_) => "Ok(<arena>)".to_string(),
    };
    assert!(
        matches!(third, Err(Error::ResourceExhausted(_))),
        "third lease should ResourceExhausted with pool of 2; got {third_err_msg}"
    );
    drop((a, b));
    // After dropping both, the pool refills.
    let _again = pool.lease().expect("re-lease after drop");
}

#[test]
fn default_limits_admit_qcif_and_cif() {
    // Sanity: defaults are 32k x 32k pixels, well above CIF.
    let limits = DecoderLimits::default();
    assert!(limits.max_pixels_per_frame >= 352 * 288);
    let codec_id = CodecId::new(oxideav_h261::CODEC_ID_STR);
    let dec = H261Decoder::with_limits(codec_id, limits);
    assert_eq!(dec.limits().max_arenas_in_flight, 8);
    assert_eq!(dec.arena_pool().max_arenas(), 8);
}

#[test]
fn pool_buffer_returns_after_decode() {
    // Decode succeeds → arena is released → another lease succeeds.
    // We don't have a real H.261 fixture here (those live in
    // reference_clip.rs and depend on /tmp/h261/*); instead we
    // just ensure the pool behaves correctly across multiple
    // `with_limits` instantiations and arena-leasing cycles.
    let limits = DecoderLimits::default().with_max_arenas_in_flight(1);
    let codec_id = CodecId::new(oxideav_h261::CODEC_ID_STR);
    let dec = H261Decoder::with_limits(codec_id, limits);
    let pool = dec.arena_pool().clone();
    assert_eq!(pool.max_arenas(), 1);
    {
        let _a = pool.lease().expect("first lease");
        // Pool is exhausted while `_a` is alive.
        assert!(matches!(pool.lease(), Err(Error::ResourceExhausted(_))));
    }
    // `_a` dropped — pool refills.
    let _b = pool.lease().expect("re-lease after drop");
}
