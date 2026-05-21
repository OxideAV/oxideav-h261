//! End-to-end: encode several H.261 I-pictures and verify they satisfy
//! the §5.2 per-picture cap and the Annex B HRD buffer model at the
//! canonical 64 kbit/s QCIF channel rate.
//!
//! This validates that the encoder's per-GOB MQUANT rate controller is
//! actually delivering streams that a conforming H.261 receiver could
//! buffer without underflow — the on-paper compliance check, applied to
//! real bytes coming out of `encode_intra_picture`.

use oxideav_h261::encoder::encode_intra_picture;
use oxideav_h261::hrd::{
    check_overflow, check_picture_cap, walk_buffer, HrdParams, PictureCapStatus,
};
use oxideav_h261::picture::SourceFormat;

/// Build a QCIF gradient frame.
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
fn intra_picture_under_quant_8_fits_qcif_64kbit_cap() {
    let (y, cb, cr) = gradient_qcif();
    let bytes = encode_qcif_intra(&y, &cb, &cr, 8, 0);
    let bits = (bytes.len() * 8) as u32;
    assert_eq!(
        check_picture_cap(bits, SourceFormat::Qcif),
        PictureCapStatus::Ok,
        "intra picture at quant=8 produced {bits} bits, exceeding the §5.2 QCIF cap of 65536"
    );
}

#[test]
fn intra_picture_at_low_quant_still_within_qcif_cap() {
    // Lower quant ⇒ more bits. quant=2 is near the bottom of the
    // legal range (§4.2.3.3 forbids 0 for INTRA-CBP modes). Even on a
    // smooth gradient it should still fit a single QCIF picture in 8 KiB.
    let (y, cb, cr) = gradient_qcif();
    let bytes = encode_qcif_intra(&y, &cb, &cr, 2, 0);
    let bits = (bytes.len() * 8) as u32;
    assert_eq!(
        check_picture_cap(bits, SourceFormat::Qcif),
        PictureCapStatus::Ok,
        "intra picture at quant=2 produced {bits} bits, exceeding the §5.2 QCIF cap of 65536"
    );
}

#[test]
fn hrd_walk_holds_for_10_intra_pictures_at_64kbit() {
    // 10 I-pictures encoded at quant=12 (moderately compressed). At
    // 64 kbit/s with 29.97 fps the channel delivers ≈ 2135 bits per
    // frame interval; smooth-content I-pictures should average to
    // similar bits, so the HRD walk should hold without underflow.
    let (y, cb, cr) = gradient_qcif();
    let mut sizes = Vec::with_capacity(10);
    for tr in 0..10u8 {
        let bytes = encode_qcif_intra(&y, &cb, &cr, 12, tr);
        let bits = (bytes.len() * 8) as u32;
        assert_eq!(
            check_picture_cap(bits, SourceFormat::Qcif),
            PictureCapStatus::Ok,
        );
        sizes.push(bits);
    }

    let params = HrdParams::new(64_000);
    // I-pictures are big; with N=1 the buffer will underflow because
    // each picture is far larger than the 2135 bits arriving per
    // interval. Try a more realistic skip (N=10 ⇒ 10× more arrival).
    // In real H.261 the encoder mixes I + P + skipped pictures; here
    // we're checking the HRD reports the underflow correctly.
    let trace_n1 = walk_buffer(&sizes, 1, params);
    assert!(
        trace_n1.first_underflow.is_some(),
        "HRD should flag underflow when 10 I-pictures are sent at full 29.97 fps over 64 kbit/s — got trace {trace_n1:?}"
    );

    // At N=10 (one picture every 10 frame periods ≈ 3 fps) the budget
    // multiplies by 10 ⇒ ≈21354 bits per picture; large enough that
    // the smooth gradient's I-pictures should slot in.
    let trace_n10 = walk_buffer(&sizes, 10, params);
    assert_eq!(
        trace_n10.first_underflow, None,
        "HRD should be compliant at N=10 (≈3 fps over 64 kbit/s): {trace_n10:?}"
    );
}

#[test]
fn hrd_overflow_check_holds_for_normal_drain() {
    // A matched-rate sequence doesn't overflow the receiver's
    // B + 256 kbits buffer. Encode 30 I-pictures at quant=12, then
    // pick a skip factor N that matches the encoder's actual bit rate
    // so per-interval arrival ≈ per-picture size and the buffer drains.
    // TR is a 5-bit field (§4.2.1) ⇒ 32 distinct values, so 30 pictures
    // is comfortably below the limit.
    let (y, cb, cr) = gradient_qcif();
    let sizes: Vec<u32> = (0..30u8)
        .map(|tr| (encode_qcif_intra(&y, &cb, &cr, 12, tr).len() * 8) as u32)
        .collect();
    // At 64 kbit/s with N=4 (≈ 7.5 fps), per-interval arrival is
    // ≈ 8541 bits — close to a typical I-picture size for the gradient
    // at quant=12. Receiver buffer cap is B + 256 kbits ≈ 270685, so
    // even with some over-allocation early on we shouldn't trip overflow.
    let params = HrdParams::new(64_000);
    assert_eq!(
        check_overflow(&sizes, 4, params),
        None,
        "64 kbit/s channel at N=4 (7.5 fps) should drain 30 I-pictures without overflow"
    );
}
