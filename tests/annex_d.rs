//! Integration test for H.261 Annex D (still image transmission).
//!
//! Verifies that:
//!
//! * `write_picture_header_full` with `hi_res_off = false` and a TR
//!   built by `still_image_tr` round-trips through `parse_picture_header`
//!   and `PictureHeader::still_image_sub_index` to the original sub-image
//!   index, for every quadrant and both source formats.
//! * `subsample_still_image` partitions the still-image samples bit-exact
//!   into the four sub-images so that `reassemble_still_image` recovers
//!   the original triple unchanged (luma + both chroma planes).
//! * The pre-spec'd still-image dimensions match §D.2: QCIF video format
//!   ⇒ CIF still image (352 × 288), CIF video format ⇒ 704 × 576 still.

use oxideav_core::bits::{BitReader, BitWriter};
use oxideav_h261::annex_d::{
    parse_still_image_tr, reassemble_still_image, still_image_chroma_dimensions,
    still_image_dimensions, still_image_tr, subsample_still_image, SubImageIndex,
};
use oxideav_h261::encoder::write_picture_header_full;
use oxideav_h261::picture::{parse_picture_header, SourceFormat};

#[test]
fn picture_header_round_trip_annex_d_qcif() {
    for n in 0u8..4 {
        let idx = SubImageIndex::from_u8(n);
        let mut bw = BitWriter::new();
        let tr = still_image_tr(idx);
        write_picture_header_full(&mut bw, SourceFormat::Qcif, tr, /* hi_res_off */ false);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let hdr = parse_picture_header(&mut br).expect("annex-d picture header parses");
        assert_eq!(hdr.source_format, SourceFormat::Qcif);
        assert!(!hdr.hi_res_off, "HI_RES bit should be cleared");
        assert_eq!(hdr.temporal_reference, tr);
        let recovered = hdr
            .still_image_sub_index()
            .expect("annex-d helper succeeds")
            .expect("hi_res off ⇒ sub-image present");
        assert_eq!(recovered, idx);
        assert_eq!(parse_still_image_tr(tr), Ok(idx));
    }
}

#[test]
fn picture_header_round_trip_annex_d_cif() {
    for n in 0u8..4 {
        let idx = SubImageIndex::from_u8(n);
        let mut bw = BitWriter::new();
        let tr = still_image_tr(idx);
        write_picture_header_full(&mut bw, SourceFormat::Cif, tr, /* hi_res_off */ false);
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let hdr = parse_picture_header(&mut br).expect("annex-d picture header parses");
        assert_eq!(hdr.source_format, SourceFormat::Cif);
        assert!(!hdr.hi_res_off);
        let recovered = hdr
            .still_image_sub_index()
            .expect("annex-d helper succeeds")
            .expect("hi_res off ⇒ sub-image present");
        assert_eq!(recovered, idx);
    }
}

#[test]
fn picture_header_motion_video_has_no_sub_index() {
    let mut bw = BitWriter::new();
    write_picture_header_full(&mut bw, SourceFormat::Qcif, 7, /* hi_res_off */ true);
    let bytes = bw.into_bytes();
    let mut br = BitReader::new(&bytes);
    let hdr = parse_picture_header(&mut br).unwrap();
    assert!(hdr.hi_res_off);
    assert_eq!(hdr.still_image_sub_index(), Ok(None));
}

#[test]
fn full_still_image_subsample_reassemble_round_trip_qcif() {
    // §D.2: QCIF video format ⇒ still image is CIF.
    let fmt = SourceFormat::Qcif;
    let (sw, sh) = still_image_dimensions(fmt);
    let (cw, ch) = still_image_chroma_dimensions(fmt);
    assert_eq!((sw, sh), (352, 288));
    assert_eq!((cw, ch), (176, 144));

    // Deterministic synthetic still image — each byte unique enough that
    // a swap in the sub-sampling would be detectable.
    let mut y = vec![0u8; (sw * sh) as usize];
    let mut cb = vec![0u8; (cw * ch) as usize];
    let mut cr = vec![0u8; (cw * ch) as usize];
    let mut state: u32 = 0x1357_9BDF;
    for b in y.iter_mut().chain(cb.iter_mut()).chain(cr.iter_mut()) {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        *b = ((state >> 8) & 0xFF) as u8;
    }

    let subs = subsample_still_image(fmt, &y, &cb, &cr);
    // Each sub-image is the video format (QCIF here).
    let (vw, vh) = fmt.dimensions();
    let (cvw, cvh) = (vw / 2, vh / 2);
    for s in &subs {
        assert_eq!(s.y.len(), (vw * vh) as usize);
        assert_eq!(s.cb.len(), (cvw * cvh) as usize);
        assert_eq!(s.cr.len(), (cvw * cvh) as usize);
    }

    let (y2, cb2, cr2) = reassemble_still_image(fmt, &subs);
    assert_eq!(y2, y, "luma round-trip");
    assert_eq!(cb2, cb, "cb round-trip");
    assert_eq!(cr2, cr, "cr round-trip");
}

#[test]
fn full_still_image_subsample_reassemble_round_trip_cif() {
    // §D.2: CIF video format ⇒ still image is 704 × 576.
    let fmt = SourceFormat::Cif;
    let (sw, sh) = still_image_dimensions(fmt);
    let (cw, ch) = still_image_chroma_dimensions(fmt);
    assert_eq!((sw, sh), (704, 576));
    assert_eq!((cw, ch), (352, 288));

    let mut y = vec![0u8; (sw * sh) as usize];
    let mut cb = vec![0u8; (cw * ch) as usize];
    let mut cr = vec![0u8; (cw * ch) as usize];
    let mut state: u32 = 0xABCDEF01;
    for b in y.iter_mut().chain(cb.iter_mut()).chain(cr.iter_mut()) {
        state = state.wrapping_mul(48271);
        *b = ((state >> 16) & 0xFF) as u8;
    }

    let subs = subsample_still_image(fmt, &y, &cb, &cr);
    let (y2, cb2, cr2) = reassemble_still_image(fmt, &subs);
    assert_eq!(y2, y);
    assert_eq!(cb2, cb);
    assert_eq!(cr2, cr);
}

/// §D.3 transmission order: sub-images are emitted as 0 → 1 → 2 → 3
/// in sequence. We don't enforce wire-level ordering inside the crate
/// (an encoder driving sub-images calls `H261Encoder::encode_frame`
/// per sub-image), but the `SubImageIndex::next` iterator walks the
/// sequence and stops after the last.
#[test]
fn transmission_order_walk() {
    let mut idx = Some(SubImageIndex::Zero);
    let mut seen = Vec::new();
    while let Some(i) = idx {
        seen.push(i.as_u8());
        idx = i.next();
    }
    assert_eq!(seen, vec![0, 1, 2, 3]);
}
