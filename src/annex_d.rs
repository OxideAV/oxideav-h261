//! H.261 Annex D — Still image transmission.
//!
//! Annex D forms an integral part of Recommendation H.261 (03/93). It
//! describes how an H.261 video coder transmits a still image at four
//! times the normal video resolution by temporarily stopping the motion
//! video. The still image is sub-sampled 2:1 horizontally and vertically
//! into four sub-images carried in the currently transmitted video
//! format; the four sub-images are emitted in order 0, 1, 2, 3 and the
//! receiver re-assembles them into the full-resolution still image.
//!
//! ## Spec mapping (§D.2 / §D.3)
//!
//! * **§D.2 — Still image format.** The still image is four times the
//!   currently transmitted video format. If the video format is QCIF
//!   (176 × 144), the still image is a CIF frame (352 × 288). If the
//!   video format is CIF (352 × 288), the still image is 704 × 576
//!   (a CCIR-601 frame). The 2:1 × 2:1 sub-sampling pattern of
//!   Figure D.1 labels each sample with the sub-image index it belongs
//!   to. Repeating the 2 × 2 tile across the still image:
//!
//!   ```text
//!     col0 col1
//!   ┌──────────
//!   │  0   3      row0
//!   │  1   2      row1
//!   ```
//!
//!   so the per-sub-image origin within each 2 × 2 tile is
//!
//!   ```text
//!     idx | (dx, dy)
//!     ----+----------
//!      0  | (0, 0)
//!      1  | (0, 1)
//!      2  | (1, 1)
//!      3  | (1, 0)
//!   ```
//!
//!   Each sub-image therefore has the same dimensions as the currently
//!   transmitted video format.
//!
//! * **§D.3 — Picture layer multiplex.** When `HI_RES = 0`, the two
//!   lower bits of the temporal reference (TR) identify one of the
//!   four sub-images 0, 1, 2 or 3; the three higher bits of TR shall
//!   be set to 0. The encoder transmits the four sub-images in
//!   sequential order. It is allowed to transmit more than one frame
//!   for each sub-image, but it must not go back once it starts
//!   transmitting the next sub-image. The encoder is allowed to resume
//!   motion video at any time by setting `HI_RES` back to 1.
//!
//! * **§D.5 — Other considerations.** All video coding modes are
//!   allowed (intra / inter / motion compensation), the multiplex
//!   below the picture layer is unchanged, the per-frame bit cap of
//!   §5.2 still applies to each sub-image, and the forward error
//!   correction layer (§5.4) is unaffected.
//!
//! ## What this module provides
//!
//! Annex D semantics live above the standard motion-video pipeline:
//! the encoder still produces a sequence of `oxideav_h261` picture
//! payloads (one per sub-image), the decoder still parses them with
//! the normal picture / GOB / MB / block layers, and the only
//! Annex-D-specific machinery is
//!
//! * mapping `SubImageIndex` to / from the 5-bit `TR` field
//!   ([`still_image_tr`] / [`parse_still_image_tr`]);
//! * mapping the currently transmitted video [`SourceFormat`] to the
//!   still-image dimensions and the per-sub-image origin
//!   ([`still_image_dimensions`] / [`subimage_origin`]);
//! * the sub-sampling / re-assembly transform between a full-resolution
//!   still image and its four sub-images ([`subsample_still_image`] /
//!   [`reassemble_still_image`]).
//!
//! Callers wire these into the rest of the codec themselves: feed each
//! sub-image into the regular [`crate::encoder::H261Encoder`], stamp the
//! Annex-D `TR` from [`still_image_tr`], and clear the picture header's
//! HI_RES bit. Reception is the mirror image: parse the picture header,
//! check [`crate::picture::PictureHeader::still_image_sub_index`], and
//! feed the four decoded sub-image frames to [`reassemble_still_image`].

use crate::picture::SourceFormat;

/// Annex D sub-image index — one of the four 2:1 × 2:1 sub-sampled
/// sub-images that together form a full-resolution still image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SubImageIndex {
    /// First sub-image (origin `(0, 0)` within each 2 × 2 tile).
    Zero,
    /// Second sub-image (origin `(0, 1)` within each 2 × 2 tile).
    One,
    /// Third sub-image (origin `(1, 1)` within each 2 × 2 tile).
    Two,
    /// Fourth sub-image (origin `(1, 0)` within each 2 × 2 tile).
    Three,
}

impl SubImageIndex {
    /// Raw 0..=3 index.
    pub fn as_u8(self) -> u8 {
        match self {
            SubImageIndex::Zero => 0,
            SubImageIndex::One => 1,
            SubImageIndex::Two => 2,
            SubImageIndex::Three => 3,
        }
    }

    /// Construct from a 0..=3 index, panicking on values outside that range.
    /// Use [`SubImageIndex::try_from_u8`] for the fallible variant.
    pub fn from_u8(n: u8) -> Self {
        Self::try_from_u8(n).expect("SubImageIndex::from_u8 requires 0..=3")
    }

    /// Construct from a 0..=3 index, returning `None` for out-of-range values.
    pub fn try_from_u8(n: u8) -> Option<Self> {
        match n {
            0 => Some(SubImageIndex::Zero),
            1 => Some(SubImageIndex::One),
            2 => Some(SubImageIndex::Two),
            3 => Some(SubImageIndex::Three),
            _ => None,
        }
    }

    /// The next sub-image in transmission order (§D.3 sequential order).
    /// `Zero -> One -> Two -> Three -> None` (no successor after the last).
    pub fn next(self) -> Option<Self> {
        match self {
            SubImageIndex::Zero => Some(SubImageIndex::One),
            SubImageIndex::One => Some(SubImageIndex::Two),
            SubImageIndex::Two => Some(SubImageIndex::Three),
            SubImageIndex::Three => None,
        }
    }
}

/// Error returned by [`parse_still_image_tr`] when a 5-bit TR field
/// cannot be interpreted as an Annex D still-image temporal reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnnexDTrError {
    /// The TR's top 3 bits are non-zero. §D.3 requires the three
    /// higher bits of TR to be set to 0 when `HI_RES = 0`.
    HighBitsNonZero { tr: u8 },
    /// The TR exceeds the 5-bit field. Programming error; the caller
    /// should have masked with `0b11111`.
    OutOfRange { tr: u8 },
}

/// Encode a sub-image index into the 5-bit TR field, per §D.3:
/// the two lower bits of TR identify the sub-image; the three higher
/// bits shall be 0. The returned `u8` has only the low 5 bits set
/// (range `0..=3`).
pub fn still_image_tr(idx: SubImageIndex) -> u8 {
    idx.as_u8() & 0b11
}

/// Parse a 5-bit TR field from a picture transmitted with `HI_RES = 0`
/// and return the sub-image index.
///
/// `tr` must be the value of the parsed TR field (5 bits, `0..=31`).
/// Returns [`AnnexDTrError::HighBitsNonZero`] if `tr & 0b11100` is
/// non-zero — a violation of §D.3 that the caller may want to surface
/// as a malformed-Annex-D-stream diagnostic. Returns
/// [`AnnexDTrError::OutOfRange`] for `tr > 0b11111`.
pub fn parse_still_image_tr(tr: u8) -> Result<SubImageIndex, AnnexDTrError> {
    if tr > 0b11111 {
        return Err(AnnexDTrError::OutOfRange { tr });
    }
    if tr & 0b11100 != 0 {
        return Err(AnnexDTrError::HighBitsNonZero { tr });
    }
    // Bottom 2 bits are 0..=3, always a valid index by construction.
    Ok(SubImageIndex::from_u8(tr & 0b11))
}

/// Still-image dimensions for a given currently transmitted video
/// format (§D.2): the still image is four times the video format.
///
/// * QCIF (176 × 144) → still image (352 × 288), i.e. the CIF size.
/// * CIF (352 × 288) → still image (704 × 576), a CCIR-601 luma frame.
///
/// The chroma planes follow the 4:2:0 ratio (still-image luma /
/// 2 in each dimension) — see [`still_image_chroma_dimensions`].
pub fn still_image_dimensions(fmt: SourceFormat) -> (u32, u32) {
    let (w, h) = fmt.dimensions();
    (w * 2, h * 2)
}

/// Still-image chroma dimensions for a given currently transmitted
/// video format, assuming H.261's 4:2:0 chroma sub-sampling.
pub fn still_image_chroma_dimensions(fmt: SourceFormat) -> (u32, u32) {
    let (w, h) = still_image_dimensions(fmt);
    (w / 2, h / 2)
}

/// Within each 2 × 2 tile of the still image, the per-sub-image origin
/// `(dx, dy)` (Figure D.1 pattern, repeated with period 2):
///
/// ```text
///   idx | (dx, dy)
///   ----+----------
///    0  | (0, 0)
///    1  | (0, 1)
///    2  | (1, 1)
///    3  | (1, 0)
/// ```
pub fn subimage_origin(idx: SubImageIndex) -> (u32, u32) {
    match idx {
        SubImageIndex::Zero => (0, 0),
        SubImageIndex::One => (0, 1),
        SubImageIndex::Two => (1, 1),
        SubImageIndex::Three => (1, 0),
    }
}

/// 2:1 × 2:1 sub-sample one plane (luma or chroma) of the full
/// still-image plane into the four Annex D sub-images, returning a
/// `[Vec<u8>; 4]` indexed by [`SubImageIndex::as_u8`].
///
/// `still_plane` is `still_w * still_h` bytes, row-major top-to-bottom.
/// Each output sub-image plane is `(still_w / 2) * (still_h / 2)` bytes.
///
/// Panics if `still_w` or `still_h` is odd, or if the input length
/// doesn't match. Annex D's spec'd still-image dimensions are always
/// even (a multiple of the video format size), so the only callers that
/// can hit those panics have mismatched their own dimensions.
pub fn subsample_plane(still_plane: &[u8], still_w: u32, still_h: u32) -> [Vec<u8>; 4] {
    assert!(still_w % 2 == 0, "Annex D still-image width must be even");
    assert!(still_h % 2 == 0, "Annex D still-image height must be even");
    let expected = (still_w as usize) * (still_h as usize);
    assert_eq!(
        still_plane.len(),
        expected,
        "still_plane length {} doesn't match {} x {}",
        still_plane.len(),
        still_w,
        still_h
    );

    let sub_w = (still_w / 2) as usize;
    let sub_h = (still_h / 2) as usize;
    let mut out: [Vec<u8>; 4] = [
        vec![0u8; sub_w * sub_h],
        vec![0u8; sub_w * sub_h],
        vec![0u8; sub_w * sub_h],
        vec![0u8; sub_w * sub_h],
    ];

    for idx_u8 in 0u8..4 {
        let idx = SubImageIndex::from_u8(idx_u8);
        let (dx, dy) = subimage_origin(idx);
        let dst = &mut out[idx_u8 as usize];
        for y in 0..sub_h {
            let src_y = y * 2 + (dy as usize);
            let src_row = &still_plane[src_y * (still_w as usize)..][..(still_w as usize)];
            let dst_row = &mut dst[y * sub_w..][..sub_w];
            for x in 0..sub_w {
                let src_x = x * 2 + (dx as usize);
                dst_row[x] = src_row[src_x];
            }
        }
    }
    out
}

/// 2:1 × 2:1 sub-sample a full-resolution still image (YUV 4:2:0) into
/// four sub-images each at the currently transmitted video format
/// (§D.2 + Figure D.1).
///
/// `still_y`, `still_cb`, `still_cr` are tightly packed row-major planes
/// of dimensions [`still_image_dimensions`] (luma) and
/// [`still_image_chroma_dimensions`] (chroma) for `fmt`.
///
/// Returns four `SubImagePlanes` in transmission order
/// (0 → 1 → 2 → 3); each `SubImagePlanes` carries Y/Cb/Cr tightly
/// packed at the standard video-format dimensions for `fmt`.
pub fn subsample_still_image(
    fmt: SourceFormat,
    still_y: &[u8],
    still_cb: &[u8],
    still_cr: &[u8],
) -> [SubImagePlanes; 4] {
    let (sw, sh) = still_image_dimensions(fmt);
    let (cw, ch) = still_image_chroma_dimensions(fmt);
    let y_planes = subsample_plane(still_y, sw, sh);
    let cb_planes = subsample_plane(still_cb, cw, ch);
    let cr_planes = subsample_plane(still_cr, cw, ch);

    let [y0, y1, y2, y3] = y_planes;
    let [cb0, cb1, cb2, cb3] = cb_planes;
    let [cr0, cr1, cr2, cr3] = cr_planes;
    [
        SubImagePlanes {
            y: y0,
            cb: cb0,
            cr: cr0,
        },
        SubImagePlanes {
            y: y1,
            cb: cb1,
            cr: cr1,
        },
        SubImagePlanes {
            y: y2,
            cb: cb2,
            cr: cr2,
        },
        SubImagePlanes {
            y: y3,
            cb: cb3,
            cr: cr3,
        },
    ]
}

/// Tight-packed Y / Cb / Cr planes for a single Annex D sub-image at
/// the currently transmitted video format's dimensions. Used as the
/// element type of the [`subsample_still_image`] return value and as
/// the input to [`reassemble_still_image`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubImagePlanes {
    /// Luma plane, `video_w * video_h` bytes (e.g. 176 × 144 for QCIF).
    pub y: Vec<u8>,
    /// Cb plane, `(video_w / 2) * (video_h / 2)` bytes.
    pub cb: Vec<u8>,
    /// Cr plane, `(video_w / 2) * (video_h / 2)` bytes.
    pub cr: Vec<u8>,
}

/// Inverse of [`subsample_plane`]: re-assemble four sub-image planes
/// into a single full-resolution still-image plane of dimensions
/// `still_w × still_h` (with `still_w` and `still_h` both even).
///
/// `subs[i]` corresponds to [`SubImageIndex::from_u8(i)`] and must have
/// length `(still_w / 2) * (still_h / 2)`. The reverse interleave
/// places sample `subs[idx][y * sub_w + x]` at still-image position
/// `(2x + dx, 2y + dy)` where `(dx, dy) = subimage_origin(idx)`.
///
/// Panics on a dimension mismatch (symmetric with [`subsample_plane`]).
pub fn reassemble_plane(subs: &[Vec<u8>; 4], still_w: u32, still_h: u32) -> Vec<u8> {
    assert!(still_w % 2 == 0, "Annex D still-image width must be even");
    assert!(still_h % 2 == 0, "Annex D still-image height must be even");
    let sub_w = (still_w / 2) as usize;
    let sub_h = (still_h / 2) as usize;
    for (i, p) in subs.iter().enumerate() {
        assert_eq!(
            p.len(),
            sub_w * sub_h,
            "sub-image {i} length {} doesn't match {} x {}",
            p.len(),
            sub_w,
            sub_h
        );
    }
    let mut out = vec![0u8; (still_w as usize) * (still_h as usize)];
    for idx_u8 in 0u8..4 {
        let idx = SubImageIndex::from_u8(idx_u8);
        let (dx, dy) = subimage_origin(idx);
        let src = &subs[idx_u8 as usize];
        for y in 0..sub_h {
            let dst_y = y * 2 + (dy as usize);
            let dst_row = &mut out[dst_y * (still_w as usize)..][..(still_w as usize)];
            let src_row = &src[y * sub_w..][..sub_w];
            for x in 0..sub_w {
                let dst_x = x * 2 + (dx as usize);
                dst_row[dst_x] = src_row[x];
            }
        }
    }
    out
}

/// Inverse of [`subsample_still_image`]: re-assemble four decoded
/// sub-image YUV-4:2:0 frames into a single full-resolution still-image
/// triple `(y, cb, cr)` of [`still_image_dimensions`] for luma and
/// [`still_image_chroma_dimensions`] for chroma.
///
/// `subs[i]` corresponds to [`SubImageIndex::from_u8(i)`].
pub fn reassemble_still_image(
    fmt: SourceFormat,
    subs: &[SubImagePlanes; 4],
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (sw, sh) = still_image_dimensions(fmt);
    let (cw, ch) = still_image_chroma_dimensions(fmt);
    let y_subs = [
        subs[0].y.clone(),
        subs[1].y.clone(),
        subs[2].y.clone(),
        subs[3].y.clone(),
    ];
    let cb_subs = [
        subs[0].cb.clone(),
        subs[1].cb.clone(),
        subs[2].cb.clone(),
        subs[3].cb.clone(),
    ];
    let cr_subs = [
        subs[0].cr.clone(),
        subs[1].cr.clone(),
        subs[2].cr.clone(),
        subs[3].cr.clone(),
    ];
    let y = reassemble_plane(&y_subs, sw, sh);
    let cb = reassemble_plane(&cb_subs, cw, ch);
    let cr = reassemble_plane(&cr_subs, cw, ch);
    (y, cb, cr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_image_index_round_trip_u8() {
        for n in 0u8..4 {
            let idx = SubImageIndex::from_u8(n);
            assert_eq!(idx.as_u8(), n);
            assert_eq!(SubImageIndex::try_from_u8(n), Some(idx));
        }
        assert_eq!(SubImageIndex::try_from_u8(4), None);
        assert_eq!(SubImageIndex::try_from_u8(255), None);
    }

    #[test]
    fn sub_image_next_is_sequential() {
        assert_eq!(SubImageIndex::Zero.next(), Some(SubImageIndex::One));
        assert_eq!(SubImageIndex::One.next(), Some(SubImageIndex::Two));
        assert_eq!(SubImageIndex::Two.next(), Some(SubImageIndex::Three));
        assert_eq!(SubImageIndex::Three.next(), None);
    }

    #[test]
    fn still_image_tr_low_bits_only() {
        // §D.3: only the two low bits of TR encode the sub-image; the
        // three high bits shall be 0.
        for n in 0u8..4 {
            let idx = SubImageIndex::from_u8(n);
            let tr = still_image_tr(idx);
            assert!(tr <= 3, "tr {tr} should fit in 2 bits");
            assert_eq!(tr & 0b11100, 0, "high 3 bits of TR must be 0");
            assert_eq!(parse_still_image_tr(tr), Ok(idx));
        }
    }

    #[test]
    fn parse_still_image_tr_rejects_high_bits() {
        // tr with any of bits 2..=4 set ⇒ violation of §D.3.
        for tr in [0b00100, 0b01000, 0b10000, 0b11111, 0b11100] {
            assert_eq!(
                parse_still_image_tr(tr),
                Err(AnnexDTrError::HighBitsNonZero { tr })
            );
        }
    }

    #[test]
    fn parse_still_image_tr_rejects_out_of_range() {
        for tr in [32u8, 33, 64, 128, 255] {
            assert_eq!(
                parse_still_image_tr(tr),
                Err(AnnexDTrError::OutOfRange { tr })
            );
        }
    }

    #[test]
    fn still_image_dimensions_are_4x_video_format() {
        assert_eq!(still_image_dimensions(SourceFormat::Qcif), (352, 288));
        assert_eq!(still_image_dimensions(SourceFormat::Cif), (704, 576));
        assert_eq!(
            still_image_chroma_dimensions(SourceFormat::Qcif),
            (176, 144)
        );
        assert_eq!(still_image_chroma_dimensions(SourceFormat::Cif), (352, 288));
    }

    #[test]
    fn subimage_origin_matches_figure_d1() {
        // The 2 × 2 tile labelled (with rows = y, cols = x):
        //   row=0: 0 3
        //   row=1: 1 2
        assert_eq!(subimage_origin(SubImageIndex::Zero), (0, 0));
        assert_eq!(subimage_origin(SubImageIndex::One), (0, 1));
        assert_eq!(subimage_origin(SubImageIndex::Two), (1, 1));
        assert_eq!(subimage_origin(SubImageIndex::Three), (1, 0));
    }

    /// On a small `4 x 4` plane the labels written into each cell
    /// must mirror Figure D.1 when sub-sampled and dispatched into
    /// the four sub-images, then re-assembled bit-exact.
    #[test]
    fn subsample_then_reassemble_round_trip_small() {
        // Synthetic 4 x 4 plane labelled by (idx within tile, x, y) — we
        // use a value that uniquely identifies the source position so we
        // can verify the reverse mapping.
        let w = 4u32;
        let h = 4u32;
        let mut plane = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                plane[(y * w + x) as usize] = ((y * 16) + x) as u8;
            }
        }
        let subs = subsample_plane(&plane, w, h);
        // Each sub-image is 2 x 2.
        for p in &subs {
            assert_eq!(p.len(), 4);
        }
        // Sub-image 0 picks origin (0, 0): plane[(0,0), (2,0), (0,2), (2,2)].
        assert_eq!(subs[0], vec![0, 2, 32, 34]);
        // Sub-image 1 picks origin (0, 1): plane[(0,1), (2,1), (0,3), (2,3)].
        assert_eq!(subs[1], vec![16, 18, 48, 50]);
        // Sub-image 2 picks origin (1, 1): plane[(1,1), (3,1), (1,3), (3,3)].
        assert_eq!(subs[2], vec![17, 19, 49, 51]);
        // Sub-image 3 picks origin (1, 0): plane[(1,0), (3,0), (1,2), (3,2)].
        assert_eq!(subs[3], vec![1, 3, 33, 35]);

        // Round-trip exactly.
        let recon = reassemble_plane(&subs, w, h);
        assert_eq!(recon, plane);
    }

    /// Round-trip the QCIF / CIF still image planes (§D.2 spec'd sizes)
    /// through subsample → reassemble. Each sub-image becomes the
    /// currently transmitted video format.
    #[test]
    fn still_image_round_trip_qcif() {
        // QCIF → still image is CIF (352 × 288), each sub-image is QCIF.
        let fmt = SourceFormat::Qcif;
        let (sw, sh) = still_image_dimensions(fmt);
        let (cw, ch) = still_image_chroma_dimensions(fmt);
        assert_eq!((sw, sh), (352, 288));
        assert_eq!((cw, ch), (176, 144));

        // Synthesise a deterministic still image (PRNG content); the
        // exact content doesn't matter for the round trip, only that
        // every sample is unique enough to detect a swap.
        let mut y = vec![0u8; (sw * sh) as usize];
        let mut cb = vec![0u8; (cw * ch) as usize];
        let mut cr = vec![0u8; (cw * ch) as usize];
        let mut s: u32 = 0x1234_5678;
        for b in y.iter_mut().chain(cb.iter_mut()).chain(cr.iter_mut()) {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (s >> 16) as u8;
        }

        let subs = subsample_still_image(fmt, &y, &cb, &cr);
        // Each sub-image is QCIF.
        let (vw, vh) = fmt.dimensions();
        let (cvw, cvh) = (vw / 2, vh / 2);
        for s in &subs {
            assert_eq!(s.y.len(), (vw * vh) as usize);
            assert_eq!(s.cb.len(), (cvw * cvh) as usize);
            assert_eq!(s.cr.len(), (cvw * cvh) as usize);
        }
        let (y2, cb2, cr2) = reassemble_still_image(fmt, &subs);
        assert_eq!(y2, y);
        assert_eq!(cb2, cb);
        assert_eq!(cr2, cr);
    }

    #[test]
    fn still_image_round_trip_cif() {
        // CIF → still image is 704 × 576, each sub-image is CIF.
        let fmt = SourceFormat::Cif;
        let (sw, sh) = still_image_dimensions(fmt);
        let (cw, ch) = still_image_chroma_dimensions(fmt);
        assert_eq!((sw, sh), (704, 576));
        assert_eq!((cw, ch), (352, 288));
        let mut y = vec![0u8; (sw * sh) as usize];
        let mut cb = vec![0u8; (cw * ch) as usize];
        let mut cr = vec![0u8; (cw * ch) as usize];
        let mut s: u32 = 0xCAFE_BABE;
        for b in y.iter_mut().chain(cb.iter_mut()).chain(cr.iter_mut()) {
            s = s.wrapping_mul(48271);
            *b = (s >> 16) as u8;
        }
        let subs = subsample_still_image(fmt, &y, &cb, &cr);
        let (y2, cb2, cr2) = reassemble_still_image(fmt, &subs);
        assert_eq!(y2, y);
        assert_eq!(cb2, cb);
        assert_eq!(cr2, cr);
    }

    /// Each sub-image covers exactly one of the four cells of every
    /// 2 × 2 tile — no overlap, no gaps. Verified by checking that
    /// after summing the four sub-image counts at every position of
    /// the still image, every position is covered exactly once.
    #[test]
    fn sub_image_origins_partition_the_2x2_tile() {
        let mut hits = [[0u8; 2]; 2];
        for n in 0u8..4 {
            let (dx, dy) = subimage_origin(SubImageIndex::from_u8(n));
            hits[dy as usize][dx as usize] += 1;
        }
        assert_eq!(hits, [[1u8; 2]; 2]);
    }
}
