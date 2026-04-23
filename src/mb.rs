//! H.261 macroblock-layer decoding — §4.2.3 of ITU-T Rec. H.261.
//!
//! Per-MB decode sequence:
//! 1. **MBA** (VLC, Table 1) — absolute MB address for the first MB of a
//!    GOB; otherwise an address *difference* from the previous coded MB.
//!    MBA stuffing is discarded; GBSC is not a normal MBA entry (detected
//!    by the start-code scanner in `decoder.rs`).
//! 2. **MTYPE** (VLC, Table 2) — selects INTRA / INTER / INTER+MC /
//!    INTER+MC+FIL and tells us whether MQUANT, MVD, CBP, TCOEFF are present.
//! 3. **MQUANT** (5-bit FLC) — new quantiser if flagged.
//! 4. **MVD** — two VLCs (x then y); each decoded with the paired
//!    representative + the previous-MB predictor per §4.2.3.4. Only present
//!    for INTER+MC / INTER+MC+FIL.
//! 5. **CBP** (VLC, Table 4) — 6-bit CBP packed as `32*P1 + 16*P2 + 8*P3 +
//!    4*P4 + 2*P5 + P6`, where `Pn` is the luma block n=1..=4 or chroma
//!    n=5..=6 (Figure 10). Present when MTYPE says so.
//! 6. **Block layer** — 6 blocks transmitted in order Y0, Y1, Y2, Y3, Cb, Cr.
//!    INTRA always transmits all six. INTER transmits only the CBP-flagged
//!    ones; the rest come from the motion-compensated predictor (or zero
//!    residual when MC is not signalled).
//!
//! # Motion compensation
//!
//! H.261 MC is **integer-pel only**, with each luma component in `-15..=15`.
//! Chroma MVs are `floor(abs(luma)/2) * sign(luma)` (spec: halved, truncated
//! toward zero; see `luma_to_chroma_mv`).
//!
//! MVD is signalled as a paired differential (Table 3) — for each code the
//! decoder picks whichever paired representative brings the reconstructed MV
//! into `-15..=15`. The predictor is the previous coded MB's vector, reset
//! to zero at the start of every GOB, at MBA discontinuity (non-consecutive
//! MB address), and whenever the previous MB was not MC-coded (§4.2.3.4).
//!
//! # Loop filter (§3.2.3)
//!
//! When MTYPE carries the FIL flag, a separable 1/4-1/2-1/4 filter is
//! applied to each predicted 8x8 block before the residual is added. The
//! filter taps at block edges are replaced with `0, 1, 0` (no change).

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

use crate::block::decode_inter_block;
use crate::tables::{
    decode_vlc, MbaSym, MtypeInfo, MvdSym, Prediction, CBP_TABLE, MBA_TABLE, MTYPE_TABLE, MVD_TABLE,
};

/// Reconstructed H.261 picture: three pel planes (Y, Cb, Cr), stride = MB-aligned width.
pub struct Picture {
    pub width: usize,
    pub height: usize,
    pub mb_width: usize,
    pub mb_height: usize,
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
    pub y_stride: usize,
    pub c_stride: usize,
}

impl Picture {
    pub fn new(width: usize, height: usize) -> Self {
        let mb_w = width.div_ceil(16);
        let mb_h = height.div_ceil(16);
        let y_stride = mb_w * 16;
        let c_stride = mb_w * 8;
        Self {
            width,
            height,
            mb_width: mb_w,
            mb_height: mb_h,
            y_stride,
            c_stride,
            y: vec![0u8; y_stride * mb_h * 16],
            cb: vec![0u8; c_stride * mb_h * 8],
            cr: vec![0u8; c_stride * mb_h * 8],
        }
    }
}

/// State kept between consecutive MBs inside a GOB — the MV predictor and
/// the last MBA seen (to detect non-consecutive addresses per §4.2.3.4).
#[derive(Clone, Copy, Debug)]
pub struct MbContext {
    pub mv: (i32, i32),
    /// True iff the previous MB was MC-coded (InterMc / InterMcFil). Reset at
    /// GOB start.
    pub prev_was_mc: bool,
    /// Previous MBA within the GOB. `0` means "no previous MB yet".
    pub prev_mba: u8,
}

impl MbContext {
    pub fn reset() -> Self {
        Self {
            mv: (0, 0),
            prev_was_mc: false,
            prev_mba: 0,
        }
    }
}

/// Build a neutral-grey reference picture (Y=128, Cb=Cr=128) of the given
/// dimensions. Used as the first-picture fallback when an encoder emits
/// INTER MBs before any I-frame establishes a reference.
fn grey_reference(width: usize, height: usize) -> Picture {
    let mut p = Picture::new(width, height);
    p.y.fill(128);
    p.cb.fill(128);
    p.cr.fill(128);
    p
}

/// Chroma MV per §3.2.2: halve + truncate toward zero.
pub fn luma_to_chroma_mv(v: i32) -> i32 {
    // `/ 2` in Rust truncates toward zero for signed.
    v / 2
}

/// Fold MVD paired representative into `[-15, 15]` given the predictor.
/// Returns the chosen signed reconstructed MV.
///
/// §4.2.3.4: exactly one of `predictor + a` and `predictor + b` is in the
/// range `[-15, 15]`; we pick that one. If both are in-range (impossible
/// when `|predictor| <= 15` and `|a-b| == 32` with `a != 0`), we pick `a`.
fn reconstruct_mv(predictor: i32, sym: MvdSym) -> i32 {
    let a = predictor + sym.a as i32;
    let b = predictor + sym.b as i32;
    let in_range = |v: i32| (-15..=15).contains(&v);
    match (in_range(a), in_range(b)) {
        (true, _) => a,
        (false, true) => b,
        (false, false) => a, // shouldn't happen on well-formed streams
    }
}

/// Decode a single macroblock at MB position `(mb_col, mb_row)` within the
/// current GOB's top-left origin `(gob_x, gob_y)` (in luma pels). Updates
/// `ctx` for the next MB, and writes into `pic`.
///
/// `reference` is optional — it's `None` for the very first I-picture (in
/// which case an INTER MB is a decode error) and `Some(&prev_picture)` for
/// every subsequent picture.
#[allow(clippy::too_many_arguments)]
pub fn decode_macroblock(
    br: &mut BitReader<'_>,
    mba: u8,
    gob_x: usize,
    gob_y: usize,
    quant: &mut u32,
    ctx: &mut MbContext,
    pic: &mut Picture,
    reference: Option<&Picture>,
) -> Result<()> {
    // Figure out where this MB is in luma pels.
    let idx = (mba - 1) as usize;
    let mb_col = idx % 11;
    let mb_row = idx / 11;
    let luma_x = gob_x + mb_col * 16;
    let luma_y = gob_y + mb_row * 16;

    // MTYPE.
    let mtype: MtypeInfo = decode_vlc(br, MTYPE_TABLE)?;

    // MQUANT (if present).
    if mtype.mquant {
        let q = br.read_u32(5)?;
        if q == 0 {
            return Err(Error::invalid("h261 MB: MQUANT == 0"));
        }
        *quant = q;
    }

    // MVD (if present). Predictor is the previous MB's MV if the previous MB
    // was MC-coded AND this MBA is consecutive with the previous one;
    // otherwise zero.
    let mut mv = (0i32, 0i32);
    if mtype.mvd {
        // Decide predictor.
        let pred = if ctx.prev_was_mc && ctx.prev_mba != 0 && mba == ctx.prev_mba + 1 {
            ctx.mv
        } else {
            (0, 0)
        };
        let sym_x: MvdSym = decode_vlc(br, MVD_TABLE)?;
        let sym_y: MvdSym = decode_vlc(br, MVD_TABLE)?;
        let mx = reconstruct_mv(pred.0, sym_x);
        let my = reconstruct_mv(pred.1, sym_y);
        mv = (mx, my);
    }

    // CBP — if absent, either:
    //   * INTRA: all 6 blocks coded.
    //   * INTER+MC without CBP (InterMc variant with tcoeff=false): no blocks coded.
    let cbp: u8 = if mtype.cbp {
        decode_vlc(br, CBP_TABLE)?
    } else if mtype.prediction == Prediction::Intra {
        0b111111
    } else {
        // Inter+MC with tcoeff=false: no blocks coded, just prediction.
        0
    };

    // Block order is Y1, Y2, Y3, Y4, Cb, Cr (Figure 10).
    // Per CBP formula: CBP = 32*P1 + 16*P2 + 8*P3 + 4*P4 + 2*P5 + P6.
    let block_coded = [
        (cbp >> 5) & 1 != 0, // Y1 - top-left
        (cbp >> 4) & 1 != 0, // Y2 - top-right
        (cbp >> 3) & 1 != 0, // Y3 - bottom-left
        (cbp >> 2) & 1 != 0, // Y4 - bottom-right
        (cbp >> 1) & 1 != 0, // Cb
        cbp & 1 != 0,        // Cr
    ];

    // INTRA path: each block is self-contained (INTRA DC + AC + IDCT, pel-domain).
    if mtype.prediction == Prediction::Intra {
        for i in 0..6 {
            // INTRA always codes every block irrespective of CBP flagging.
            let _ = block_coded[i];
            let mut out = [0u8; 64];
            crate::block::decode_intra_block(br, *quant, &mut out)?;
            write_block(pic, i, luma_x, luma_y, &out);
        }
        ctx.mv = (0, 0);
        ctx.prev_was_mc = false;
        ctx.prev_mba = mba;
        return Ok(());
    }

    // INTER path: build predictor, add residual. When there's no prior
    // reference we fall back to a mid-grey reference — H.261 doesn't
    // explicitly tag I vs P pictures, and some encoders emit inter MBs even
    // in the very first picture (their internal reference is initialised to
    // 128).
    let fallback_ref;
    let reference_ref: &Picture = match reference {
        Some(r) => r,
        None => {
            fallback_ref = grey_reference(pic.width, pic.height);
            &fallback_ref
        }
    };
    let reference = reference_ref;

    // Per-block MV: luma MV for Y1..Y4; chroma MV for Cb/Cr.
    let (mvx, mvy) = mv;
    let cmvx = luma_to_chroma_mv(mvx);
    let cmvy = luma_to_chroma_mv(mvy);

    for i in 0..4usize {
        // Luma block top-left (in luma pels).
        let (sub_x, sub_y) = match i {
            0 => (0, 0),
            1 => (8, 0),
            2 => (0, 8),
            3 => (8, 8),
            _ => unreachable!(),
        };
        let bx = (luma_x + sub_x) as i32;
        let by = (luma_y + sub_y) as i32;
        let mut pred = [0u8; 64];
        copy_block_integer(
            &reference.y,
            reference.y_stride,
            reference.y_stride as i32,
            (reference.y.len() / reference.y_stride) as i32,
            bx,
            by,
            mvx,
            mvy,
            &mut pred,
        );
        if mtype.filter {
            pred = apply_loop_filter(&pred);
        }
        if block_coded[i] && mtype.tcoeff {
            let mut resid = [0i32; 64];
            decode_inter_block(br, *quant, &mut resid)?;
            let mut out = [0u8; 64];
            for j in 0..64 {
                out[j] = (pred[j] as i32 + resid[j]).clamp(0, 255) as u8;
            }
            write_block(pic, i, luma_x, luma_y, &out);
        } else {
            write_block(pic, i, luma_x, luma_y, &pred);
        }
    }

    // Chroma blocks.
    for ci in 0..2usize {
        let (ref_plane, ref_stride) = if ci == 0 {
            (&reference.cb, reference.c_stride)
        } else {
            (&reference.cr, reference.c_stride)
        };
        let ref_h = (ref_plane.len() / ref_stride) as i32;
        let cx = (luma_x / 2) as i32;
        let cy = (luma_y / 2) as i32;
        let mut pred = [0u8; 64];
        copy_block_integer(
            ref_plane,
            ref_stride,
            ref_stride as i32,
            ref_h,
            cx,
            cy,
            cmvx,
            cmvy,
            &mut pred,
        );
        if mtype.filter {
            pred = apply_loop_filter(&pred);
        }
        let block_i = 4 + ci;
        if block_coded[block_i] && mtype.tcoeff {
            let mut resid = [0i32; 64];
            decode_inter_block(br, *quant, &mut resid)?;
            let mut out = [0u8; 64];
            for j in 0..64 {
                out[j] = (pred[j] as i32 + resid[j]).clamp(0, 255) as u8;
            }
            write_block(pic, block_i, luma_x, luma_y, &out);
        } else {
            write_block(pic, block_i, luma_x, luma_y, &pred);
        }
    }

    ctx.mv = mv;
    ctx.prev_was_mc = matches!(
        mtype.prediction,
        Prediction::InterMc | Prediction::InterMcFil
    );
    ctx.prev_mba = mba;
    Ok(())
}

/// Copy an 8x8 integer-pel block from `ref_plane` at luma position `(bx,by)`
/// plus `(mvx, mvy)`. Samples outside the plane are clamped to the nearest edge.
#[allow(clippy::too_many_arguments)]
fn copy_block_integer(
    ref_plane: &[u8],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    bx: i32,
    by: i32,
    mvx: i32,
    mvy: i32,
    out: &mut [u8; 64],
) {
    let sx = bx + mvx;
    let sy = by + mvy;
    for j in 0..8 {
        for i in 0..8 {
            let x = (sx + i).clamp(0, ref_w - 1) as usize;
            let y = (sy + j).clamp(0, ref_h - 1) as usize;
            out[(j as usize) * 8 + i as usize] = ref_plane[y * ref_stride + x];
        }
    }
}

/// Apply the H.261 loop filter (§3.2.3) to an 8x8 block. Separable 1/4-1/2-1/4
/// filter with edge taps replaced by 0-1-0. Rounds to nearest, half-up.
fn apply_loop_filter(src: &[u8; 64]) -> [u8; 64] {
    // Horizontal pass.
    let mut h = [0i32; 64];
    for j in 0..8 {
        for i in 0..8 {
            let v = if i == 0 || i == 7 {
                src[j * 8 + i] as i32
            } else {
                let a = src[j * 8 + i - 1] as i32;
                let b = src[j * 8 + i] as i32;
                let c = src[j * 8 + i + 1] as i32;
                // 1/4 a + 1/2 b + 1/4 c = (a + 2b + c) / 4; round half-up.
                (a + 2 * b + c + 2) >> 2
            };
            h[j * 8 + i] = v;
        }
    }
    // Vertical pass.
    let mut out = [0u8; 64];
    for i in 0..8 {
        for j in 0..8 {
            let v = if j == 0 || j == 7 {
                h[j * 8 + i]
            } else {
                let a = h[(j - 1) * 8 + i];
                let b = h[j * 8 + i];
                let c = h[(j + 1) * 8 + i];
                (a + 2 * b + c + 2) >> 2
            };
            out[j * 8 + i] = v.clamp(0, 255) as u8;
        }
    }
    out
}

/// Write a reconstructed 8x8 block into the picture. `block_idx`:
/// 0-3 = Y1..Y4 (arrangement per Figure 10), 4 = Cb, 5 = Cr.
fn write_block(pic: &mut Picture, block_idx: usize, luma_x: usize, luma_y: usize, out: &[u8; 64]) {
    let (plane, stride, px, py): (&mut [u8], usize, usize, usize) = match block_idx {
        0 => (pic.y.as_mut_slice(), pic.y_stride, luma_x, luma_y),
        1 => (pic.y.as_mut_slice(), pic.y_stride, luma_x + 8, luma_y),
        2 => (pic.y.as_mut_slice(), pic.y_stride, luma_x, luma_y + 8),
        3 => (pic.y.as_mut_slice(), pic.y_stride, luma_x + 8, luma_y + 8),
        4 => (pic.cb.as_mut_slice(), pic.c_stride, luma_x / 2, luma_y / 2),
        5 => (pic.cr.as_mut_slice(), pic.c_stride, luma_x / 2, luma_y / 2),
        _ => unreachable!(),
    };
    for j in 0..8 {
        for i in 0..8 {
            plane[(py + j) * stride + (px + i)] = out[j * 8 + i];
        }
    }
}

/// Decode the MBA VLC plus any stuffing codewords that precede it. Returns
/// `None` if the reader hits a start-code boundary (the caller has already
/// positioned us past the 16-bit zero prefix; from the MB scanner's
/// perspective the GOB has ended).
///
/// Returns `Some(mba_diff)` where `mba_diff` is the value from Table 1 (1..=33).
pub fn decode_mba_diff(br: &mut BitReader<'_>) -> Result<Option<u8>> {
    loop {
        // Peek 16 bits to see if we're at a start code (15 leading zeros
        // followed by a `1`). When the encoder inserts MBA stuffing, it can
        // emit arbitrarily many stuffing codewords; we also need to handle
        // a real start code as the end-of-GOB signal.
        let avail = br.bits_remaining().min(16) as u32;
        if avail < 16 {
            return Ok(None);
        }
        let peek = br.peek_u32(16)?;
        if peek == 0x0001 {
            return Ok(None);
        }
        // Not a start code — decode a regular MBA VLC.
        let sym: MbaSym = decode_vlc(br, MBA_TABLE)?;
        match sym {
            MbaSym::Diff(d) => return Ok(Some(d)),
            MbaSym::Stuffing => continue,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chroma_mv_truncate_toward_zero() {
        assert_eq!(luma_to_chroma_mv(0), 0);
        assert_eq!(luma_to_chroma_mv(1), 0);
        assert_eq!(luma_to_chroma_mv(2), 1);
        assert_eq!(luma_to_chroma_mv(3), 1);
        assert_eq!(luma_to_chroma_mv(-1), 0);
        assert_eq!(luma_to_chroma_mv(-2), -1);
        assert_eq!(luma_to_chroma_mv(-3), -1);
        assert_eq!(luma_to_chroma_mv(15), 7);
        assert_eq!(luma_to_chroma_mv(-15), -7);
    }

    #[test]
    fn mv_paired_selection_in_range() {
        // Predictor 10, MVD "a=-2, b=30" → 10 + -2 = 8 (in range).
        let s = MvdSym { a: -2, b: 30 };
        assert_eq!(reconstruct_mv(10, s), 8);
        // Predictor 10, MVD "a=10, b=-22" → 10+10=20 out, 10-22=-12 in → -12.
        let s = MvdSym { a: 10, b: -22 };
        assert_eq!(reconstruct_mv(10, s), -12);
    }

    #[test]
    fn loop_filter_flat_block_identity() {
        let src = [128u8; 64];
        let out = apply_loop_filter(&src);
        for &v in out.iter() {
            assert_eq!(v, 128);
        }
    }
}
