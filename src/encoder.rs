//! H.261 encoder — Baseline (I + P pictures, no MC).
//!
//! Public entry points:
//!
//! * [`encode_intra_picture`] — single I-picture, stateless.
//! * [`encode_inter_picture`] — single P-picture, requires a reconstructed
//!   reference picture (typically the local recon of the previous output).
//! * [`H261Encoder`] — stateful sequence encoder that produces an I-picture
//!   on the first frame and P-pictures thereafter, maintaining the local
//!   reconstruction across calls.
//!
//! The implementation is deliberately simple — no rate control, no motion
//! estimation, no loop filter. P-pictures use the `Inter` MTYPE only (no
//! MC, no FIL); the encoder either emits a residual block or skips the MB
//! (via the MBA-difference jump) when the predicted reconstruction is good
//! enough on its own.
//!
//! ## Picture layer (§4.2.1)
//!
//! ```text
//!   PSC (20)  0000 0000 0000 0001 0000
//!   TR  (5)   temporal reference
//!   PTYPE(6)  bit1 split, bit2 doccam, bit3 freeze-release,
//!             bit4 source format (0=QCIF, 1=CIF), bit5 HI_RES (1=off),
//!             bit6 spare (always 1 per §4.1)
//!   PEI (1)   0 — we never emit PSPARE
//! ```
//!
//! ## GOB layer (§4.2.2)
//!
//! ```text
//!   GBSC (16) 0000 0000 0000 0001
//!   GN   (4)  1..=12 (CIF) or 1,3,5 (QCIF)
//!   GQUANT(5) quantiser index 1..=31
//!   GEI  (1)  0 — we never emit GSPARE
//! ```
//!
//! ## MB layer
//!
//! For INTRA MBs (I-picture and any I MBs in a P-picture):
//!
//! * MBA VLC — `Diff(1)` for the first coded MB, then 1-differences.
//! * MTYPE = `Intra` (4-bit `0001`), no MQUANT override.
//! * CBP is implicit (all 6 blocks are always coded).
//! * 6 blocks (Y1, Y2, Y3, Y4, Cb, Cr). Each block is INTRA DC + AC.
//!
//! For INTER MBs (P-picture, no MC):
//!
//! * MBA VLC — difference from the previous coded MB (skipped MBs are
//!   absorbed into the difference per §4.2.3.3).
//! * MTYPE = `Inter` (1-bit `1`) — has CBP + TCOEFF, no MQUANT/MVD/FIL.
//! * CBP — selects which of the 6 blocks carry residual data. CBP can
//!   never be 0 with this MTYPE (Table 4 starts at CBP=1); if all six
//!   blocks would be uncoded we instead skip the MB entirely.
//! * Each coded block is a TCOEFF VLC stream with `1s` first-coefficient
//!   shortcut and `11s` for subsequent (0,1) coefficients.

use oxideav_core::bits::BitWriter;
use oxideav_core::{Error, Result};

use crate::fdct::{fdct_intra, fdct_signed};
use crate::idct::{idct_intra, idct_signed};
use crate::mb::Picture;
use crate::picture::SourceFormat;
use crate::quant::{quant_ac, quant_intra_dc};
use crate::tables::{
    encode_cbp, encode_mba_diff, lookup_tcoeff, MBA_STUFFING, MTYPE_INTRA, MTYPE_INTRA_MQUANT,
    ZIGZAG,
};

/// Default GOB-level quantiser. QUANT in `1..=31`. 8 is a balanced
/// quality/bit-rate point.
pub const DEFAULT_QUANT: u32 = 8;

/// MTYPE `Inter` — 1-bit `1`. Has CBP + TCOEFF, no MQUANT / MVD / FIL.
const MTYPE_INTER: (u8, u32) = (1, 0b1);

/// Encode a single INTRA picture.
///
/// `y`, `cb`, `cr` are packed planes with the specified strides. `quant`
/// is the GOB-level QUANT (1..=31). `temporal_reference` is the 5-bit TR
/// field (mod 32) the decoder uses for lip-sync.
pub fn encode_intra_picture(
    fmt: SourceFormat,
    y: &[u8],
    y_stride: usize,
    cb: &[u8],
    cb_stride: usize,
    cr: &[u8],
    cr_stride: usize,
    quant: u32,
    temporal_reference: u8,
) -> Result<Vec<u8>> {
    let (bytes, _recon) = encode_intra_picture_with_recon(
        fmt,
        y,
        y_stride,
        cb,
        cb_stride,
        cr,
        cr_stride,
        quant,
        temporal_reference,
    )?;
    Ok(bytes)
}

/// Encode an INTRA picture and also return a locally reconstructed
/// `Picture` matching what a conformant decoder would produce. The
/// reconstruction can be passed to [`encode_inter_picture`] as the
/// reference for the next P-frame.
pub fn encode_intra_picture_with_recon(
    fmt: SourceFormat,
    y: &[u8],
    y_stride: usize,
    cb: &[u8],
    cb_stride: usize,
    cr: &[u8],
    cr_stride: usize,
    quant: u32,
    temporal_reference: u8,
) -> Result<(Vec<u8>, Picture)> {
    validate_inputs(
        fmt,
        y,
        y_stride,
        cb,
        cb_stride,
        cr,
        cr_stride,
        quant,
        temporal_reference,
    )?;

    let (w, h) = fmt.dimensions();
    let mut recon = Picture::new(w as usize, h as usize);

    let mut bw = BitWriter::with_capacity(4096);
    write_picture_header(&mut bw, fmt, temporal_reference);

    for &gn in fmt.gob_numbers() {
        write_gob_header(&mut bw, gn, quant);
        let (gob_x, gob_y) = gob_origin_luma(fmt, gn);
        encode_gob_intra(
            &mut bw, y, y_stride, cb, cb_stride, cr, cr_stride, gob_x, gob_y, quant, &mut recon,
        );
    }

    bw.align_to_byte();
    Ok((bw.finish(), recon))
}

/// Encode a single INTER (P) picture against a reference reconstruction.
///
/// Returns the elementary-stream bytes and an updated reconstruction
/// suitable for use as the reference for the next P-frame.
///
/// Quantisation strategy: each MB is tested for "skippable" (residual all
/// zero after quantisation); if not skippable, encode it as INTER (no MC).
/// If the residual quantises to all-zero blocks across all six positions
/// — which would correspond to CBP=0 (forbidden by Table 4) — we still
/// skip the MB instead of forcing a CBP.
#[allow(clippy::too_many_arguments)]
pub fn encode_inter_picture(
    fmt: SourceFormat,
    y: &[u8],
    y_stride: usize,
    cb: &[u8],
    cb_stride: usize,
    cr: &[u8],
    cr_stride: usize,
    quant: u32,
    temporal_reference: u8,
    reference: &Picture,
) -> Result<(Vec<u8>, Picture)> {
    validate_inputs(
        fmt,
        y,
        y_stride,
        cb,
        cb_stride,
        cr,
        cr_stride,
        quant,
        temporal_reference,
    )?;
    let (w, h) = fmt.dimensions();
    if reference.width != w as usize || reference.height != h as usize {
        return Err(Error::invalid(format!(
            "h261 encode: reference dims {}x{} mismatch picture {}x{}",
            reference.width, reference.height, w, h
        )));
    }

    let mut recon = Picture::new(w as usize, h as usize);

    let mut bw = BitWriter::with_capacity(8192);
    write_picture_header(&mut bw, fmt, temporal_reference);

    for &gn in fmt.gob_numbers() {
        write_gob_header(&mut bw, gn, quant);
        let (gob_x, gob_y) = gob_origin_luma(fmt, gn);
        encode_gob_inter(
            &mut bw, y, y_stride, cb, cb_stride, cr, cr_stride, gob_x, gob_y, quant, reference,
            &mut recon,
        );
    }

    bw.align_to_byte();
    Ok((bw.finish(), recon))
}

/// Stateful sequence encoder. The first call to [`Self::encode_frame`]
/// emits an INTRA picture; subsequent calls emit P-pictures predicted
/// from the local reconstruction of the previous emitted frame.
pub struct H261Encoder {
    fmt: SourceFormat,
    quant: u32,
    /// Counter for the temporal reference field. Wraps mod 32.
    next_tr: u8,
    /// Local reconstruction of the most recently emitted picture, kept as
    /// the prediction reference for the next P-frame.
    reference: Option<Picture>,
    /// Number of frames between forced INTRA refreshes. 0 = never refresh
    /// after the first I.
    intra_period: u32,
    frames_since_intra: u32,
}

impl H261Encoder {
    /// Build a new encoder for the given source format and quantiser.
    pub fn new(fmt: SourceFormat, quant: u32) -> Self {
        debug_assert!((1..=31).contains(&quant));
        Self {
            fmt,
            quant,
            next_tr: 0,
            reference: None,
            intra_period: 30, // an I-refresh roughly every second at 30 fps
            frames_since_intra: 0,
        }
    }

    /// Override the I-refresh period (number of frames between forced
    /// INTRAs, including the first). `0` disables refresh.
    pub fn with_intra_period(mut self, period: u32) -> Self {
        self.intra_period = period;
        self
    }

    /// Encode one frame from packed YUV 4:2:0 planes. Returns the H.261
    /// elementary-stream bytes for this picture.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_frame(
        &mut self,
        y: &[u8],
        y_stride: usize,
        cb: &[u8],
        cb_stride: usize,
        cr: &[u8],
        cr_stride: usize,
    ) -> Result<Vec<u8>> {
        let force_intra = self.reference.is_none()
            || (self.intra_period != 0 && self.frames_since_intra >= self.intra_period);

        let (bytes, recon) = if force_intra {
            self.frames_since_intra = 1;
            encode_intra_picture_with_recon(
                self.fmt,
                y,
                y_stride,
                cb,
                cb_stride,
                cr,
                cr_stride,
                self.quant,
                self.next_tr,
            )?
        } else {
            self.frames_since_intra += 1;
            let reference = self
                .reference
                .as_ref()
                .expect("reference must exist by now");
            encode_inter_picture(
                self.fmt,
                y,
                y_stride,
                cb,
                cb_stride,
                cr,
                cr_stride,
                self.quant,
                self.next_tr,
                reference,
            )?
        };

        self.reference = Some(recon);
        self.next_tr = self.next_tr.wrapping_add(1) & 0x1F;
        Ok(bytes)
    }
}

/// Common front-door checks used by both intra and inter entry points.
#[allow(clippy::too_many_arguments)]
fn validate_inputs(
    fmt: SourceFormat,
    y: &[u8],
    y_stride: usize,
    cb: &[u8],
    cb_stride: usize,
    cr: &[u8],
    cr_stride: usize,
    quant: u32,
    temporal_reference: u8,
) -> Result<()> {
    if !(1..=31).contains(&quant) {
        return Err(Error::invalid(format!(
            "h261 encode: QUANT out of range: {quant}"
        )));
    }
    if temporal_reference > 31 {
        return Err(Error::invalid(format!(
            "h261 encode: TR out of range: {temporal_reference}"
        )));
    }
    let (_w, h) = fmt.dimensions();
    let h = h as usize;
    if y.len() < y_stride * h || cb.len() < cb_stride * (h / 2) || cr.len() < cr_stride * (h / 2) {
        return Err(Error::invalid("h261 encode: input plane too short"));
    }
    Ok(())
}

/// Emit the 32-bit picture header (§4.2.1).
pub fn write_picture_header(bw: &mut BitWriter, fmt: SourceFormat, tr: u8) {
    bw.write_u32(0x00010, 20); // PSC
    bw.write_u32(tr as u32, 5); // TR
                                // PTYPE — six single-bit flags, MSB first.
                                // bit1 split-screen indicator off
    bw.write_u32(0, 1);
    // bit2 document-camera indicator off
    bw.write_u32(0, 1);
    // bit3 freeze-picture release off
    bw.write_u32(0, 1);
    // bit4 source format
    let fmt_bit = match fmt {
        SourceFormat::Qcif => 0,
        SourceFormat::Cif => 1,
    };
    bw.write_u32(fmt_bit, 1);
    // bit5 HI_RES — "1 = off" (we don't use Annex D).
    bw.write_u32(1, 1);
    // bit6 spare — per §4.1 unused bits are set to 1.
    bw.write_u32(1, 1);
    // PEI = 0 — no PSPARE.
    bw.write_u32(0, 1);
}

/// Emit a GOB header (§4.2.2) with the given GN and GQUANT.
pub fn write_gob_header(bw: &mut BitWriter, gn: u8, gquant: u32) {
    debug_assert!((1..=12).contains(&gn));
    debug_assert!((1..=31).contains(&gquant));
    bw.write_u32(0x0001, 16); // GBSC
    bw.write_u32(gn as u32, 4);
    bw.write_u32(gquant, 5);
    // GEI = 0 — no GSPARE.
    bw.write_u32(0, 1);
}

fn gob_origin_luma(fmt: SourceFormat, gn: u8) -> (usize, usize) {
    match fmt {
        SourceFormat::Cif => crate::gob::cif_gob_origin_luma(gn),
        SourceFormat::Qcif => crate::gob::qcif_gob_origin_luma(gn),
    }
}

/// Encode the 33 INTRA macroblocks of one GOB and write their pel-domain
/// reconstruction into `recon`.
#[allow(clippy::too_many_arguments)]
fn encode_gob_intra(
    bw: &mut BitWriter,
    y: &[u8],
    y_stride: usize,
    cb: &[u8],
    cb_stride: usize,
    cr: &[u8],
    cr_stride: usize,
    gob_x: usize,
    gob_y: usize,
    quant: u32,
    recon: &mut Picture,
) {
    let mut prev_mba: u8 = 0;
    for mba in 1u8..=33 {
        let diff = mba - prev_mba;
        let (bits, code) = encode_mba_diff(diff);
        bw.write_u32(code, bits as u32);
        // MTYPE = INTRA (4-bit 0001). No MQUANT override — we reuse
        // GQUANT for every MB.
        bw.write_u32(MTYPE_INTRA.1, MTYPE_INTRA.0 as u32);

        let mb_col = (mba - 1) as usize % 11;
        let mb_row = (mba - 1) as usize / 11;
        let luma_x = gob_x + mb_col * 16;
        let luma_y = gob_y + mb_row * 16;
        encode_intra_mb_blocks(
            bw, y, y_stride, cb, cb_stride, cr, cr_stride, luma_x, luma_y, quant, recon,
        );

        prev_mba = mba;
    }
}

/// Encode the macroblocks of one GOB as INTER (no MC, residual against the
/// reference). Skipped MBs are not transmitted but their reconstructed
/// pixels (= reference pixels) are still written into `recon`.
#[allow(clippy::too_many_arguments)]
fn encode_gob_inter(
    bw: &mut BitWriter,
    y: &[u8],
    y_stride: usize,
    cb: &[u8],
    cb_stride: usize,
    cr: &[u8],
    cr_stride: usize,
    gob_x: usize,
    gob_y: usize,
    quant: u32,
    reference: &Picture,
    recon: &mut Picture,
) {
    let mut prev_mba: u8 = 0;
    for mba in 1u8..=33 {
        let mb_col = (mba - 1) as usize % 11;
        let mb_row = (mba - 1) as usize / 11;
        let luma_x = gob_x + mb_col * 16;
        let luma_y = gob_y + mb_row * 16;

        // Per-block residual + quantised level arrays. Block order: Y1..Y4, Cb, Cr.
        let mut blocks_pels: [[u8; 64]; 6] = [[0u8; 64]; 6];
        let mut blocks_pred: [[u8; 64]; 6] = [[0u8; 64]; 6];
        // Read predictor (zero MV — ref pels at the same position).
        for (b, (sub_x, sub_y)) in [(0, 0), (8, 0), (0, 8), (8, 8)].iter().enumerate() {
            extract_block(
                y,
                y_stride,
                luma_x + *sub_x,
                luma_y + *sub_y,
                &mut blocks_pels[b],
            );
            extract_block(
                &reference.y,
                reference.y_stride,
                luma_x + *sub_x,
                luma_y + *sub_y,
                &mut blocks_pred[b],
            );
        }
        let cx = luma_x / 2;
        let cy = luma_y / 2;
        extract_block(cb, cb_stride, cx, cy, &mut blocks_pels[4]);
        extract_block(
            &reference.cb,
            reference.c_stride,
            cx,
            cy,
            &mut blocks_pred[4],
        );
        extract_block(cr, cr_stride, cx, cy, &mut blocks_pels[5]);
        extract_block(
            &reference.cr,
            reference.c_stride,
            cx,
            cy,
            &mut blocks_pred[5],
        );

        // Per-block: forward DCT of residual, quantise, decide if any
        // non-zero level remains. Build the CBP mask.
        let mut cbp: u8 = 0;
        // Save the quantised AC + level-0 indicator (stored as zigzag levels)
        // and the reconstructed residual (for local recon).
        let mut q_levels: [[i32; 64]; 6] = [[0i32; 64]; 6];
        let mut recon_blocks: [[u8; 64]; 6] = [[0u8; 64]; 6];

        for b in 0..6 {
            let mut resid = [0i32; 64];
            for i in 0..64 {
                resid[i] = blocks_pels[b][i] as i32 - blocks_pred[b][i] as i32;
            }
            let mut coeffs = [0i32; 64];
            fdct_signed(&resid, &mut coeffs);
            // Inter quantisation — the same scalar quantiser as INTRA AC,
            // applied to all 64 coefficients.
            let mut levels = [0i32; 64];
            let mut any_nonzero = false;
            for i in 0..64 {
                let l = quant_ac(coeffs[i], quant);
                levels[i] = l;
                if l != 0 {
                    any_nonzero = true;
                }
            }
            q_levels[b] = levels;

            if any_nonzero {
                cbp |= 1 << (5 - b);
                // Reconstruct: dequant + IDCT + add predictor, clip.
                let mut dequant = [0i32; 64];
                for i in 0..64 {
                    dequant[i] = crate::block::dequant_ac(levels[i], quant);
                }
                let mut residual = [0i32; 64];
                idct_signed(&dequant, &mut residual);
                for i in 0..64 {
                    let v = blocks_pred[b][i] as i32 + residual[i];
                    recon_blocks[b][i] = v.clamp(0, 255) as u8;
                }
            } else {
                // No residual — reconstruction equals the predictor.
                recon_blocks[b] = blocks_pred[b];
            }
        }

        // Decide: skip or emit?
        if cbp == 0 {
            // All blocks are perfectly predicted (or quantise to zero).
            // Skip the MB — leave it absorbed in the next coded MBA difference.
            // The reconstruction is just the predictor.
            for b in 0..6 {
                write_block_to_picture(recon, b, luma_x, luma_y, &recon_blocks[b]);
            }
            continue;
        }

        // Emit MBA difference (jumps over any preceding skipped MBs).
        let diff = mba - prev_mba;
        let (bits, code) = encode_mba_diff(diff);
        bw.write_u32(code, bits as u32);
        // MTYPE = Inter (1-bit `1`).
        bw.write_u32(MTYPE_INTER.1, MTYPE_INTER.0 as u32);
        // CBP — Table 4. Cannot be 0 here.
        let (cbits, ccode) = encode_cbp(cbp);
        bw.write_u32(ccode, cbits as u32);

        // Emit the coded blocks.
        for b in 0..6 {
            if cbp & (1 << (5 - b)) != 0 {
                emit_inter_block_levels(bw, &q_levels[b]);
            }
            write_block_to_picture(recon, b, luma_x, luma_y, &recon_blocks[b]);
        }

        prev_mba = mba;
    }
}

/// Extract the 8x8 pel block at `(bx, by)` from `plane` and run the
/// forward DCT + per-block intra encode. Also writes the reconstructed
/// block into `recon`.
#[allow(clippy::too_many_arguments)]
fn encode_intra_mb_blocks(
    bw: &mut BitWriter,
    y: &[u8],
    y_stride: usize,
    cb: &[u8],
    cb_stride: usize,
    cr: &[u8],
    cr_stride: usize,
    luma_x: usize,
    luma_y: usize,
    quant: u32,
    recon: &mut Picture,
) {
    // Y1..Y4
    for (b, (sub_x, sub_y)) in [(0, 0), (8, 0), (0, 8), (8, 8)].iter().enumerate() {
        let mut pels = [0u8; 64];
        extract_block(y, y_stride, luma_x + *sub_x, luma_y + *sub_y, &mut pels);
        let mut out = [0u8; 64];
        encode_intra_block(bw, &pels, quant, &mut out);
        write_block_to_picture(recon, b, luma_x, luma_y, &out);
    }
    let cx = luma_x / 2;
    let cy = luma_y / 2;
    let mut cb_pels = [0u8; 64];
    extract_block(cb, cb_stride, cx, cy, &mut cb_pels);
    let mut cb_out = [0u8; 64];
    encode_intra_block(bw, &cb_pels, quant, &mut cb_out);
    write_block_to_picture(recon, 4, luma_x, luma_y, &cb_out);
    let mut cr_pels = [0u8; 64];
    extract_block(cr, cr_stride, cx, cy, &mut cr_pels);
    let mut cr_out = [0u8; 64];
    encode_intra_block(bw, &cr_pels, quant, &mut cr_out);
    write_block_to_picture(recon, 5, luma_x, luma_y, &cr_out);
}

fn extract_block(plane: &[u8], stride: usize, x: usize, y: usize, out: &mut [u8; 64]) {
    for j in 0..8 {
        for i in 0..8 {
            let px = (y + j) * stride + (x + i);
            out[j * 8 + i] = plane.get(px).copied().unwrap_or(0);
        }
    }
}

/// Write a reconstructed 8x8 block into the picture. `block_idx`:
/// 0-3 = Y1..Y4 (Figure 10), 4 = Cb, 5 = Cr.
fn write_block_to_picture(
    pic: &mut Picture,
    block_idx: usize,
    luma_x: usize,
    luma_y: usize,
    out: &[u8; 64],
) {
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

/// Encode one 8x8 intra block: DC (8-bit FLC) + AC (TCOEFF VLCs) + EOB.
/// Also writes the reconstructed pels (decoder-equivalent IDCT of the
/// actually-emitted coefficients) into `recon_out`.
fn encode_intra_block(bw: &mut BitWriter, pels: &[u8; 64], quant: u32, recon_out: &mut [u8; 64]) {
    // Forward DCT.
    let mut coeffs = [0i32; 64];
    fdct_intra(pels, &mut coeffs);

    // DC first: raw-transform DC → FLC per Table 6.
    let dc_code = quant_intra_dc(coeffs[0]);
    bw.write_u32(dc_code as u32, 8);
    // Reconstructed DC level (the value the decoder will see).
    let dc_rec: i32 = if dc_code == 0xFF {
        1024
    } else {
        (dc_code as i32) * 8
    };

    // AC coefficients in zigzag order. quant_ac maps coeff → signed level
    // (in the VLC range -127..=127, with level=0 = dead-zone band).
    let mut zz_levels = [0i32; 63];
    for i in 1..64 {
        zz_levels[i - 1] = quant_ac(coeffs[ZIGZAG[i]], quant);
    }

    // Walk the scan collecting (run, level) pairs.
    let mut run: u32 = 0;
    for &lvl in zz_levels.iter() {
        if lvl == 0 {
            run += 1;
            continue;
        }
        emit_runlevel(bw, run as u8, lvl, /*is_first_inter=*/ false);
        run = 0;
    }
    bw.write_u32(0b10, 2); // EOB

    // Local reconstruction: dequant AC, place at zigzag positions, IDCT.
    let mut rec_coeffs = [0i32; 64];
    rec_coeffs[0] = dc_rec;
    for i in 1..64 {
        rec_coeffs[ZIGZAG[i]] = crate::block::dequant_ac(zz_levels[i - 1], quant);
    }
    idct_intra(&rec_coeffs, recon_out);
}

/// Emit one inter block as TCOEFF VLCs in zigzag order. The first
/// transmitted coefficient uses the `1s` first-coefficient shortcut when
/// `|level| == 1`; subsequent (0,1) pairs use `11s`.
fn emit_inter_block_levels(bw: &mut BitWriter, levels: &[i32; 64]) {
    // Walk in zigzag order.
    let mut run: u32 = 0;
    let mut first = true;
    for i in 0..64 {
        let lvl = levels[ZIGZAG[i]];
        if lvl == 0 {
            run += 1;
            continue;
        }
        emit_runlevel(bw, run as u8, lvl, first);
        first = false;
        run = 0;
    }
    bw.write_u32(0b10, 2); // EOB
}

/// Emit one (run, level) VLC entry. `is_first_inter` selects the special
/// "1s" first-coefficient code for INTER blocks (not used for INTRA;
/// Table 5 note (a): "Never used in INTRA macroblocks").
fn emit_runlevel(bw: &mut BitWriter, run: u8, level: i32, is_first_inter: bool) {
    debug_assert_ne!(level, 0);
    let abs = level.unsigned_abs() as u8;
    let sign = if level < 0 { 1 } else { 0 };

    // Special short code for run=0, abs=1: "1s" if first-in-inter, "11s" otherwise.
    if run == 0 && abs == 1 {
        if is_first_inter {
            bw.write_u32(1, 1); // `1`
        } else {
            bw.write_u32(0b11, 2); // `11`
        }
        bw.write_u32(sign, 1);
        return;
    }

    if let Some((bits, code)) = lookup_tcoeff(run, abs) {
        bw.write_u32(code, bits as u32);
        bw.write_u32(sign, 1);
        return;
    }

    // Fallback: escape — 6-bit prefix `000001`, 6-bit run, 8-bit signed level.
    bw.write_u32(0b0000_01, 6);
    bw.write_u32(run as u32 & 0x3F, 6);
    let enc = if level < 0 {
        (level + 256) as u32
    } else {
        level as u32
    };
    bw.write_u32(enc & 0xFF, 8);
}

#[allow(dead_code)]
fn _unused_refs() {
    let _ = MBA_STUFFING;
    let _ = MTYPE_INTRA_MQUANT;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::{decode_picture_body, pic_to_video_frame, H261Decoder};
    use crate::picture::parse_picture_header;
    use oxideav_core::bits::BitReader;
    use oxideav_core::packet::PacketFlags;
    use oxideav_core::Decoder;
    use oxideav_core::{CodecId, Frame, Packet, TimeBase};

    fn neutral_qcif() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y = vec![128u8; 176 * 144];
        let cb = vec![128u8; 88 * 72];
        let cr = vec![128u8; 88 * 72];
        (y, cb, cr)
    }

    fn gradient_qcif() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let w = 176usize;
        let h = 144usize;
        let mut y = vec![0u8; w * h];
        for j in 0..h {
            for i in 0..w {
                y[j * w + i] = (32 + (i * 192) / w) as u8;
            }
        }
        let cb = vec![128u8; (w / 2) * (h / 2)];
        let cr = vec![128u8; (w / 2) * (h / 2)];
        (y, cb, cr)
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        if a.is_empty() {
            return f64::INFINITY;
        }
        let mut sse = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = *x as f64 - *y as f64;
            sse += d * d;
        }
        let mse = sse / a.len() as f64;
        if mse <= 0.0 {
            return f64::INFINITY;
        }
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }

    fn decode_one(bytes: Vec<u8>) -> oxideav_core::VideoFrame {
        let codec_id = CodecId::new(crate::CODEC_ID_STR);
        let mut decoder = H261Decoder::new(codec_id);
        let pkt = Packet {
            stream_index: 0,
            data: bytes,
            pts: Some(0),
            dts: Some(0),
            duration: None,
            time_base: TimeBase::new(1, 30_000),
            flags: PacketFlags {
                keyframe: true,
                ..Default::default()
            },
        };
        decoder.send_packet(&pkt).expect("send");
        decoder.flush().ok();
        match decoder.receive_frame().expect("frame") {
            Frame::Video(v) => v,
            _ => panic!("video"),
        }
    }

    #[test]
    fn picture_header_roundtrip() {
        let mut bw = BitWriter::new();
        write_picture_header(&mut bw, SourceFormat::Qcif, 7);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let hdr = parse_picture_header(&mut br).expect("parse");
        assert_eq!(hdr.temporal_reference, 7);
        assert_eq!(hdr.source_format, SourceFormat::Qcif);
        assert_eq!(hdr.width, 176);
        assert_eq!(hdr.height, 144);
    }

    #[test]
    fn gob_header_roundtrip() {
        let mut bw = BitWriter::new();
        write_gob_header(&mut bw, 3, 8);
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        let hdr = crate::gob::parse_gob_header(&mut br).expect("parse GOB");
        assert_eq!(hdr.gn, 3);
        assert_eq!(hdr.gquant, 8);
    }

    #[test]
    fn encode_qcif_grey_roundtrips_through_our_decoder() {
        let (y, cb, cr) = neutral_qcif();
        let bytes = encode_intra_picture(SourceFormat::Qcif, &y, 176, &cb, 88, &cr, 88, 8, 0)
            .expect("encode");
        assert!(!bytes.is_empty());
        let vf = decode_one(bytes);
        assert_eq!(vf.width, 176);
        assert_eq!(vf.height, 144);
        let y_plane = &vf.planes[0].data;
        let mut max_err = 0i32;
        for &p in y_plane {
            max_err = max_err.max((p as i32 - 128).abs());
        }
        assert!(max_err <= 2, "max Y error was {max_err}");
        for &p in &vf.planes[1].data {
            assert!((p as i32 - 128).abs() <= 2);
        }
        for &p in &vf.planes[2].data {
            assert!((p as i32 - 128).abs() <= 2);
        }
    }

    #[test]
    fn encode_cif_grey_roundtrips() {
        let y = vec![128u8; 352 * 288];
        let cb = vec![128u8; 176 * 144];
        let cr = vec![128u8; 176 * 144];
        let bytes = encode_intra_picture(SourceFormat::Cif, &y, 352, &cb, 176, &cr, 176, 8, 0)
            .expect("encode cif");
        assert!(!bytes.is_empty());

        let mut br = BitReader::new(&bytes);
        let hdr = parse_picture_header(&mut br).expect("pic header");
        let pic = decode_picture_body(&mut br, &hdr, &bytes, None).expect("body");
        let vf = pic_to_video_frame(&pic, Some(0), TimeBase::new(1, 30_000));
        assert_eq!(vf.width, 352);
        assert_eq!(vf.height, 288);
        for &p in &vf.planes[0].data {
            assert!((p as i32 - 128).abs() <= 2, "Y pel {p} too far from 128");
        }
    }

    #[test]
    fn encode_qcif_gradient_plausible_decode() {
        let (y, cb, cr) = gradient_qcif();
        let bytes = encode_intra_picture(SourceFormat::Qcif, &y, 176, &cb, 88, &cr, 88, 8, 0)
            .expect("encode gradient");
        let vf = decode_one(bytes);
        let yp = &vf.planes[0].data;
        let w = 176usize;
        let sample = |x: usize, yy: usize| yp[yy * w + x] as i32;
        let expected = |x: usize| 32 + (x * 192) as i32 / w as i32;
        for &x in &[24usize, 80, 152] {
            let got = sample(x, 72);
            let want = expected(x);
            assert!(
                (got - want).abs() <= 40,
                "gradient at x={x}: got {got}, want ~{want}"
            );
        }
    }

    /// Encoder local-recon should match what the decoder produces when
    /// fed our own bitstream. This is the contract that lets
    /// `H261Encoder` chain P-frames safely.
    #[test]
    fn intra_local_recon_matches_decoder() {
        let (y, cb, cr) = gradient_qcif();
        let (bytes, recon) =
            encode_intra_picture_with_recon(SourceFormat::Qcif, &y, 176, &cb, 88, &cr, 88, 8, 0)
                .expect("encode");
        let vf = decode_one(bytes);
        // Compare Y plane pel-by-pel.
        let dy = &vf.planes[0].data;
        let dcb = &vf.planes[1].data;
        let dcr = &vf.planes[2].data;
        let w = 176usize;
        let h = 144usize;
        let mut diff_count = 0;
        for j in 0..h {
            for i in 0..w {
                let a = recon.y[j * recon.y_stride + i] as i32;
                let b = dy[j * w + i] as i32;
                if a != b {
                    diff_count += 1;
                }
            }
        }
        assert_eq!(diff_count, 0, "intra recon Y mismatch in {diff_count} pels");
        for j in 0..(h / 2) {
            for i in 0..(w / 2) {
                let a = recon.cb[j * recon.c_stride + i];
                let b = dcb[j * (w / 2) + i];
                assert_eq!(a, b, "Cb mismatch at ({i},{j})");
                let a2 = recon.cr[j * recon.c_stride + i];
                let b2 = dcr[j * (w / 2) + i];
                assert_eq!(a2, b2, "Cr mismatch at ({i},{j})");
            }
        }
    }

    /// Stateful sequence encoder should emit an I-picture for the first
    /// frame and P-pictures thereafter, with each decoded frame matching
    /// the input within an acceptable PSNR.
    #[test]
    fn sequence_ipi_qcif_roundtrip() {
        let (y0, cb0, cr0) = gradient_qcif();
        // Frame 1: same content (should be all-skip P-MBs after quantisation).
        let (y1, cb1, cr1) = (y0.clone(), cb0.clone(), cr0.clone());

        let mut enc = H261Encoder::new(SourceFormat::Qcif, 8);
        let pkt0 = enc
            .encode_frame(&y0, 176, &cb0, 88, &cr0, 88)
            .expect("frame 0");
        let pkt1 = enc
            .encode_frame(&y1, 176, &cb1, 88, &cr1, 88)
            .expect("frame 1");

        // Concatenate both pictures into one stream and decode.
        let mut stream = Vec::new();
        stream.extend_from_slice(&pkt0);
        stream.extend_from_slice(&pkt1);

        let codec_id = CodecId::new(crate::CODEC_ID_STR);
        let mut decoder = H261Decoder::new(codec_id);
        let pkt = Packet {
            stream_index: 0,
            data: stream,
            pts: Some(0),
            dts: Some(0),
            duration: None,
            time_base: TimeBase::new(1, 30_000),
            flags: PacketFlags {
                keyframe: true,
                ..Default::default()
            },
        };
        decoder.send_packet(&pkt).expect("send");
        decoder.flush().ok();

        let f0 = match decoder.receive_frame().expect("f0") {
            Frame::Video(v) => v,
            _ => panic!("video"),
        };
        let f1 = match decoder.receive_frame().expect("f1") {
            Frame::Video(v) => v,
            _ => panic!("video"),
        };
        // Both frames should be present. Compare each Y plane against its
        // respective input using PSNR.
        let y_size = 176 * 144;
        let p0 = psnr(&f0.planes[0].data, &y0);
        let p1 = psnr(&f1.planes[0].data, &y1);
        assert!(p0 >= 28.0, "I-frame Y PSNR too low: {p0:.2} dB");
        assert!(p1 >= 28.0, "P-frame Y PSNR too low: {p1:.2} dB");
        // P-frame should be very small relative to I-frame (mostly skipped MBs)
        assert!(
            pkt1.len() < pkt0.len() / 2,
            "expected P-frame ({}) to be much smaller than I-frame ({})",
            pkt1.len(),
            pkt0.len()
        );
        // Sanity: avoid unused.
        let _ = y_size;
    }

    /// P-picture against its own intra-encoded reference must be byte-tight
    /// (mostly skips) and roundtrip cleanly.
    #[test]
    fn inter_picture_self_predict_is_mostly_skips() {
        let (y, cb, cr) = gradient_qcif();
        let (_iframe, recon) =
            encode_intra_picture_with_recon(SourceFormat::Qcif, &y, 176, &cb, 88, &cr, 88, 8, 0)
                .expect("intra");
        // Re-encode the same source (= recon may differ a little due to
        // quantisation) as P-frame against `recon`.
        let (pframe, _new_recon) =
            encode_inter_picture(SourceFormat::Qcif, &y, 176, &cb, 88, &cr, 88, 8, 1, &recon)
                .expect("inter");
        // Expectation: size is dominated by 3 GOB headers (~24 bits each)
        // plus minimal overhead — should be well under 100 bytes.
        assert!(
            pframe.len() < 100,
            "self-predict P-frame too big: {} bytes",
            pframe.len()
        );
    }

    /// Modify a small region between the two frames and verify the P-picture
    /// adapts (carries some non-zero residual) and that the decode matches
    /// the second source within acceptable PSNR.
    #[test]
    fn inter_picture_adapts_to_change() {
        let (y0, cb0, cr0) = gradient_qcif();
        // Frame 1: shift the gradient by adding a constant offset to a
        // small rectangle.
        let mut y1 = y0.clone();
        for j in 32..64 {
            for i in 32..96 {
                y1[j * 176 + i] = y1[j * 176 + i].saturating_add(32);
            }
        }

        let mut enc = H261Encoder::new(SourceFormat::Qcif, 8);
        let pkt0 = enc.encode_frame(&y0, 176, &cb0, 88, &cr0, 88).expect("f0");
        let pkt1 = enc.encode_frame(&y1, 176, &cb0, 88, &cr0, 88).expect("f1");

        let mut stream = Vec::new();
        stream.extend_from_slice(&pkt0);
        stream.extend_from_slice(&pkt1);
        let mut decoder = H261Decoder::new(CodecId::new(crate::CODEC_ID_STR));
        let pkt = Packet {
            stream_index: 0,
            data: stream,
            pts: Some(0),
            dts: Some(0),
            duration: None,
            time_base: TimeBase::new(1, 30_000),
            flags: PacketFlags {
                keyframe: true,
                ..Default::default()
            },
        };
        decoder.send_packet(&pkt).expect("send");
        decoder.flush().ok();
        let _f0 = decoder.receive_frame().expect("f0");
        let f1 = match decoder.receive_frame().expect("f1") {
            Frame::Video(v) => v,
            _ => panic!("video"),
        };
        let p1 = psnr(&f1.planes[0].data, &y1);
        assert!(p1 >= 26.0, "P-frame adapted Y PSNR too low: {p1:.2} dB");
    }
}
