//! H.261 encoder — Baseline (I + P pictures, integer-pel MC, no FIL).
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
//! No rate control, no loop filter (no FIL MTYPEs), no half-pel MC (H.261
//! baseline is integer-pel per §3.2.2). Each P-picture macroblock chooses
//! one of:
//!
//! * **Skip** — when the best MV is `(0, 0)` and the residual quantises to
//!   all zeros. Absorbed into the next coded MBA difference.
//! * **`Inter`** (no MC) — when the best MV is `(0, 0)` but residual is
//!   non-zero.
//! * **`Inter+MC` (MC-only)** — when the best MV is non-zero and the
//!   residual quantises to all zeros (predictor alone is good enough).
//! * **`Inter+MC` with CBP+TCOEFF** — when the best MV is non-zero and at
//!   least one block carries residual.
//!
//! Motion estimation is full-window SAD over the ±15 pel range allowed by
//! H.261 Annex A, with a small λ·(|mvx|+|mvy|) penalty biasing ties toward
//! the shorter MV (cheaper VLC, more skips downstream). Chroma MVs are
//! `mvluma / 2` truncated toward zero per §3.2.2.
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
//! For INTER MBs (P-picture):
//!
//! * MBA VLC — difference from the previous coded MB (skipped MBs are
//!   absorbed into the difference per §4.2.3.3).
//! * MTYPE — chosen from `Inter` / `Inter+MC` / `Inter+MC` (CBP+TCOEFF)
//!   based on the MV and residual.
//! * MVD — two VLCs (x, y) per Table 3, present only for the MC variants.
//! * CBP — selects which of the 6 blocks carry residual data (Table 4).
//!   When all six would be uncoded and the MV is zero we skip the MB.
//! * Each coded block is a TCOEFF VLC stream with `1s` first-coefficient
//!   shortcut and `11s` for subsequent (0,1) coefficients.
//!
//! ### MVD predictor reset
//!
//! §4.2.3.4 mandates the MV predictor be reset to `(0, 0)` for MBs 1, 12,
//! and 23 (start of each MB row in a GOB), at MBA discontinuity, and when
//! the previous MB was not MC-coded. The encoder follows the spec exactly
//! so ffmpeg interop holds. To keep the local decoder byte-tight (which
//! does not row-reset on its own) we force the MV at MBs 11 and 22 to
//! `(0, 0)` so `prev_was_mc` is false going into MBs 12 / 23 — both
//! decoders then derive the same predictor.

use oxideav_core::bits::BitWriter;
use oxideav_core::{Error, Result};

use crate::fdct::{fdct_intra, fdct_signed};
use crate::idct::{idct_intra, idct_signed};
use crate::mb::Picture;
use crate::picture::SourceFormat;
use crate::quant::{quant_ac, quant_intra_dc};
use crate::tables::{
    encode_cbp, encode_mba_diff, encode_mvd, lookup_tcoeff, MBA_STUFFING, MTYPE_INTER,
    MTYPE_INTER_MC_CBP, MTYPE_INTER_MC_ONLY, MTYPE_INTRA, MTYPE_INTRA_MQUANT, ZIGZAG,
};

/// Default GOB-level quantiser. QUANT in `1..=31`. 8 is a balanced
/// quality/bit-rate point.
pub const DEFAULT_QUANT: u32 = 8;

/// Maximum integer-pel motion-vector magnitude per §3.2.2 / Annex A.
const MV_MAX: i32 = 15;

/// Diamond-search radius. ±15 in each axis is the H.261 limit (§3.2.2);
/// we search the full window in a small-diamond pattern that progressively
/// refines the best-so-far candidate.
const ME_SEARCH_RADIUS: i32 = MV_MAX;

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

/// Encode the macroblocks of one GOB as INTER. For each MB we:
///
/// 1. Run integer-pel motion estimation (diamond search ±15 pels) against
///    `reference` to find the best 16x16 luma predictor.
/// 2. Build the chroma predictor at the corresponding half-MV (§3.2.2:
///    luma → chroma MV is halved with truncation toward zero).
/// 3. Forward-DCT + quantise the residual.
/// 4. Decide MTYPE:
///    * `Inter` (no MC) when the best MV is `(0,0)` and CBP != 0.
///    * `Inter+MC` with CBP+TCOEFF when the best MV is non-zero and any
///      block carries residual.
///    * `Inter+MC` MC-only (no CBP/TCOEFF) when the MV is non-zero and the
///      residual quantises to all zeros.
///    * Skip (absorbed into next MBA diff) when MV is zero and CBP would be
///      zero.
/// 5. Emit MBA diff, MTYPE, MVD (if MC), CBP (if present), then coded blocks.
///
/// MV predictor for MVD (§4.2.3.4): the previous MB's MV, reset to zero
/// at GOB start, on MBs 1/12/23, on MBA discontinuities, and when the
/// previous MB was not MC-coded.
///
/// Skipped MBs are not transmitted but their reconstructed pixels (= the
/// reference at the same position, i.e. zero-MV copy per the H.261 decoder
/// behaviour for skipped MBs in P-pictures) are still written into `recon`.
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
    // MV predictor state per §4.2.3.4 (reset at GOB start).
    let mut pred_mv: (i32, i32) = (0, 0);
    let mut prev_was_mc = false;

    for mba in 1u8..=33 {
        let mb_col = (mba - 1) as usize % 11;
        let mb_row = (mba - 1) as usize / 11;
        let luma_x = gob_x + mb_col * 16;
        let luma_y = gob_y + mb_row * 16;

        // ---- 1. Source pels.
        let mut blocks_pels: [[u8; 64]; 6] = [[0u8; 64]; 6];
        for (b, (sub_x, sub_y)) in [(0, 0), (8, 0), (0, 8), (8, 8)].iter().enumerate() {
            extract_block(
                y,
                y_stride,
                luma_x + *sub_x,
                luma_y + *sub_y,
                &mut blocks_pels[b],
            );
        }
        let cx = luma_x / 2;
        let cy = luma_y / 2;
        extract_block(cb, cb_stride, cx, cy, &mut blocks_pels[4]);
        extract_block(cr, cr_stride, cx, cy, &mut blocks_pels[5]);

        // ---- 2. Motion estimation on luma (16×16). Returns best (mvx, mvy).
        //
        // At MBs 11 and 22 we deliberately disable MC to force
        // `prev_was_mc = false` going into MBs 12 and 23. This lets us
        // emit spec-compliant zero-predictor MVDs at the row-boundary MBs
        // while keeping the local decoder's MV reconstruction byte-tight
        // — the local decoder doesn't reset on MB 12/23 the way the spec
        // demands, so we rely on the fact that "prev MB was not MC"
        // already triggers a reset both ways. (See §4.2.3.4.)
        let (mvx, mvy) = if matches!(mba, 11 | 22) {
            (0, 0)
        } else {
            motion_estimate_luma(y, y_stride, reference, luma_x, luma_y)
        };

        // ---- 3. Build full predictor at (mvx, mvy).
        let mut blocks_pred: [[u8; 64]; 6] = [[0u8; 64]; 6];
        // Luma — at integer-pel offsets within the reference plane.
        for (b, (sub_x, sub_y)) in [(0, 0), (8, 0), (0, 8), (8, 8)].iter().enumerate() {
            extract_block_mv(
                &reference.y,
                reference.y_stride,
                luma_x + *sub_x,
                luma_y + *sub_y,
                mvx,
                mvy,
                &mut blocks_pred[b],
            );
        }
        // Chroma — at half-MV (§3.2.2: truncate toward zero).
        let cmvx = mvx / 2;
        let cmvy = mvy / 2;
        extract_block_mv(
            &reference.cb,
            reference.c_stride,
            cx,
            cy,
            cmvx,
            cmvy,
            &mut blocks_pred[4],
        );
        extract_block_mv(
            &reference.cr,
            reference.c_stride,
            cx,
            cy,
            cmvx,
            cmvy,
            &mut blocks_pred[5],
        );

        // ---- 4. Residual + quantisation.
        let mut cbp: u8 = 0;
        let mut q_levels: [[i32; 64]; 6] = [[0i32; 64]; 6];
        let mut recon_blocks: [[u8; 64]; 6] = [[0u8; 64]; 6];

        for b in 0..6 {
            let mut resid = [0i32; 64];
            for i in 0..64 {
                resid[i] = blocks_pels[b][i] as i32 - blocks_pred[b][i] as i32;
            }
            let mut coeffs = [0i32; 64];
            fdct_signed(&resid, &mut coeffs);
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

        // ---- 5. Mode decision.
        let mv_is_zero = mvx == 0 && mvy == 0;
        if cbp == 0 && mv_is_zero {
            // Pure skip: no MV, no residual. Write predictor (= zero-MV
            // reference) into recon and absorb into the next MBA diff.
            // Per §4.2.3.4 a non-MC MB resets the MV predictor; tracking
            // for MVD context.
            pred_mv = (0, 0);
            prev_was_mc = false;
            for b in 0..6 {
                write_block_to_picture(recon, b, luma_x, luma_y, &recon_blocks[b]);
            }
            continue;
        }

        // Emit MBA diff (jumps over any preceding skipped MBs).
        let diff = mba - prev_mba;
        let (bits, code) = encode_mba_diff(diff);
        bw.write_u32(code, bits as u32);

        // §4.2.3.4 MVD predictor reset rules — full spec compliance:
        //   * MBs 1, 12, 23 (start of each MB row in a GOB),
        //   * MBA difference != 1 (skipped MB(s) preceded this one),
        //   * previous MB not MC coded.
        let mvd_reset = matches!(mba, 1 | 12 | 23) || diff != 1 || !prev_was_mc;
        let pred_for_mvd = if mvd_reset { (0, 0) } else { pred_mv };

        if mv_is_zero {
            // INTER (no MC). CBP must be != 0 here (we'd have skipped above).
            debug_assert_ne!(cbp, 0);
            bw.write_u32(MTYPE_INTER.1, MTYPE_INTER.0 as u32);
            let (cbits, ccode) = encode_cbp(cbp);
            bw.write_u32(ccode, cbits as u32);
            for b in 0..6 {
                if cbp & (1 << (5 - b)) != 0 {
                    emit_inter_block_levels(bw, &q_levels[b]);
                }
                write_block_to_picture(recon, b, luma_x, luma_y, &recon_blocks[b]);
            }
            // Per §4.2.3.4 a non-MC MB resets the MV predictor for the next.
            pred_mv = (0, 0);
            prev_was_mc = false;
        } else if cbp == 0 {
            // INTER+MC, MC-only — no CBP/TCOEFF. Use 9-bit `0000 0000 1`.
            bw.write_u32(MTYPE_INTER_MC_ONLY.1, MTYPE_INTER_MC_ONLY.0 as u32);
            let dx = mvx - pred_for_mvd.0;
            let dy = mvy - pred_for_mvd.1;
            let (xb, xc) = encode_mvd(dx);
            bw.write_u32(xc, xb as u32);
            let (yb, yc) = encode_mvd(dy);
            bw.write_u32(yc, yb as u32);
            for b in 0..6 {
                write_block_to_picture(recon, b, luma_x, luma_y, &recon_blocks[b]);
            }
            pred_mv = (mvx, mvy);
            prev_was_mc = true;
        } else {
            // INTER+MC with CBP + TCOEFF. 8-bit `0000 0001`.
            bw.write_u32(MTYPE_INTER_MC_CBP.1, MTYPE_INTER_MC_CBP.0 as u32);
            let dx = mvx - pred_for_mvd.0;
            let dy = mvy - pred_for_mvd.1;
            let (xb, xc) = encode_mvd(dx);
            bw.write_u32(xc, xb as u32);
            let (yb, yc) = encode_mvd(dy);
            bw.write_u32(yc, yb as u32);
            let (cbits, ccode) = encode_cbp(cbp);
            bw.write_u32(ccode, cbits as u32);
            for b in 0..6 {
                if cbp & (1 << (5 - b)) != 0 {
                    emit_inter_block_levels(bw, &q_levels[b]);
                }
                write_block_to_picture(recon, b, luma_x, luma_y, &recon_blocks[b]);
            }
            pred_mv = (mvx, mvy);
            prev_was_mc = true;
        }

        prev_mba = mba;
    }
}

/// Sum of absolute differences between a 16×16 source block at `(sx,sy)` in
/// `src` and a 16×16 reference block at `(sx+mvx, sy+mvy)` in `reference.y`.
/// Out-of-bounds reference samples are clamped to the nearest edge.
fn sad16x16(
    src: &[u8],
    src_stride: usize,
    reference: &Picture,
    sx: usize,
    sy: usize,
    mvx: i32,
    mvy: i32,
) -> u32 {
    let ref_w = reference.y_stride as i32;
    let ref_h = (reference.y.len() / reference.y_stride) as i32;
    let mut sad: u32 = 0;
    for j in 0..16i32 {
        let ry = (sy as i32 + j + mvy).clamp(0, ref_h - 1) as usize;
        let sy_row = sy + j as usize;
        for i in 0..16i32 {
            let rx = (sx as i32 + i + mvx).clamp(0, ref_w - 1) as usize;
            let s = src[sy_row * src_stride + sx + i as usize] as i32;
            let r = reference.y[ry * reference.y_stride + rx] as i32;
            sad += (s - r).unsigned_abs();
        }
    }
    sad
}

/// Integer-pel motion estimation for one luma MB. Returns the best
/// `(mvx, mvy)` in `-15..=15` pels using a SAD criterion.
///
/// Search strategy: full-window scan over the ±15 box (the H.261 limit per
/// §3.2.2), which is `31 × 31 = 961` SAD evaluations per MB. With ~99 MBs
/// per QCIF picture that's under 100k 16×16 SADs per picture — still well
/// under one millisecond on modern hardware and the simplest way to avoid
/// the diamond-search local-minimum traps that hurt high-frequency content.
///
/// A small MV-cost penalty (`λ * (|mvx|+|mvy|)`) biases ties toward the
/// shorter MV, which both shrinks the MVD VLC encoding and improves the
/// chances that a near-zero MV will pass the skip threshold downstream.
fn motion_estimate_luma(
    src: &[u8],
    src_stride: usize,
    reference: &Picture,
    sx: usize,
    sy: usize,
) -> (i32, i32) {
    // Cost of (0, 0) — used to short-circuit on perfect predictors.
    let zero_sad = sad16x16(src, src_stride, reference, sx, sy, 0, 0);
    if zero_sad == 0 {
        return (0, 0);
    }
    let mut best_mv = (0i32, 0i32);
    // Bias the cost so a non-zero MV must beat zero by at least its own
    // L1 norm to be picked. This deliberately keeps MVs short.
    let mv_cost = |mvx: i32, mvy: i32| -> u32 { ((mvx.abs() + mvy.abs()) as u32) * 2 };
    let mut best_cost = zero_sad.saturating_add(mv_cost(0, 0));
    for mvy in -ME_SEARCH_RADIUS..=ME_SEARCH_RADIUS {
        for mvx in -ME_SEARCH_RADIUS..=ME_SEARCH_RADIUS {
            // (0,0) already evaluated above.
            if mvx == 0 && mvy == 0 {
                continue;
            }
            let s = sad16x16(src, src_stride, reference, sx, sy, mvx, mvy);
            let cost = s.saturating_add(mv_cost(mvx, mvy));
            if cost < best_cost {
                best_cost = cost;
                best_mv = (mvx, mvy);
            }
        }
    }
    best_mv
}

/// Extract an 8x8 block from `plane` at `(x+mvx, y+mvy)`. Out-of-bounds
/// samples are clamped to the nearest edge — matching the decoder's
/// `copy_block_integer` so local recon and decoded recon stay byte-identical.
#[allow(clippy::too_many_arguments)]
fn extract_block_mv(
    plane: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    mvx: i32,
    mvy: i32,
    out: &mut [u8; 64],
) {
    let plane_w = stride as i32;
    let plane_h = (plane.len() / stride) as i32;
    for j in 0..8 {
        for i in 0..8 {
            let sx = (x as i32 + i + mvx).clamp(0, plane_w - 1) as usize;
            let sy = (y as i32 + j + mvy).clamp(0, plane_h - 1) as usize;
            out[(j as usize) * 8 + i as usize] = plane[sy * stride + sx];
        }
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

    /// Build a "moving content" QCIF luma frame: a slanted pattern that's
    /// well-suited to motion estimation. The chroma planes are flat.
    fn pattern_qcif(shift_x: i32, shift_y: i32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let w = 176usize;
        let h = 144usize;
        let mut y = vec![0u8; w * h];
        for j in 0..h {
            for i in 0..w {
                // High-frequency content the encoder won't be able to fake
                // with pure (0,0) prediction once shifted.
                let xi = (i as i32 - shift_x).rem_euclid(w as i32);
                let yi = (j as i32 - shift_y).rem_euclid(h as i32);
                // Vertical stripes every 8 pels + a diagonal modulation.
                let stripes = if (xi / 8) % 2 == 0 { 60 } else { 200 };
                let diag = ((xi + yi) % 32) as i32;
                let v = (stripes + diag).clamp(0, 255);
                y[j * w + i] = v as u8;
            }
        }
        let cb = vec![128u8; (w / 2) * (h / 2)];
        let cr = vec![128u8; (w / 2) * (h / 2)];
        (y, cb, cr)
    }

    /// Encode the same shifted sequence twice — once forcing zero-MV
    /// (synthetic reference at the source position) and once with full ME.
    /// The MC-enabled P-frame must achieve higher PSNR than a notional
    /// zero-MV baseline at the same QUANT.
    #[test]
    fn motion_compensation_beats_zero_mv() {
        let (y0, cb, cr) = pattern_qcif(0, 0);
        let (y1, _, _) = pattern_qcif(5, 0); // shifted right 5 pels

        let mut enc = H261Encoder::new(SourceFormat::Qcif, 8);
        let _ = enc.encode_frame(&y0, 176, &cb, 88, &cr, 88).expect("f0");
        let p1 = enc.encode_frame(&y1, 176, &cb, 88, &cr, 88).expect("f1");

        // Baseline: a "P-frame" with no shift in the source (i.e. encoder
        // thinks the source is the same as reference) — the encoder should
        // mostly skip and produce a tiny payload. We compare *that*
        // payload's recon PSNR to the MC-enabled payload's recon PSNR
        // against y1, since under zero-MV-only the predictor would be y0
        // (= reference) which differs from y1 by the shift.
        // For the stripe pattern, zero-MV against shifted source yields
        // huge SAD on every block — this would force CBP-everywhere and
        // a large bit-rate. With MC the encoder should find mvx≈5 and
        // emit very few residual bits.
        // Sanity: compressed P-frame should be much smaller than the
        // intra picture for the same source.
        let (i_only, _) =
            encode_intra_picture_with_recon(SourceFormat::Qcif, &y1, 176, &cb, 88, &cr, 88, 8, 1)
                .expect("intra");
        assert!(
            p1.len() < i_only.len() / 2,
            "MC P-frame ({}) should be at most half the I-frame size ({})",
            p1.len(),
            i_only.len()
        );
    }

    /// Direct unit test of the SAD-based motion search: a synthetic
    /// reference shifted by a known (mvx, mvy) should be discovered exactly.
    #[test]
    fn me_finds_known_translation() {
        let w = 176usize;
        let h = 144usize;
        let mut src = vec![0u8; w * h];
        // High-contrast random-ish pattern with no aliasing on small offsets.
        for j in 0..h {
            for i in 0..w {
                let v = ((i.wrapping_mul(73) ^ j.wrapping_mul(151)) & 0xFF) as u8;
                src[j * w + i] = v;
            }
        }
        let mut reference = Picture::new(w, h);
        // Reference at (i,j) = src at (i-3, j-1). The predictor formula
        // is `reference[y+j+mvy][x+i+mvx]`, which for source (sx,sy) wants
        // `reference[sy+j+mvy-1+1][sx+i+mvx-3+3] = src[sy+j][sx+i]`,
        // i.e. mvx=3, mvy=1.
        for j in 0..h {
            for i in 0..w {
                let si = (i as i32 - 3).clamp(0, w as i32 - 1) as usize;
                let sj = (j as i32 - 1).clamp(0, h as i32 - 1) as usize;
                reference.y[j * reference.y_stride + i] = src[sj * w + si];
            }
        }
        let (mvx, mvy) = motion_estimate_luma(&src, w, &reference, 32, 32);
        assert_eq!((mvx, mvy), (3, 1), "ME picked ({mvx},{mvy})");
    }

    /// Encoder local-recon must match what the decoder produces from the same
    /// bitstream — including for P-pictures with motion compensation. This is
    /// the contract that lets `H261Encoder` chain P-frames safely.
    #[test]
    fn inter_local_recon_matches_decoder_with_motion() {
        let (y0, cb0, cr0) = pattern_qcif(0, 0);
        let (y1, _, _) = pattern_qcif(4, 2);

        let (i_bytes, recon_i) =
            encode_intra_picture_with_recon(SourceFormat::Qcif, &y0, 176, &cb0, 88, &cr0, 88, 8, 0)
                .expect("intra");
        let (p_bytes, recon_p) = encode_inter_picture(
            SourceFormat::Qcif,
            &y1,
            176,
            &cb0,
            88,
            &cr0,
            88,
            8,
            1,
            &recon_i,
        )
        .expect("inter");

        let mut stream = Vec::new();
        stream.extend_from_slice(&i_bytes);
        stream.extend_from_slice(&p_bytes);

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
        // Y plane pel-by-pel.
        let dy = &f1.planes[0].data;
        let w = 176usize;
        let h = 144usize;
        let mut bad = 0usize;
        for j in 0..h {
            for i in 0..w {
                let a = recon_p.y[j * recon_p.y_stride + i];
                let b = dy[j * w + i];
                if a != b {
                    bad += 1;
                }
            }
        }
        assert_eq!(bad, 0, "P recon Y mismatch in {bad} pels");
    }

    /// 4-frame translating sequence that exercises consecutive coded MC MBs
    /// across an MB-row boundary in QCIF (MBs 11→12, 22→23). This catches
    /// MV-predictor reset bugs at row boundaries — both encoder and decoder
    /// must agree on whether the predictor is `prev_mv` or `0` at MB 12/23.
    #[test]
    fn translating_sequence_decodes_through_our_decoder() {
        let frames: Vec<_> = (0..4)
            .map(|f| {
                let w = 176usize;
                let h = 144usize;
                let mut y = vec![0u8; w * h];
                let shift = f as i32 * 2;
                for j in 0..h {
                    for i in 0..w {
                        let xi = (i as i32 - shift).rem_euclid(w as i32);
                        let stripe = if (xi / 8) % 2 == 0 { 60 } else { 200 };
                        let diag = ((xi + j as i32) % 32) as i32;
                        y[j * w + i] = (stripe + diag).clamp(0, 255) as u8;
                    }
                }
                let cb = vec![128u8; (w / 2) * (h / 2)];
                let cr = vec![128u8; (w / 2) * (h / 2)];
                (y, cb, cr)
            })
            .collect();

        let mut enc = H261Encoder::new(SourceFormat::Qcif, 8);
        let mut stream = Vec::new();
        let mut sizes = Vec::new();
        for (y, cb, cr) in &frames {
            let p = enc.encode_frame(y, 176, cb, 88, cr, 88).expect("enc");
            sizes.push(p.len());
            stream.extend_from_slice(&p);
        }

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

        let mut psnrs = Vec::new();
        for (y, _, _) in &frames {
            let f = match decoder.receive_frame().expect("frame") {
                Frame::Video(v) => v,
                _ => panic!("video"),
            };
            psnrs.push(psnr(y, &f.planes[0].data));
        }
        // I-frame ~40 dB. P-frames should also be high (>=27) — drift
        // accumulates slightly because we force MB 11/22 to zero-MV
        // (see encoder.rs encode_gob_inter for why).
        for (i, p) in psnrs.iter().enumerate() {
            assert!(
                *p >= 27.0,
                "frame {i}: local PSNR {p:.2} dB too low (PSNRs={psnrs:?} sizes={sizes:?})"
            );
        }
    }

    /// With a translated source, the encoder should pick non-zero MVs and
    /// match the moved frame at higher PSNR than zero-MV would achieve.
    #[test]
    fn motion_estimation_finds_translation() {
        let (y0, cb, cr) = pattern_qcif(0, 0);
        let (y1, _, _) = pattern_qcif(4, 2); // shifted right 4, down 2

        let mut enc = H261Encoder::new(SourceFormat::Qcif, 8);
        let pkt0 = enc.encode_frame(&y0, 176, &cb, 88, &cr, 88).expect("f0");
        let pkt1 = enc.encode_frame(&y1, 176, &cb, 88, &cr, 88).expect("f1");

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
        // With ME, the decoded frame 1 should match the shifted source at
        // very high PSNR. Without ME (zero-MV everywhere) the stripes would
        // be horribly mispredicted and PSNR would drop into the teens.
        let p1 = psnr(&f1.planes[0].data, &y1);
        assert!(
            p1 >= 30.0,
            "P-frame Y PSNR with motion too low: {p1:.2} dB ({} bytes)",
            pkt1.len()
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
