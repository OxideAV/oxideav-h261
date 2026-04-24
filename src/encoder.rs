//! H.261 encoder — foundation.
//!
//! The public entry point is [`encode_intra_picture`], which takes a
//! Y/Cb/Cr 4:2:0 source frame at either QCIF (176x144) or CIF (352x288)
//! and produces an elementary-stream byte buffer containing a single
//! I-picture: PSC + picture header + all GOBs + INTRA macroblocks +
//! zero-padded tail.
//!
//! The encoder is deliberately simple — no rate control, no dead-zone
//! tuning, no motion estimation. It's designed to be the minimum
//! decodable artefact so we can verify each piece of the VLC / DCT /
//! quantisation stack end-to-end against the reference decoder (either
//! our own or ffmpeg's).
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
//! ## MB layer — intra only
//!
//! For each of the 33 MBs in a GOB we emit:
//!
//! * MBA VLC — `Diff(1)` for the first coded MB, then 1-differences.
//! * MTYPE = `Intra` (4-bit `0001`), optionally `Intra+MQUANT` (7-bit
//!   `0000 001`) when GQUANT doesn't match the current MQUANT.
//! * CBP is absent for INTRA (all 6 blocks are always coded).
//! * 6 blocks (Y1, Y2, Y3, Y4, Cb, Cr). Each block is:
//!   * INTRA DC — 8-bit FLC per Table 6.
//!   * Zero or more AC `(run, level)` entries, each as a TCOEFF VLC
//!     prefix + sign bit, or 20-bit escape.
//!   * EOB — `10`.
//!
//! The residual path (INTER family + motion compensation) is a follow-up;
//! this module only ever produces I-pictures.

use oxideav_core::bits::BitWriter;
use oxideav_core::{Error, Result};

use crate::fdct::fdct_intra;
use crate::picture::SourceFormat;
use crate::quant::{quant_ac, quant_intra_dc};
use crate::tables::{
    encode_cbp, encode_mba_diff, lookup_tcoeff, MBA_STUFFING, MTYPE_INTRA, MTYPE_INTRA_MQUANT,
    ZIGZAG,
};

/// Minimum quantiser we use. QUANT=1 gives best quality but the DCT AC
/// dynamic range can push `|coeff|` past 2047 which doesn't fit the
/// level's 12-bit storage; we still clamp but quality stays fine.
pub const DEFAULT_QUANT: u32 = 8;

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
    if y.len() < y_stride * h
        || cb.len() < cb_stride * (h / 2)
        || cr.len() < cr_stride * (h / 2)
    {
        return Err(Error::invalid("h261 encode: input plane too short"));
    }

    let mut bw = BitWriter::with_capacity(4096);
    write_picture_header(&mut bw, fmt, temporal_reference);

    for &gn in fmt.gob_numbers() {
        write_gob_header(&mut bw, gn, quant);
        let (gob_x, gob_y) = gob_origin_luma(fmt, gn);
        encode_gob_intra(
            &mut bw, y, y_stride, cb, cb_stride, cr, cr_stride, gob_x, gob_y, quant,
        );
    }

    // Pad to a byte boundary with zeros — the H.261 decoder will either
    // consume the trailing PSC (we don't emit one) or hit EOF and flush.
    bw.align_to_byte();
    Ok(bw.finish())
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

/// Encode the 33 INTRA macroblocks of one GOB.
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
) {
    let mut prev_mba: u8 = 0;
    for mba in 1u8..=33 {
        // MBA difference — always 1 in a fully-coded intra GOB.
        let diff = mba - prev_mba;
        let (bits, code) = encode_mba_diff(diff);
        bw.write_u32(code, bits as u32);
        // MTYPE = INTRA (4-bit 0001). No MQUANT override — we reuse
        // GQUANT for every MB.
        bw.write_u32(MTYPE_INTRA.1, MTYPE_INTRA.0 as u32);

        // Block position in luma pels within the full picture.
        let mb_col = (mba - 1) as usize % 11;
        let mb_row = (mba - 1) as usize / 11;
        let luma_x = gob_x + mb_col * 16;
        let luma_y = gob_y + mb_row * 16;
        encode_intra_mb_blocks(
            bw, y, y_stride, cb, cb_stride, cr, cr_stride, luma_x, luma_y, quant,
        );

        prev_mba = mba;
    }
}

/// Extract the 8x8 intra pel block at `(bx, by)` from `plane` and run the
/// forward DCT + per-block encode.
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
) {
    // Y1..Y4
    for (sub_x, sub_y) in [(0, 0), (8, 0), (0, 8), (8, 8)] {
        let mut pels = [0u8; 64];
        extract_block(y, y_stride, luma_x + sub_x, luma_y + sub_y, &mut pels);
        encode_intra_block(bw, &pels, quant);
    }
    // Cb, Cr at chroma coords (luma / 2).
    let cx = luma_x / 2;
    let cy = luma_y / 2;
    let mut cb_pels = [0u8; 64];
    extract_block(cb, cb_stride, cx, cy, &mut cb_pels);
    encode_intra_block(bw, &cb_pels, quant);
    let mut cr_pels = [0u8; 64];
    extract_block(cr, cr_stride, cx, cy, &mut cr_pels);
    encode_intra_block(bw, &cr_pels, quant);
}

fn extract_block(plane: &[u8], stride: usize, x: usize, y: usize, out: &mut [u8; 64]) {
    for j in 0..8 {
        for i in 0..8 {
            let px = (y + j) * stride + (x + i);
            out[j * 8 + i] = plane.get(px).copied().unwrap_or(0);
        }
    }
}

/// Encode one 8x8 intra block: DC (8-bit FLC) + AC (TCOEFF VLCs) + EOB.
fn encode_intra_block(bw: &mut BitWriter, pels: &[u8; 64], quant: u32) {
    // Forward DCT.
    let mut coeffs = [0i32; 64];
    fdct_intra(pels, &mut coeffs);

    // DC first: raw-transform DC → FLC per Table 6.
    let dc_code = quant_intra_dc(coeffs[0]);
    bw.write_u32(dc_code as u32, 8);

    // AC coefficients: zigzag scan starting at index 1, RLE, then EOB.
    // ZIGZAG[i] is the raster position of the i-th coefficient in scan
    // order. Skip i=0 (DC already emitted).
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
    // End of block — always `10`.
    bw.write_u32(0b10, 2);
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

    // Try VLC table lookup.
    if let Some((bits, code)) = lookup_tcoeff(run, abs) {
        bw.write_u32(code, bits as u32);
        bw.write_u32(sign, 1);
        return;
    }

    // Fallback: escape — 6-bit prefix `000001`, 6-bit run, 8-bit signed level.
    bw.write_u32(0b0000_01, 6);
    bw.write_u32(run as u32 & 0x3F, 6);
    // 8-bit two's complement, excluding the forbidden -128.
    let enc = if level < 0 {
        (level + 256) as u32
    } else {
        level as u32
    };
    bw.write_u32(enc & 0xFF, 8);
}

// Silence unused-import warnings until the CBP / MVD / mquant paths land.
#[allow(dead_code)]
fn _unused_refs() {
    let _ = encode_cbp(1);
    let _ = MBA_STUFFING;
    let _ = MTYPE_INTRA_MQUANT;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::{decode_picture_body, pic_to_video_frame, H261Decoder};
    use crate::picture::parse_picture_header;
    use oxideav_codec::Decoder;
    use oxideav_core::bits::BitReader;
    use oxideav_core::packet::PacketFlags;
    use oxideav_core::{CodecId, Frame, Packet, TimeBase};

    /// Build a neutral-grey QCIF YUV420 source (Y=128, Cb=Cr=128).
    fn neutral_qcif() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y = vec![128u8; 176 * 144];
        let cb = vec![128u8; 88 * 72];
        let cr = vec![128u8; 88 * 72];
        (y, cb, cr)
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
        let bytes = encode_intra_picture(
            SourceFormat::Qcif,
            &y,
            176,
            &cb,
            88,
            &cr,
            88,
            /*quant=*/ 8,
            /*tr=*/ 0,
        )
        .expect("encode");
        assert!(!bytes.is_empty());

        // Decode with our own decoder. We pass the bytes as a packet and
        // flush to force one frame out.
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
        let frame = decoder.receive_frame().expect("frame");
        let vf = match frame {
            Frame::Video(v) => v,
            _ => panic!("expected video"),
        };
        assert_eq!(vf.width, 176);
        assert_eq!(vf.height, 144);
        // All Y pels should be very close to 128.
        let y_plane = &vf.planes[0].data;
        let mut max_err = 0i32;
        for &p in y_plane {
            max_err = max_err.max((p as i32 - 128).abs());
        }
        assert!(max_err <= 2, "max Y error was {max_err}");
        // Chroma sanity.
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
        let bytes = encode_intra_picture(
            SourceFormat::Cif,
            &y,
            352,
            &cb,
            176,
            &cr,
            176,
            8,
            0,
        )
        .expect("encode cif");
        assert!(!bytes.is_empty());

        // Also parse the body with our low-level helper to confirm all 12 GOBs present.
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
        // Build a horizontal Y gradient 32..=224 across the 176 columns.
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
        let bytes = encode_intra_picture(
            SourceFormat::Qcif,
            &y,
            w,
            &cb,
            w / 2,
            &cr,
            w / 2,
            8,
            0,
        )
        .expect("encode gradient");

        let mut decoder = H261Decoder::new(CodecId::new(crate::CODEC_ID_STR));
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
        let frame = decoder.receive_frame().expect("frame");
        let vf = match frame {
            Frame::Video(v) => v,
            _ => panic!("video"),
        };
        // Just check a few sample points are within a reasonable quantisation error.
        let y = &vf.planes[0].data;
        let sample = |x: usize, yy: usize| y[yy * w + x] as i32;
        let expected = |x: usize| 32 + (x * 192) as i32 / w as i32;
        // Pick centre of a few macroblocks.
        for &x in &[24usize, 80, 152] {
            let got = sample(x, 72);
            let want = expected(x);
            assert!(
                (got - want).abs() <= 40,
                "gradient at x={x}: got {got}, want ~{want}"
            );
        }
    }
}
