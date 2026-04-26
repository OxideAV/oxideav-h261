//! H.261 decoder front-end — walks start codes, dispatches picture / GOB /
//! MB parsing.
//!
//! The decoder buffers incoming packets until a complete coded picture is
//! identified by a pair of PSCs (or PSC + EOF). Each picture produces one
//! `VideoFrame` of YUV 4:2:0 pels in an `oxideav_core::Frame::Video`.
//!
//! H.261 has no explicit end-of-sequence marker — on flush we drain whatever
//! is still buffered.

use std::collections::VecDeque;

use oxideav_core::bits::BitReader;
use oxideav_core::frame::VideoPlane;
use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, Rational, Result, TimeBase, VideoFrame,
};

use crate::gob::{cif_gob_origin_luma, parse_gob_header, qcif_gob_origin_luma};
use crate::mb::{decode_macroblock, decode_mba_diff, MbContext, Picture};
use crate::picture::{parse_picture_header, PictureHeader, SourceFormat};
use crate::start_code::{find_next_start_code, StartCode, GN_PICTURE};

/// Factory for the registry.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(H261Decoder::new(params.codec_id.clone())))
}

pub struct H261Decoder {
    codec_id: CodecId,
    buffer: Vec<u8>,
    ready_frames: VecDeque<VideoFrame>,
    pending_pts: Option<i64>,
    pending_tb: TimeBase,
    eof: bool,
    /// Previous decoded picture, kept as the motion-compensation reference
    /// for the next picture.
    reference: Option<Picture>,
}

impl H261Decoder {
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            buffer: Vec::new(),
            ready_frames: VecDeque::new(),
            pending_pts: None,
            pending_tb: TimeBase::new(1, 30_000),
            eof: false,
            reference: None,
        }
    }

    fn process(&mut self) -> Result<()> {
        let data = std::mem::take(&mut self.buffer);
        let mut pos = 0usize;
        let first_psc = loop {
            match find_next_start_code(&data, pos) {
                Some(sc) if sc.gn == GN_PICTURE => break sc,
                Some(sc) => pos = sc.byte_pos + 3,
                None => return Ok(()),
            }
        };
        let mut cur = first_psc.byte_pos;
        loop {
            let mut scan = cur + 3;
            let next_psc = loop {
                match find_next_start_code(&data, scan) {
                    Some(sc) if sc.gn == GN_PICTURE => break Some(sc),
                    Some(sc) => scan = sc.byte_pos + 3,
                    None => break None,
                }
            };
            let end = next_psc.map(|s| s.byte_pos).unwrap_or(data.len());
            if next_psc.is_none() && !self.eof {
                self.buffer.extend_from_slice(&data[cur..]);
                return Ok(());
            }
            let pic_bytes = &data[cur..end];
            self.decode_one_picture(pic_bytes)?;
            match next_psc {
                Some(sc) => cur = sc.byte_pos,
                None => return Ok(()),
            }
        }
    }

    fn decode_one_picture(&mut self, bytes: &[u8]) -> Result<()> {
        let mut br = BitReader::new(bytes);
        let hdr = parse_picture_header(&mut br)?;
        let pic = decode_picture_body(&mut br, &hdr, bytes, self.reference.as_ref())?;
        let frame = pic_to_video_frame(&pic, self.pending_pts, self.pending_tb);
        self.reference = Some(pic);
        self.ready_frames.push_back(frame);
        Ok(())
    }
}

/// Decode the body of a picture (everything after the picture header).
///
/// Strategy: walk the GOB start codes present in `bytes`. For each GOB we
/// expect to find, parse its header (which tells us GN and GQUANT), then
/// iterate MBs by MBA differences until we run out of MBs or hit the next
/// start code.
pub fn decode_picture_body(
    br: &mut BitReader<'_>,
    hdr: &PictureHeader,
    bytes: &[u8],
    reference: Option<&Picture>,
) -> Result<Picture> {
    let mut pic = Picture::new(hdr.width as usize, hdr.height as usize);
    let gobs: Vec<StartCode> = collect_start_codes(bytes);

    let expected_gns = hdr.source_format.gob_numbers();
    for &expected_gn in expected_gns {
        // Seek to the GBSC for this GOB (starting from current bit position).
        let cur_bit = br.bit_position();
        let target = gobs.iter().find(|g| g.bit_pos >= cur_bit);
        let Some(target) = target else {
            return Err(Error::invalid(format!(
                "h261: missing GBSC for GN={expected_gn} (no further start codes)"
            )));
        };
        if target.gn == GN_PICTURE {
            return Err(Error::invalid(format!(
                "h261: expected GBSC for GN={expected_gn} but found a PSC first"
            )));
        }
        if target.gn != expected_gn {
            return Err(Error::invalid(format!(
                "h261: GOB order mismatch — expected GN={expected_gn}, found GN={}",
                target.gn
            )));
        }
        // Align to the first zero bit of the GBSC.
        let pad = target.bit_pos - cur_bit;
        if pad > 0 {
            br.skip(pad as u32)?;
        }
        let gob_hdr = parse_gob_header(br)?;
        let mut quant = gob_hdr.gquant as u32;
        let (gob_x, gob_y) = match hdr.source_format {
            SourceFormat::Cif => cif_gob_origin_luma(gob_hdr.gn),
            SourceFormat::Qcif => qcif_gob_origin_luma(gob_hdr.gn),
        };

        let mut ctx = MbContext::reset();
        let mut current_mba: i32 = 0;

        loop {
            // Stop when we hit the next start code. The 16-bit prefix is
            // `0000 0000 0000 0001` — 15 zeros + a `1` at any bit position.
            //
            // §4.2.2: the start code can only appear after a complete MB
            // (the encoder always byte-aligns the picture, and the next
            // picture's PSC begins on that byte boundary). If we have fewer
            // than 16 bits remaining we cannot be at a start code (it would
            // need ≥16 bits) — but we may still have a valid MB to decode
            // followed by padding zeros, so do NOT break early on
            // `bits_remaining < 16`; let the MB decoder consume what's there
            // and fall out via `decode_mba_diff` returning `None`.
            //
            // For the start-code check itself we need at least 16 bits.
            // Without 16 bits there can't be a start code, so skip the test.
            let remaining_bits = br.bits_remaining();
            if remaining_bits == 0 {
                break;
            }
            if remaining_bits >= 16 {
                let peek16 = br.peek_u32(16)?;
                if peek16 == 0x0001 {
                    break;
                }
            }
            let diff = match decode_mba_diff(br)? {
                Some(d) => d as i32,
                None => break,
            };
            let new_mba = current_mba + diff;
            if !(1..=33).contains(&new_mba) {
                return Err(Error::invalid(format!(
                    "h261 MB: MBA out of range {new_mba} (GN={}, prev_mba={})",
                    gob_hdr.gn, current_mba
                )));
            }
            // Handle skipped MBs (those with no transmitted data — they're
            // copied from the reference with zero MV in INTER mode, or left
            // zero for an I-opening picture).
            if let Some(ref_pic) = reference {
                for skipped_mba in (current_mba + 1)..new_mba {
                    copy_skipped_mb(&mut pic, ref_pic, skipped_mba as u8, gob_x, gob_y);
                }
                // Non-consecutive MBA resets the MV predictor (§4.2.3.4).
                if new_mba != current_mba + 1 {
                    ctx = MbContext::reset();
                }
            }
            current_mba = new_mba;
            decode_macroblock(
                br,
                new_mba as u8,
                gob_x,
                gob_y,
                &mut quant,
                &mut ctx,
                &mut pic,
                reference,
            )?;
        }
        // Pad any remaining skipped MBs through MBA=33.
        if let Some(ref_pic) = reference {
            for skipped_mba in (current_mba + 1)..=33 {
                copy_skipped_mb(&mut pic, ref_pic, skipped_mba as u8, gob_x, gob_y);
            }
        }
    }

    Ok(pic)
}

/// Copy an MB verbatim from `reference` into `pic` at the same position.
/// Used for skipped MBs in a P-like picture.
fn copy_skipped_mb(pic: &mut Picture, reference: &Picture, mba: u8, gob_x: usize, gob_y: usize) {
    let idx = (mba - 1) as usize;
    let mb_col = idx % 11;
    let mb_row = idx / 11;
    let luma_x = gob_x + mb_col * 16;
    let luma_y = gob_y + mb_row * 16;
    // Guard against picture-edge overflows on malformed streams.
    if luma_x + 16 > pic.y_stride || luma_y + 16 > pic.y.len() / pic.y_stride {
        return;
    }
    for j in 0..16 {
        let dst_off = (luma_y + j) * pic.y_stride + luma_x;
        let src_off = (luma_y + j) * reference.y_stride + luma_x;
        if dst_off + 16 > pic.y.len() || src_off + 16 > reference.y.len() {
            return;
        }
        pic.y[dst_off..dst_off + 16].copy_from_slice(&reference.y[src_off..src_off + 16]);
    }
    let cx = luma_x / 2;
    let cy = luma_y / 2;
    for j in 0..8 {
        let dst_off = (cy + j) * pic.c_stride + cx;
        let src_off = (cy + j) * reference.c_stride + cx;
        if dst_off + 8 > pic.cb.len() || src_off + 8 > reference.cb.len() {
            return;
        }
        pic.cb[dst_off..dst_off + 8].copy_from_slice(&reference.cb[src_off..src_off + 8]);
        pic.cr[dst_off..dst_off + 8].copy_from_slice(&reference.cr[src_off..src_off + 8]);
    }
}

fn collect_start_codes(bytes: &[u8]) -> Vec<StartCode> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while let Some(sc) = find_next_start_code(bytes, pos) {
        out.push(sc);
        pos = sc.byte_pos + 3;
    }
    out
}

/// Build a stride-packed YUV420P `VideoFrame` from a `Picture`.
///
/// Stream-level properties (pixel format, width, height, time base) live on
/// the stream's `CodecParameters`; the frame only carries pts + planes. The
/// `_tb` argument is retained for source-compat but ignored.
pub fn pic_to_video_frame(pic: &Picture, pts: Option<i64>, _tb: TimeBase) -> VideoFrame {
    let w = pic.width;
    let h = pic.height;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        y[row * w..row * w + w].copy_from_slice(&pic.y[row * pic.y_stride..row * pic.y_stride + w]);
    }
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for row in 0..ch {
        cb[row * cw..row * cw + cw]
            .copy_from_slice(&pic.cb[row * pic.c_stride..row * pic.c_stride + cw]);
        cr[row * cw..row * cw + cw]
            .copy_from_slice(&pic.cr[row * pic.c_stride..row * pic.c_stride + cw]);
    }
    VideoFrame {
        pts,
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: cb,
            },
            VideoPlane {
                stride: cw,
                data: cr,
            },
        ],
    }
}

impl Decoder for H261Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.pending_pts = packet.pts;
        self.pending_tb = packet.time_base;
        self.buffer.extend_from_slice(&packet.data);
        self.process()
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.ready_frames.pop_front() {
            return Ok(Frame::Video(f));
        }
        if self.eof {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.buffer.clear();
        self.ready_frames.clear();
        self.pending_pts = None;
        self.eof = false;
        self.reference = None;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        self.process()
    }
}

/// Build a `CodecParameters` from a parsed picture header.
pub fn codec_parameters_from_header(hdr: &PictureHeader) -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
    params.width = Some(hdr.width);
    params.height = Some(hdr.height);
    // H.261 specifies 30000/1001 fps nominally (§3.1).
    params.frame_rate = Some(Rational::new(30_000, 1001));
    params
}
