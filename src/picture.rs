//! H.261 picture header parser — §4.2.1 of ITU-T Rec. H.261 (03/93).
//!
//! Picture layout (in bitstream order, MSB-first):
//!
//! | Field    | Bits | Notes                                                |
//! |----------|------|------------------------------------------------------|
//! | PSC      | 20   | `0000 0000 0000 0001 0000`                           |
//! | TR       | 5    | Temporal reference (mod 32)                          |
//! | PTYPE    | 6    | b1 split-screen, b2 document-cam, b3 freeze-release, |
//! |          |      | b4 source format (0=QCIF, 1=CIF), b5 HI_RES          |
//! |          |      | (1 = off), b6 spare.                                 |
//! | PEI      | 1    | If `1`, 8-bit PSPARE follows, then PEI repeats.      |
//! | PSPARE   | 8    |                                                      |
//!
//! The GOB layer follows immediately.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

/// H.261 source format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceFormat {
    /// QCIF, 176 x 144 luma, 3 GOBs (1, 3, 5).
    Qcif,
    /// CIF, 352 x 288 luma, 12 GOBs (1..=12).
    Cif,
}

impl SourceFormat {
    /// `(luma_width, luma_height)`.
    pub fn dimensions(self) -> (u32, u32) {
        match self {
            SourceFormat::Qcif => (176, 144),
            SourceFormat::Cif => (352, 288),
        }
    }

    /// GOB numbers used by this format. QCIF uses GN=1,3,5 (the odd-numbered
    /// GOBs of a CIF picture). CIF uses GN=1..=12 in order.
    pub fn gob_numbers(self) -> &'static [u8] {
        match self {
            SourceFormat::Qcif => &[1, 3, 5],
            SourceFormat::Cif => &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
    }
}

/// Parsed H.261 picture header.
#[derive(Clone, Debug)]
pub struct PictureHeader {
    pub temporal_reference: u8,
    pub split_screen: bool,
    pub document_camera: bool,
    pub freeze_release: bool,
    pub source_format: SourceFormat,
    /// HI_RES still-image mode (Annex D). `true` when signalled off (bit = 1),
    /// `false` when signalled on. We currently treat Annex D as a no-op.
    pub hi_res_off: bool,
    pub width: u32,
    pub height: u32,
}

/// Parse the picture header assuming `br` is positioned at the start of the
/// PSC (20-bit zero-prefix + 1 sync).
pub fn parse_picture_header(br: &mut BitReader<'_>) -> Result<PictureHeader> {
    // PSC: 20 bits = 0000 0000 0000 0001 0000 = 0x00010.
    let psc = br.read_u32(20)?;
    const PSC_VALUE: u32 = 0x00010;
    if psc != PSC_VALUE {
        return Err(Error::invalid(format!(
            "h261 picture: bad PSC 0x{psc:05x} (want 0x{PSC_VALUE:05x})"
        )));
    }

    let tr = br.read_u32(5)? as u8;

    // PTYPE 6 bits.
    let split_screen = br.read_u1()? == 1;
    let document_camera = br.read_u1()? == 1;
    let freeze_release = br.read_u1()? == 1;
    let source_fmt_bit = br.read_u1()?;
    let hi_res_bit = br.read_u1()?; // 0 = on (Annex D), 1 = off
    let _spare = br.read_u1()?;
    let source_format = if source_fmt_bit == 0 {
        SourceFormat::Qcif
    } else {
        SourceFormat::Cif
    };

    // PEI / PSPARE loop.
    loop {
        let pei = br.read_u1()?;
        if pei == 0 {
            break;
        }
        let _pspare = br.read_u32(8)?;
    }

    let (width, height) = source_format.dimensions();
    Ok(PictureHeader {
        temporal_reference: tr,
        split_screen,
        document_camera,
        freeze_release,
        source_format,
        hi_res_off: hi_res_bit == 1,
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal QCIF I-picture header (no optional fields) for a round-trip
    /// through the parser.
    ///
    /// PSC (20)        = 0000 0000 0000 0001 0000
    /// TR (5)          = 00001 (=1)
    /// PTYPE (6)       = 1 0 0 0 1 0 (split, cam, freeze, fmt=QCIF, HI_RES off, spare)
    /// PEI (1)         = 0
    ///
    /// Total = 32 bits = 4 bytes.
    fn minimal_qcif_header() -> Vec<u8> {
        // Concatenate the bit fields:
        // PSC:  0000 0000 0000 0001 0000
        // TR:   0 0001
        // PT:   1 0 0 0 1 0
        // PEI:  0
        //
        // Layout:
        //  [0..20] = 00000000 00000001 0000____
        //  TR starts at bit 20: 0_0001  -> bits 20..25
        //  PTYPE starts at bit 25: 100010  -> bits 25..31
        //  PEI at bit 31: 0
        //
        //  byte 0 = 00000000
        //  byte 1 = 00000001
        //  byte 2 = 0000_0000  -> top 4 bits are the last 4 bits of the 20-bit PSC (=0000),
        //                        next 4 bits are TR's high nibble=0000
        //  Wait — TR is 5 bits: 00001. It begins at bit 20. Bits 20..24 of the byte-aligned
        //  stream are bits 4..0 of byte 2 (byte 2 covers bits 16..23). So TR's 5 bits span
        //  byte 2 bit 4..0 and byte 3 bit 7. Let's just build bit-by-bit.
        let mut bits: Vec<u8> = Vec::new();
        let append = |v: &mut Vec<u8>, val: u32, n: u32| {
            for i in (0..n).rev() {
                v.push(((val >> i) & 1) as u8);
            }
        };
        append(&mut bits, 0x00010, 20); // PSC
        append(&mut bits, 1, 5); // TR = 1
        append(&mut bits, 1, 1); // split
        append(&mut bits, 0, 1); // cam
        append(&mut bits, 0, 1); // freeze
        append(&mut bits, 0, 1); // fmt = QCIF
        append(&mut bits, 1, 1); // HI_RES off
        append(&mut bits, 0, 1); // spare
        append(&mut bits, 0, 1); // PEI = 0
                                 // Pad to byte boundary with zeros.
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
    fn parses_qcif_header() {
        let data = minimal_qcif_header();
        let mut br = BitReader::new(&data);
        let p = parse_picture_header(&mut br).unwrap();
        assert_eq!(p.temporal_reference, 1);
        assert_eq!(p.source_format, SourceFormat::Qcif);
        assert!(p.split_screen);
        assert!(!p.document_camera);
        assert!(!p.freeze_release);
        assert!(p.hi_res_off);
        assert_eq!(p.width, 176);
        assert_eq!(p.height, 144);
    }
}
