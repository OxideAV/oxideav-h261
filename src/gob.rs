//! H.261 GOB layer parser — §4.2.2 of ITU-T Rec. H.261.
//!
//! GOB layout:
//!
//! | Field   | Bits | Notes                                                  |
//! |---------|------|--------------------------------------------------------|
//! | GBSC    | 16   | `0000 0000 0000 0001`                                  |
//! | GN      | 4    | Group number 1..=12 (CIF) or 1,3,5 (QCIF); 13-15 rsvd  |
//! | GQUANT  | 5    | Initial quantiser for this GOB (1..=31)                |
//! | GEI     | 1    | If `1`, 8-bit GSPARE follows, then GEI repeats.        |
//! | GSPARE  | 8    |                                                        |
//!
//! Each GOB covers a 176x48 luma region (equivalently 11 x 3 macroblocks).
//! GOBs are numbered 1..=12 in CIF with the arrangement shown in Figure 6.
//! QCIF pictures transmit only GOBs 1, 3, 5 (the top-left 3 GOBs of CIF).

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

/// Parsed GOB header (not including the preceding start code).
#[derive(Clone, Debug)]
pub struct GobHeader {
    pub gn: u8,
    pub gquant: u8,
}

/// Parse a GOB header from a reader positioned at the start of the 16-bit
/// GBSC zero prefix.
///
/// Consumes the GBSC (16 bits), GN (4 bits), GQUANT (5 bits), and the
/// GEI/GSPARE loop.
pub fn parse_gob_header(br: &mut BitReader<'_>) -> Result<GobHeader> {
    // GBSC = 16 bits = 0x0001.
    let gbsc = br.read_u32(16)?;
    if gbsc != 0x0001 {
        return Err(Error::invalid(format!(
            "h261 GOB: bad GBSC 0x{gbsc:04x} (want 0x0001)"
        )));
    }
    let gn = br.read_u32(4)? as u8;
    if gn == 0 {
        return Err(Error::invalid(
            "h261 GOB: GN==0 indicates a PSC, not a GBSC",
        ));
    }
    if gn >= 13 {
        return Err(Error::invalid(format!("h261 GOB: GN={gn} is reserved")));
    }
    let gquant = br.read_u32(5)? as u8;
    if gquant == 0 {
        return Err(Error::invalid("h261 GOB: GQUANT == 0"));
    }
    // GEI / GSPARE loop.
    loop {
        let gei = br.read_u1()?;
        if gei == 0 {
            break;
        }
        let _gspare = br.read_u32(8)?;
    }
    Ok(GobHeader { gn, gquant })
}

/// MB address → (row, column) within a GOB. H.261 GOBs are 3 rows x 11
/// columns = 33 MBs, numbered 1..=33 (row-major, top-left first) per Figure 8.
pub fn mba_to_mb_rc(mba: u8) -> (usize, usize) {
    debug_assert!((1..=33).contains(&mba));
    let idx = (mba - 1) as usize;
    (idx / 11, idx % 11)
}

/// GOB number → (gob_row, gob_col) within a CIF picture. CIF is a 2-column,
/// 6-row arrangement of GOBs (Figure 6).
pub fn gn_to_gob_rc(gn: u8) -> (usize, usize) {
    // GN 1..=12. Figure 6:
    //   [1  2]
    //   [3  4]
    //   [5  6]
    //   [7  8]
    //   [9 10]
    //   [11 12]
    debug_assert!((1..=12).contains(&gn));
    let idx = (gn - 1) as usize;
    (idx / 2, idx % 2)
}

/// Top-left luma pel position of a CIF GOB. Luma = 176x48 per GOB.
pub fn cif_gob_origin_luma(gn: u8) -> (usize, usize) {
    let (r, c) = gn_to_gob_rc(gn);
    (c * 176, r * 48)
}

/// Top-left luma pel position of a QCIF GOB (GN must be 1, 3, or 5).
/// QCIF has 3 GOBs stacked vertically (each 176x48 luma), per the spec.
pub fn qcif_gob_origin_luma(gn: u8) -> (usize, usize) {
    let row = match gn {
        1 => 0,
        3 => 1,
        5 => 2,
        _ => panic!("qcif GOB number must be 1, 3, or 5, got {gn}"),
    };
    (0, row * 48)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mba_layout() {
        assert_eq!(mba_to_mb_rc(1), (0, 0));
        assert_eq!(mba_to_mb_rc(11), (0, 10));
        assert_eq!(mba_to_mb_rc(12), (1, 0));
        assert_eq!(mba_to_mb_rc(33), (2, 10));
    }

    #[test]
    fn gn_layout() {
        assert_eq!(gn_to_gob_rc(1), (0, 0));
        assert_eq!(gn_to_gob_rc(2), (0, 1));
        assert_eq!(gn_to_gob_rc(3), (1, 0));
        assert_eq!(gn_to_gob_rc(12), (5, 1));
    }
}
