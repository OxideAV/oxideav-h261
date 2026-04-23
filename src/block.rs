//! H.261 block-layer decoding — §4.2.4 of ITU-T Rec. H.261.
//!
//! Each 8x8 block in a macroblock is coded as a sequence of TCOEFF VLCs
//! ending in an EOB marker, in the zig-zag order given by Figure 12. The
//! INTRA macroblock always transmits all six blocks; the INTER family uses
//! the CBP (when present) to flag which blocks carry any data.
//!
//! For **INTRA** blocks the first transmitted coefficient is the DC — a
//! fixed-length 8-bit code per Table 6/H.261: the bitstream value `n` maps
//! to a reconstruction level of `8 * n`, except `n == 255` which maps to
//! `1024` (instead of the illegal `2040`). Values `0` and `128` are
//! forbidden. The remaining coefficients use the TCOEFF VLC (Table 5) with
//! the standard (run, level) escape.
//!
//! For **INTER** blocks there is no fixed DC; the first coefficient is
//! coded with the short `1s` code when its `|level| == 1`, falling back to
//! the normal VLC table otherwise. All subsequent coefficients use the
//! full table (including `11s` for run=0 level=1).
//!
//! Dequantisation (§4.2.4, QUANT in `1..=31`, step = 2*QUANT):
//!
//! ```text
//!   if QUANT odd:  REC = QUANT * (2*level + sign)
//!   if QUANT even: REC = QUANT * (2*level + sign) - sign
//!   (with `sign` = +1 for level>0, -1 for level<0, and REC=0 for level=0)
//! ```
//!
//! Clipped to `[-2048, 2047]`.

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

use crate::idct::{idct_intra, idct_signed};
use crate::tables::{decode_tcoeff, TcoeffSym, ZIGZAG};

/// Decode the 8-bit INTRA DC fixed-length code and return the IDCT-input DC
/// level per Table 6/H.261.
pub fn decode_intra_dc(br: &mut BitReader<'_>) -> Result<i32> {
    let v = br.read_u32(8)? as u8;
    if v == 0x00 || v == 0x80 {
        return Err(Error::invalid(format!(
            "h261 INTRA DC: forbidden bitstream value 0x{v:02x}"
        )));
    }
    if v == 0xFF {
        return Ok(1024);
    }
    Ok((v as i32) * 8)
}

/// Dequantise one AC coefficient per §4.2.4. `quant` is QUANT (1..=31),
/// `level` is the signed VLC-decoded level.
pub fn dequant_ac(level: i32, quant: u32) -> i32 {
    if level == 0 {
        return 0;
    }
    let q = quant as i32;
    let (sign, abs) = if level > 0 { (1, level) } else { (-1, -level) };
    // REC = q * (2*level + sign) for odd quant; subtract sign for even quant.
    // Using |level|: |REC| = q*(2|level|+1). Even quant: |REC|-=1.
    let mut mag = q * (2 * abs + 1);
    if quant & 1 == 0 {
        mag -= 1;
    }
    let rec = sign * mag;
    rec.clamp(-2048, 2047)
}

/// Decode one INTRA block — INTRA DC + TCOEFF AC + EOB — and write the 8x8
/// pel-domain intra samples (range 0..=255) into `out`.
pub fn decode_intra_block(br: &mut BitReader<'_>, quant: u32, out: &mut [u8; 64]) -> Result<()> {
    let dc_level = decode_intra_dc(br)?;
    let mut coeffs = [0i32; 64];
    coeffs[0] = dc_level;

    decode_ac_coeffs(br, &mut coeffs, 1, quant, /*is_first_inter*/ false)?;

    idct_intra(&coeffs, out);
    Ok(())
}

/// Decode one INTER block — TCOEFF (possibly `1s` first) + EOB — and return
/// the signed residual samples (clipped to -256..=255) in `out`.
pub fn decode_inter_block(br: &mut BitReader<'_>, quant: u32, out: &mut [i32; 64]) -> Result<()> {
    let mut coeffs = [0i32; 64];
    decode_ac_coeffs(br, &mut coeffs, 0, quant, /*is_first_inter*/ true)?;
    idct_signed(&coeffs, out);
    Ok(())
}

/// Common loop over (run, level) + EOB for both INTRA AC and INTER. Writes
/// dequantised coefficients into `coeffs` at zig-zag positions.
fn decode_ac_coeffs(
    br: &mut BitReader<'_>,
    coeffs: &mut [i32; 64],
    start: usize,
    quant: u32,
    is_first_inter: bool,
) -> Result<()> {
    let mut idx = start;
    let mut first = is_first_inter;
    loop {
        let sym = decode_tcoeff(br, first)?;
        first = false;
        match sym {
            TcoeffSym::Eob => return Ok(()),
            TcoeffSym::RunLevel { run, level_abs } => {
                let sign = br.read_u1()?; // 0 = positive, 1 = negative
                let level_signed = if sign == 1 {
                    -(level_abs as i32)
                } else {
                    level_abs as i32
                };
                idx = idx.saturating_add(run as usize);
                if idx > 63 {
                    return Err(Error::invalid(format!(
                        "h261 block: AC run overflow (idx={idx}, run={run})"
                    )));
                }
                coeffs[ZIGZAG[idx]] = dequant_ac(level_signed, quant);
                idx += 1;
            }
            TcoeffSym::Escape => {
                let run = br.read_u32(6)? as u8;
                let raw = br.read_u32(8)?;
                let level: i32 = if raw == 0 {
                    return Err(Error::invalid("h261 escape: level == 0 forbidden"));
                } else if raw == 0x80 {
                    return Err(Error::invalid(
                        "h261 escape: level == -128 forbidden (per Table 5 level FLC)",
                    ));
                } else if raw & 0x80 != 0 {
                    raw as i32 - 256
                } else {
                    raw as i32
                };
                idx = idx.saturating_add(run as usize);
                if idx > 63 {
                    return Err(Error::invalid(format!(
                        "h261 escape: run overflow (idx={idx}, run={run}, level={level})"
                    )));
                }
                coeffs[ZIGZAG[idx]] = dequant_ac(level, quant);
                idx += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intra_dc_basic() {
        let data = [0x10u8];
        let mut br = BitReader::new(&data);
        assert_eq!(decode_intra_dc(&mut br).unwrap(), 0x10 * 8);
    }

    #[test]
    fn intra_dc_special_ff() {
        let data = [0xFFu8];
        let mut br = BitReader::new(&data);
        assert_eq!(decode_intra_dc(&mut br).unwrap(), 1024);
    }

    #[test]
    fn intra_dc_zero_is_forbidden() {
        let data = [0x00u8];
        let mut br = BitReader::new(&data);
        assert!(decode_intra_dc(&mut br).is_err());
    }

    #[test]
    fn intra_dc_0x80_is_forbidden() {
        let data = [0x80u8];
        let mut br = BitReader::new(&data);
        assert!(decode_intra_dc(&mut br).is_err());
    }

    #[test]
    fn dequant_level_1_odd_q() {
        // level=+1, q=1 -> REC = 1*(2*1+1) = 3.
        assert_eq!(dequant_ac(1, 1), 3);
        // level=-1, q=1 -> REC = -3.
        assert_eq!(dequant_ac(-1, 1), -3);
    }

    #[test]
    fn dequant_level_1_even_q() {
        // level=+1, q=2 -> REC = 2*(2+1)-1 = 5.
        assert_eq!(dequant_ac(1, 2), 5);
        // level=-1, q=2 -> REC = -(2*(2+1)-1) = -5.
        assert_eq!(dequant_ac(-1, 2), -5);
    }

    #[test]
    fn dequant_matches_spec_row_for_q8_l1() {
        // Spec Table "Reconstruction levels": QUANT=8, Level=1 -> REC=23.
        assert_eq!(dequant_ac(1, 8), 23);
        assert_eq!(dequant_ac(2, 8), 39);
        assert_eq!(dequant_ac(-1, 8), -23);
    }
}
