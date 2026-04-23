//! Start-code scanning for H.261.
//!
//! H.261 uses a single start-code prefix, transmitted most-significant-bit
//! first into the compressed bitstream:
//! * **GBSC** — 16 bits `0000 0000 0000 0001`, followed by a 4-bit `GN`.
//!   `GN == 0` is reserved and used by the PSC to identify itself;
//!   `GN == 1..=12` are valid GOBs for CIF (QCIF uses `1, 3, 5`); `GN ==
//!   13..=15` are reserved.
//! * **PSC** — 20 bits `0000 0000 0000 0001 0000`. Structurally a GBSC
//!   with `GN == 0` followed by the 5-bit TR field.
//!
//! Crucially, unlike H.263's byte-aligned start codes, H.261 start codes
//! are **not** required to sit on byte boundaries. The shortest MBA
//! codeword is 1 bit (`1` for `Diff(1)`) and the VLC tables are designed
//! so that no legal concatenation produces 15 consecutive zero bits
//! followed by a `1` — so scanning the bitstream bit-by-bit for the
//! 16-bit pattern `0000 0000 0000 0001` uniquely locates every start
//! code.
//!
//! This scanner returns bit-granular positions in addition to the
//! byte-rounded-down position, so `decoder::decode_picture_body` can
//! re-seek the bitreader to the start of the next start code.

/// `GN == 0` indicates a PSC at this position.
pub const GN_PICTURE: u8 = 0;

/// One detected start-code event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StartCode {
    /// Byte offset of the byte containing the first of the 16 zero bits.
    pub byte_pos: usize,
    /// Bit offset from the start of `data` to the first of the 16 zero
    /// bits that lead the start code.
    pub bit_pos: u64,
    /// 4-bit `GN` field that immediately follows the 16-bit zero prefix +
    /// `1` sync bit.
    pub gn: u8,
}

/// Find the next H.261 start code in `data`, beginning at bit offset `start_bit`.
///
/// Scans bit-by-bit for the 16-bit prefix `0000 0000 0000 0001`, then
/// reads the next 4 bits as `GN`.
pub fn find_next_start_code_bits(data: &[u8], start_bit: u64) -> Option<StartCode> {
    let total_bits = data.len() as u64 * 8;
    if start_bit + 20 > total_bits {
        return None;
    }
    let read_bit = |pos: u64| -> u8 {
        let byte = (pos / 8) as usize;
        let shift = 7 - (pos % 8) as u8;
        (data[byte] >> shift) & 1
    };

    // Build a 16-bit window starting at `start_bit`. After reading bit
    // `start_bit + 15`, the window covers bits `start_bit ..= start_bit+15`.
    let mut window: u32 = 0;
    for i in 0..16 {
        window = (window << 1) | (read_bit(start_bit + i) as u32);
    }
    // Target: 15 zeros then a 1 = 0x0001.
    const TARGET: u32 = 0b0000_0000_0000_0001;
    let mut pos = start_bit + 15; // index of the most recently read bit
    loop {
        if (window & 0xFFFF) == TARGET {
            let prefix_start = pos - 15;
            if prefix_start + 20 > total_bits {
                return None;
            }
            let mut gn = 0u8;
            for i in 0..4 {
                gn = (gn << 1) | read_bit(prefix_start + 16 + i);
            }
            return Some(StartCode {
                byte_pos: (prefix_start / 8) as usize,
                bit_pos: prefix_start,
                gn,
            });
        }
        pos += 1;
        if pos >= total_bits {
            return None;
        }
        window = (window << 1) | (read_bit(pos) as u32);
    }
}

/// Convenience wrapper that scans from byte `pos` upward (bit-aligned to
/// the start of that byte).
pub fn find_next_start_code(data: &[u8], pos: usize) -> Option<StartCode> {
    find_next_start_code_bits(data, (pos as u64) * 8)
}

/// Iterator over every start code in `data`, left-to-right.
pub fn iter_start_codes(data: &[u8]) -> impl Iterator<Item = StartCode> + '_ {
    let mut bit = 0u64;
    std::iter::from_fn(move || {
        let sc = find_next_start_code_bits(data, bit)?;
        // Skip past the 16-bit prefix + 4 bit GN = 20 bits so we don't
        // re-match.
        bit = sc.bit_pos + 20;
        Some(sc)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_psc_byte_aligned() {
        let data = [0x00, 0x01, 0x00, 0x16];
        let v: Vec<_> = iter_start_codes(&data).collect();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].gn, 0);
        assert_eq!(v[0].bit_pos, 0);
    }

    #[test]
    fn finds_gbsc_gn_1_byte_aligned() {
        let data = [0x00, 0x01, 0x1F, 0x00];
        let v: Vec<_> = iter_start_codes(&data).collect();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].gn, 1);
        assert_eq!(v[0].bit_pos, 0);
    }

    #[test]
    fn finds_nonaligned_start_code() {
        // Construct a stream with a start code starting at bit 3.
        // Bit 0-2: `111` (arbitrary leading).
        // Bit 3-17: 15 zeros of the prefix.
        // Bit 18: `1` (sync — last bit of 16-bit prefix).
        // Bit 19-22: `0011` (GN=3).
        // Bit 23: `0` (pad).
        //
        // Bytes:
        //   byte 0 bits 0-7  = 1,1,1,0,0,0,0,0 = 0xE0
        //   byte 1 bits 8-15 = 0,0,0,0,0,0,0,0 = 0x00
        //   byte 2 bits 16-23= 0,0,1,0,0,1,1,0 = 0x26
        let data = [0xE0u8, 0x00, 0x26];
        let sc = find_next_start_code_bits(&data, 0).unwrap();
        assert_eq!(sc.bit_pos, 3);
        assert_eq!(sc.byte_pos, 0);
        assert_eq!(sc.gn, 3);
    }

    #[test]
    fn stops_when_not_enough_bits() {
        let data = [0x00, 0x01]; // too short for the 4-bit GN
        assert!(find_next_start_code_bits(&data, 0).is_none());
    }
}
