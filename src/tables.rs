//! VLC tables from ITU-T Rec. H.261 §4.2 (transcribed directly from the PDF).
//!
//! Tables implemented here:
//! * **Table 1** — MBA (macroblock address difference). Values 1..=33, plus
//!   a "stuffing" codeword. Start-code (16-bit zero prefix + `1`) is handled
//!   separately by the start-code scanner.
//! * **Table 2** — MTYPE (macroblock type). 10 entries tagging what follows
//!   in the MB: INTRA/INTER/INTER+MC, +MC+FIL, and whether MQUANT / MVD /
//!   CBP / TCOEFF are present.
//! * **Table 3** — MVD (motion vector data). 32 entries mapped to MV
//!   differentials in the range -16..=15 (with the paired representative
//!   given by `a` and `b = a - 32` / `a + 32` per §4.2.3.4).
//! * **Table 4** — CBP (coded block pattern). 63 entries, one per non-zero
//!   CBP value.
//! * **Table 5** — TCOEFF (transform coefficient run/level). 65 entries plus
//!   EOB, Escape, and the special first-coefficient `1s`.
//!
//! The tables are stored as simple arrays and decoded via linear scan. This
//! is plenty fast for CIF/QCIF (< 1 MB per picture).

use oxideav_core::bits::BitReader;
use oxideav_core::{Error, Result};

/// One entry in a VLC table. `code` occupies the low `bits` bits MSB-first.
#[derive(Clone, Copy, Debug)]
pub struct VlcEntry<T: Copy> {
    pub code: u32,
    pub bits: u8,
    pub value: T,
}

impl<T: Copy> VlcEntry<T> {
    pub const fn new(bits: u8, code: u32, value: T) -> Self {
        Self { code, bits, value }
    }
}

/// Decode one symbol using linear scan over `table`.
pub fn decode_vlc<T: Copy>(br: &mut BitReader<'_>, table: &[VlcEntry<T>]) -> Result<T> {
    let max_bits = table.iter().map(|e| e.bits).max().unwrap_or(0) as u32;
    if max_bits == 0 {
        return Err(Error::invalid("h261 vlc: empty table"));
    }
    let remaining = br.bits_remaining() as u32;
    let peek_bits = max_bits.min(remaining);
    if peek_bits == 0 {
        return Err(Error::invalid("h261 vlc: no bits available"));
    }
    let peeked = br.peek_u32(peek_bits)?;
    let peeked_full = peeked << (max_bits - peek_bits);
    for e in table {
        if (e.bits as u32) > peek_bits {
            continue;
        }
        let shift = max_bits - e.bits as u32;
        let prefix = peeked_full >> shift;
        if prefix == e.code {
            br.consume(e.bits as u32)?;
            return Ok(e.value);
        }
    }
    Err(Error::invalid("h261 vlc: no matching codeword"))
}

// ============================================================================
// Table 1 / H.261 — MBA (Macroblock Address Difference)
// ============================================================================

/// Symbolic MBA decode result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MbaSym {
    /// MBA difference (1..=33). Added to the previous MB address.
    Diff(u8),
    /// MBA stuffing — discard and re-decode.
    Stuffing,
}

/// MBA VLC table — Table 1/H.261. The start-code entry (16-bit zero prefix +
/// `1`) is handled separately by `start_code::find_gbsc`.
#[rustfmt::skip]
pub const MBA_TABLE: &[VlcEntry<MbaSym>] = &[
    VlcEntry::new(1,  0b1,             MbaSym::Diff(1)),
    VlcEntry::new(3,  0b011,           MbaSym::Diff(2)),
    VlcEntry::new(3,  0b010,           MbaSym::Diff(3)),
    VlcEntry::new(4,  0b0011,          MbaSym::Diff(4)),
    VlcEntry::new(4,  0b0010,          MbaSym::Diff(5)),
    VlcEntry::new(5,  0b0001_1,        MbaSym::Diff(6)),
    VlcEntry::new(5,  0b0001_0,        MbaSym::Diff(7)),
    VlcEntry::new(7,  0b0000_111,      MbaSym::Diff(8)),
    VlcEntry::new(7,  0b0000_110,      MbaSym::Diff(9)),
    VlcEntry::new(8,  0b0000_1011,     MbaSym::Diff(10)),
    VlcEntry::new(8,  0b0000_1010,     MbaSym::Diff(11)),
    VlcEntry::new(8,  0b0000_1001,     MbaSym::Diff(12)),
    VlcEntry::new(8,  0b0000_1000,     MbaSym::Diff(13)),
    VlcEntry::new(8,  0b0000_0111,     MbaSym::Diff(14)),
    VlcEntry::new(8,  0b0000_0110,     MbaSym::Diff(15)),
    VlcEntry::new(10, 0b0000_0101_11,  MbaSym::Diff(16)),
    VlcEntry::new(10, 0b0000_0101_10,  MbaSym::Diff(17)),
    VlcEntry::new(10, 0b0000_0101_01,  MbaSym::Diff(18)),
    VlcEntry::new(10, 0b0000_0101_00,  MbaSym::Diff(19)),
    VlcEntry::new(10, 0b0000_0100_11,  MbaSym::Diff(20)),
    VlcEntry::new(10, 0b0000_0100_10,  MbaSym::Diff(21)),
    VlcEntry::new(11, 0b0000_0100_011, MbaSym::Diff(22)),
    VlcEntry::new(11, 0b0000_0100_010, MbaSym::Diff(23)),
    VlcEntry::new(11, 0b0000_0100_001, MbaSym::Diff(24)),
    VlcEntry::new(11, 0b0000_0100_000, MbaSym::Diff(25)),
    VlcEntry::new(11, 0b0000_0011_111, MbaSym::Diff(26)),
    VlcEntry::new(11, 0b0000_0011_110, MbaSym::Diff(27)),
    VlcEntry::new(11, 0b0000_0011_101, MbaSym::Diff(28)),
    VlcEntry::new(11, 0b0000_0011_100, MbaSym::Diff(29)),
    VlcEntry::new(11, 0b0000_0011_011, MbaSym::Diff(30)),
    VlcEntry::new(11, 0b0000_0011_010, MbaSym::Diff(31)),
    VlcEntry::new(11, 0b0000_0011_001, MbaSym::Diff(32)),
    VlcEntry::new(11, 0b0000_0011_000, MbaSym::Diff(33)),
    VlcEntry::new(11, 0b0000_0001_111, MbaSym::Stuffing),
];

// ============================================================================
// Table 2 / H.261 — MTYPE
// ============================================================================

/// What data elements follow MTYPE and what prediction mode the MB uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtypeInfo {
    pub prediction: Prediction,
    /// MQUANT is present (5-bit FLC following MTYPE).
    pub mquant: bool,
    /// MVD is present (two VLCs, x then y).
    pub mvd: bool,
    /// CBP is present (VLC; otherwise all 6 blocks are coded).
    pub cbp: bool,
    /// TCOEFF is present for at least one block. When false, the MB is MC-only.
    pub tcoeff: bool,
    /// Loop filter applied to the motion-compensated predictor (§3.2.3).
    pub filter: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prediction {
    Intra,
    Inter,
    InterMc,
    InterMcFil,
}

macro_rules! mt {
    ($pred:ident, $mq:literal, $mvd:literal, $cbp:literal, $tc:literal, $fil:literal) => {
        MtypeInfo {
            prediction: Prediction::$pred,
            mquant: $mq,
            mvd: $mvd,
            cbp: $cbp,
            tcoeff: $tc,
            filter: $fil,
        }
    };
}

/// MTYPE VLC table — Table 2/H.261. Order matches the PDF (top row first).
#[rustfmt::skip]
pub const MTYPE_TABLE: &[VlcEntry<MtypeInfo>] = &[
    VlcEntry::new(4,  0b0001,        mt!(Intra,      false, false, false, true,  false)),
    VlcEntry::new(7,  0b0000_001,    mt!(Intra,      true,  false, false, true,  false)),
    VlcEntry::new(1,  0b1,           mt!(Inter,      false, false, true,  true,  false)),
    VlcEntry::new(5,  0b0000_1,      mt!(Inter,      true,  false, true,  true,  false)),
    VlcEntry::new(9,  0b0000_0000_1, mt!(InterMc,    false, true,  false, false, false)),
    VlcEntry::new(8,  0b0000_0001,   mt!(InterMc,    false, true,  true,  true,  false)),
    VlcEntry::new(10, 0b0000_0000_01,mt!(InterMc,    true,  true,  true,  true,  false)),
    VlcEntry::new(3,  0b001,         mt!(InterMcFil, false, true,  false, false, true)),
    VlcEntry::new(2,  0b01,          mt!(InterMcFil, false, true,  true,  true,  true)),
    VlcEntry::new(6,  0b0000_01,     mt!(InterMcFil, true,  true,  true,  true,  true)),
];

// ============================================================================
// Table 3 / H.261 — MVD (motion-vector differential)
// ============================================================================

/// Decoded MVD symbol — a raw code that maps to one of two signed values. The
/// caller picks whichever keeps the reconstructed MV in the permitted `-15..=15`
/// range (§4.2.3.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MvdSym {
    /// First representative — the value on the left of the `a & b` pair.
    pub a: i8,
    /// Second representative — `a ± 32`, guaranteed to differ from `a` in sign
    /// when both are nonzero.
    pub b: i8,
}

#[rustfmt::skip]
pub const MVD_TABLE: &[VlcEntry<MvdSym>] = &[
    VlcEntry::new(11, 0b0000_0011_001, MvdSym { a: -16, b: 16 }),
    VlcEntry::new(11, 0b0000_0011_011, MvdSym { a: -15, b: 17 }),
    VlcEntry::new(11, 0b0000_0011_101, MvdSym { a: -14, b: 18 }),
    VlcEntry::new(11, 0b0000_0011_111, MvdSym { a: -13, b: 19 }),
    VlcEntry::new(11, 0b0000_0100_001, MvdSym { a: -12, b: 20 }),
    VlcEntry::new(11, 0b0000_0100_011, MvdSym { a: -11, b: 21 }),
    VlcEntry::new(10, 0b0000_0100_11,  MvdSym { a: -10, b: 22 }),
    VlcEntry::new(10, 0b0000_0101_01,  MvdSym { a:  -9, b: 23 }),
    VlcEntry::new(10, 0b0000_0101_11,  MvdSym { a:  -8, b: 24 }),
    VlcEntry::new(8,  0b0000_0111,     MvdSym { a:  -7, b: 25 }),
    VlcEntry::new(8,  0b0000_1001,     MvdSym { a:  -6, b: 26 }),
    VlcEntry::new(8,  0b0000_1011,     MvdSym { a:  -5, b: 27 }),
    VlcEntry::new(7,  0b0000_111,      MvdSym { a:  -4, b: 28 }),
    VlcEntry::new(5,  0b0001_1,        MvdSym { a:  -3, b: 29 }),
    VlcEntry::new(4,  0b0011,          MvdSym { a:  -2, b: 30 }),
    VlcEntry::new(3,  0b011,           MvdSym { a:  -1, b: 31 }),
    VlcEntry::new(1,  0b1,             MvdSym { a:   0, b:  0 }),
    VlcEntry::new(3,  0b010,           MvdSym { a:   1, b: -31 }),
    VlcEntry::new(4,  0b0010,          MvdSym { a:   2, b: -30 }),
    VlcEntry::new(5,  0b0001_0,        MvdSym { a:   3, b: -29 }),
    VlcEntry::new(7,  0b0000_110,      MvdSym { a:   4, b: -28 }),
    VlcEntry::new(8,  0b0000_1010,     MvdSym { a:   5, b: -27 }),
    VlcEntry::new(8,  0b0000_1000,     MvdSym { a:   6, b: -26 }),
    VlcEntry::new(8,  0b0000_0110,     MvdSym { a:   7, b: -25 }),
    VlcEntry::new(10, 0b0000_0101_10,  MvdSym { a:   8, b: -24 }),
    VlcEntry::new(10, 0b0000_0101_00,  MvdSym { a:   9, b: -23 }),
    VlcEntry::new(10, 0b0000_0100_10,  MvdSym { a:  10, b: -22 }),
    VlcEntry::new(11, 0b0000_0100_010, MvdSym { a:  11, b: -21 }),
    VlcEntry::new(11, 0b0000_0100_000, MvdSym { a:  12, b: -20 }),
    VlcEntry::new(11, 0b0000_0011_110, MvdSym { a:  13, b: -19 }),
    VlcEntry::new(11, 0b0000_0011_100, MvdSym { a:  14, b: -18 }),
    VlcEntry::new(11, 0b0000_0011_010, MvdSym { a:  15, b: -17 }),
];

// ============================================================================
// Table 4 / H.261 — CBP (coded block pattern)
// ============================================================================

#[rustfmt::skip]
pub const CBP_TABLE: &[VlcEntry<u8>] = &[
    VlcEntry::new(3,  0b111,       60),
    VlcEntry::new(4,  0b1101,       4),
    VlcEntry::new(4,  0b1100,       8),
    VlcEntry::new(4,  0b1011,      16),
    VlcEntry::new(4,  0b1010,      32),
    VlcEntry::new(5,  0b1001_1,    12),
    VlcEntry::new(5,  0b1001_0,    48),
    VlcEntry::new(5,  0b1000_1,    20),
    VlcEntry::new(5,  0b1000_0,    40),
    VlcEntry::new(5,  0b0111_1,    28),
    VlcEntry::new(5,  0b0111_0,    44),
    VlcEntry::new(5,  0b0110_1,    52),
    VlcEntry::new(5,  0b0110_0,    56),
    VlcEntry::new(5,  0b0101_1,     1),
    VlcEntry::new(5,  0b0101_0,    61),
    VlcEntry::new(5,  0b0100_1,     2),
    VlcEntry::new(5,  0b0100_0,    62),
    VlcEntry::new(6,  0b0011_11,   24),
    VlcEntry::new(6,  0b0011_10,   36),
    VlcEntry::new(6,  0b0011_01,    3),
    VlcEntry::new(6,  0b0011_00,   63),
    VlcEntry::new(7,  0b0010_111,   5),
    VlcEntry::new(7,  0b0010_110,   9),
    VlcEntry::new(7,  0b0010_101,  17),
    VlcEntry::new(7,  0b0010_100,  33),
    VlcEntry::new(7,  0b0010_011,   6),
    VlcEntry::new(7,  0b0010_010,  10),
    VlcEntry::new(7,  0b0010_001,  18),
    VlcEntry::new(7,  0b0010_000,  34),
    VlcEntry::new(8,  0b0001_1111,  7),
    VlcEntry::new(8,  0b0001_1110, 11),
    VlcEntry::new(8,  0b0001_1101, 19),
    VlcEntry::new(8,  0b0001_1100, 35),
    VlcEntry::new(8,  0b0001_1011, 13),
    VlcEntry::new(8,  0b0001_1010, 49),
    VlcEntry::new(8,  0b0001_1001, 21),
    VlcEntry::new(8,  0b0001_1000, 41),
    VlcEntry::new(8,  0b0001_0111, 14),
    VlcEntry::new(8,  0b0001_0110, 50),
    VlcEntry::new(8,  0b0001_0101, 22),
    VlcEntry::new(8,  0b0001_0100, 42),
    VlcEntry::new(8,  0b0001_0011, 15),
    VlcEntry::new(8,  0b0001_0010, 51),
    VlcEntry::new(8,  0b0001_0001, 23),
    VlcEntry::new(8,  0b0001_0000, 43),
    VlcEntry::new(8,  0b0000_1111, 25),
    VlcEntry::new(8,  0b0000_1110, 37),
    VlcEntry::new(8,  0b0000_1101, 26),
    VlcEntry::new(8,  0b0000_1100, 38),
    VlcEntry::new(8,  0b0000_1011, 29),
    VlcEntry::new(8,  0b0000_1010, 45),
    VlcEntry::new(8,  0b0000_1001, 53),
    VlcEntry::new(8,  0b0000_1000, 57),
    VlcEntry::new(8,  0b0000_0111, 30),
    VlcEntry::new(8,  0b0000_0110, 46),
    VlcEntry::new(8,  0b0000_0101, 54),
    VlcEntry::new(8,  0b0000_0100, 58),
    VlcEntry::new(9,  0b0000_0011_1, 31),
    VlcEntry::new(9,  0b0000_0011_0, 47),
    VlcEntry::new(9,  0b0000_0010_1, 55),
    VlcEntry::new(9,  0b0000_0010_0, 59),
    VlcEntry::new(9,  0b0000_0001_1, 27),
    VlcEntry::new(9,  0b0000_0001_0, 39),
];

// ============================================================================
// Table 5 / H.261 — TCOEFF (transform coefficient run/level)
// ============================================================================

/// Decoded TCOEFF VLC result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TcoeffSym {
    /// End of block.
    Eob,
    /// A (run, |level|) pair. A 1-bit sign is read from the bitstream after
    /// the VLC is matched (the sign `s` in `"0100 s"` etc.).
    RunLevel { run: u8, level_abs: u8 },
    /// Escape — followed by 6-bit run, 8-bit signed level (no trailing sign).
    Escape,
}

/// TCOEFF VLC entries for all (run, level) pairs *except* the "first
/// coefficient" `1s` code and EOB/Escape. These are handled specially by
/// [`decode_tcoeff`].
///
/// Each entry's VLC is the code **before** the trailing sign bit `s`.
/// The `bits` field is the length of that code (so total bits consumed =
/// `bits + 1` for the trailing sign bit).
#[rustfmt::skip]
const TCOEFF_ENTRIES: &[VlcEntry<(u8, u8)>] = &[
    // (0,2) = 0100 s — 4 bits.
    VlcEntry::new(4,  0b0100,               (0, 2)),
    // (2,1) = 0101 s — 4 bits.
    VlcEntry::new(4,  0b0101,               (2, 1)),
    // (1,1) = 011 s — 3 bits.
    VlcEntry::new(3,  0b011,                (1, 1)),
    // (0,3) = 0010 1s — 5 bits.
    VlcEntry::new(5,  0b0010_1,             (0, 3)),
    // (3,1) = 0011 1s — 5 bits.
    VlcEntry::new(5,  0b0011_1,             (3, 1)),
    // (4,1) = 0011 0s — 5 bits.
    VlcEntry::new(5,  0b0011_0,             (4, 1)),
    // (5,1) = 0001 11s — 6 bits. (Spec Table 5; 6 bits before sign, not 5.)
    VlcEntry::new(6,  0b0001_11,            (5, 1)),
    // (6,1) = 0001 01s — 6 bits.
    VlcEntry::new(6,  0b0001_01,            (6, 1)),
    // (7,1) = 0001 00s — 6 bits.
    VlcEntry::new(6,  0b0001_00,            (7, 1)),
    // (1,2) = 0001 10s — 6 bits.
    VlcEntry::new(6,  0b0001_10,            (1, 2)),
    // (0,4) = 0000 110s — 7 bits.
    VlcEntry::new(7,  0b0000_110,           (0, 4)),
    // (8,1) = 0000 111s — 7 bits.
    VlcEntry::new(7,  0b0000_111,           (8, 1)),
    // (9,1) = 0000 101s — 7 bits.
    VlcEntry::new(7,  0b0000_101,           (9, 1)),
    // (2,2) = 0000 100s — 7 bits.
    VlcEntry::new(7,  0b0000_100,           (2, 2)),
    // (0,5) = 0010 0110 s — 8 bits.
    VlcEntry::new(8,  0b0010_0110,          (0, 5)),
    // (0,6) = 0010 0001 s — 8 bits.
    VlcEntry::new(8,  0b0010_0001,          (0, 6)),
    // (1,3) = 0010 0101 s — 8 bits.
    VlcEntry::new(8,  0b0010_0101,          (1, 3)),
    // (3,2) = 0010 0100 s — 8 bits.
    VlcEntry::new(8,  0b0010_0100,          (3, 2)),
    // (10,1) = 0010 0111 s — 8 bits.
    VlcEntry::new(8,  0b0010_0111,          (10, 1)),
    // (11,1) = 0010 0011 s — 8 bits.
    VlcEntry::new(8,  0b0010_0011,          (11, 1)),
    // (12,1) = 0010 0010 s — 8 bits.
    VlcEntry::new(8,  0b0010_0010,          (12, 1)),
    // (13,1) = 0010 0000 s — 8 bits.
    VlcEntry::new(8,  0b0010_0000,          (13, 1)),
    // (0,7) = 0000 0010 10s — 10 bits.
    VlcEntry::new(10, 0b0000_0010_10,       (0, 7)),
    // (4,2) = 0000 0011 11s — 10 bits.
    VlcEntry::new(10, 0b0000_0011_11,       (4, 2)),
    // (2,3) = 0000 0010 11s — 10 bits.
    VlcEntry::new(10, 0b0000_0010_11,       (2, 3)),
    // (5,2) = 0000 0010 01s — 10 bits.
    VlcEntry::new(10, 0b0000_0010_01,       (5, 2)),
    // (1,4) = 0000 0011 00s — 10 bits.
    VlcEntry::new(10, 0b0000_0011_00,       (1, 4)),
    // (14,1) = 0000 0011 10s — 10 bits.
    VlcEntry::new(10, 0b0000_0011_10,       (14, 1)),
    // (15,1) = 0000 0011 01s — 10 bits.
    VlcEntry::new(10, 0b0000_0011_01,       (15, 1)),
    // (16,1) = 0000 0010 00s — 10 bits.
    VlcEntry::new(10, 0b0000_0010_00,       (16, 1)),
    // (0,8) = 0000 0001 1101 s — 12 bits.
    VlcEntry::new(12, 0b0000_0001_1101,     (0, 8)),
    // (0,9) = 0000 0001 1000 s — 12 bits.
    VlcEntry::new(12, 0b0000_0001_1000,     (0, 9)),
    // (0,10) = 0000 0001 0011 s — 12 bits.
    VlcEntry::new(12, 0b0000_0001_0011,     (0, 10)),
    // (0,11) = 0000 0001 0000 s — 12 bits.
    VlcEntry::new(12, 0b0000_0001_0000,     (0, 11)),
    // (1,5) = 0000 0001 1011 s — 12 bits.
    VlcEntry::new(12, 0b0000_0001_1011,     (1, 5)),
    // (2,4) = 0000 0001 0100 s — 12 bits.
    VlcEntry::new(12, 0b0000_0001_0100,     (2, 4)),
    // (3,3) = 0000 0001 1100 s — 12 bits.
    VlcEntry::new(12, 0b0000_0001_1100,     (3, 3)),
    // (4,3) = 0000 0001 0010 s — 12 bits.
    VlcEntry::new(12, 0b0000_0001_0010,     (4, 3)),
    // (6,2) = 0000 0001 1110 s — 12 bits.
    VlcEntry::new(12, 0b0000_0001_1110,     (6, 2)),
    // (7,2) = 0000 0001 0101 s — 12 bits.
    VlcEntry::new(12, 0b0000_0001_0101,     (7, 2)),
    // (8,2) = 0000 0001 0001 s — 12 bits.
    VlcEntry::new(12, 0b0000_0001_0001,     (8, 2)),
    // (17,1) = 0000 0001 1111 s — 12 bits.
    VlcEntry::new(12, 0b0000_0001_1111,     (17, 1)),
    // (18,1) = 0000 0001 1010 s — 12 bits.
    VlcEntry::new(12, 0b0000_0001_1010,     (18, 1)),
    // (19,1) = 0000 0001 1001 s — 12 bits.
    VlcEntry::new(12, 0b0000_0001_1001,     (19, 1)),
    // (20,1) = 0000 0001 0111 s — 12 bits.
    VlcEntry::new(12, 0b0000_0001_0111,     (20, 1)),
    // (21,1) = 0000 0001 0110 s — 12 bits.
    VlcEntry::new(12, 0b0000_0001_0110,     (21, 1)),
    // (0,12) = 0000 0000 1101 0s — 13 bits code before sign.
    VlcEntry::new(13, 0b0_0000_0000_1101_0, (0, 12)),
    // (0,13) = 0000 0000 1100 1s — 13 bits code before sign.
    VlcEntry::new(13, 0b0_0000_0000_1100_1, (0, 13)),
    // (0,14) = 0000 0000 1100 0s — 13 bits.
    VlcEntry::new(13, 0b0_0000_0000_1100_0, (0, 14)),
    // (0,15) = 0000 0000 1011 1s — 13 bits.
    VlcEntry::new(13, 0b0_0000_0000_1011_1, (0, 15)),
    // (1,6) = 0000 0000 1011 0s — 13 bits.
    VlcEntry::new(13, 0b0_0000_0000_1011_0, (1, 6)),
    // (1,7) = 0000 0000 1010 1s — 13 bits.
    VlcEntry::new(13, 0b0_0000_0000_1010_1, (1, 7)),
    // (2,5) = 0000 0000 1010 0s — 13 bits.
    VlcEntry::new(13, 0b0_0000_0000_1010_0, (2, 5)),
    // (3,4) = 0000 0000 1001 1s — 13 bits.
    VlcEntry::new(13, 0b0_0000_0000_1001_1, (3, 4)),
    // (5,3) = 0000 0000 1001 0s — 13 bits.
    VlcEntry::new(13, 0b0_0000_0000_1001_0, (5, 3)),
    // (9,2) = 0000 0000 1000 1s — 13 bits.
    VlcEntry::new(13, 0b0_0000_0000_1000_1, (9, 2)),
    // (10,2) = 0000 0000 1000 0s — 13 bits.
    VlcEntry::new(13, 0b0_0000_0000_1000_0, (10, 2)),
    // (22,1) = 0000 0000 1111 1s — 13 bits.
    VlcEntry::new(13, 0b0_0000_0000_1111_1, (22, 1)),
    // (23,1) = 0000 0000 1111 0s — 13 bits.
    VlcEntry::new(13, 0b0_0000_0000_1111_0, (23, 1)),
    // (24,1) = 0000 0000 1110 1s — 13 bits.
    VlcEntry::new(13, 0b0_0000_0000_1110_1, (24, 1)),
    // (25,1) = 0000 0000 1110 0s — 13 bits.
    VlcEntry::new(13, 0b0_0000_0000_1110_0, (25, 1)),
    // (26,1) = 0000 0000 1101 1s — 13 bits.
    VlcEntry::new(13, 0b0_0000_0000_1101_1, (26, 1)),
];

/// Decode a single TCOEFF symbol given the reader and whether this is the
/// first coefficient of the block.
///
/// The caller is responsible for reading the escape's `(run, signed-level)`
/// payload and the sign bit on `RunLevel`.
pub fn decode_tcoeff(br: &mut BitReader<'_>, is_first: bool) -> Result<TcoeffSym> {
    // Special cases:
    // * EOB = `10`          — not allowed as first coefficient.
    // * `1s` = (0,1) first  — only as first coefficient.
    // * `11s` = (0,1) later — only when not first.
    // * Escape = `0000 01`.

    // Peek a wide prefix.
    let avail = br.bits_remaining().min(20) as u32;
    if avail == 0 {
        return Err(Error::invalid("h261 tcoeff: no bits"));
    }
    let peek = br.peek_u32(avail)?;
    let b0 = (peek >> (avail - 1)) & 1;
    if b0 == 1 {
        if is_first {
            // `1s` — (0,1), sign read by caller after we consume the leading `1`.
            br.consume(1)?;
            return Ok(TcoeffSym::RunLevel {
                run: 0,
                level_abs: 1,
            });
        }
        // Two-bit discriminator.
        let two = (peek >> (avail - 2)) & 0b11;
        if two == 0b10 {
            br.consume(2)?;
            return Ok(TcoeffSym::Eob);
        } else {
            // `11s` — (0,1) subsequent.
            br.consume(2)?;
            return Ok(TcoeffSym::RunLevel {
                run: 0,
                level_abs: 1,
            });
        }
    }
    // First bit is 0 — try the escape `000001` before VLC table lookup since
    // the escape collides with "no 0000 0001..." entries but the escape has
    // exactly 6 bits so a direct check is cleanest.
    if avail >= 6 {
        let six = (peek >> (avail - 6)) & 0x3F;
        if six == 0b0000_01 {
            br.consume(6)?;
            return Ok(TcoeffSym::Escape);
        }
    }
    // Fall through to the general VLC table scan.
    let sym = decode_vlc(br, TCOEFF_ENTRIES)?;
    Ok(TcoeffSym::RunLevel {
        run: sym.0,
        level_abs: sym.1,
    })
}

// ============================================================================
// Figure 12 / H.261 — Zig-zag scan
// ============================================================================

/// Zig-zag scan order (Figure 12/H.261). `ZIGZAG[i]` gives the raster
/// (row-major) position of the i-th coefficient in transmission order.
///
/// Verified against Figure 12: transmission order 1..64 begins at position
/// (0,0), goes right to (0,1), then down-left diagonal, etc.
#[rustfmt::skip]
pub const ZIGZAG: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mba_diff_1_is_one_bit() {
        let data = [0b1000_0000u8];
        let mut br = BitReader::new(&data);
        assert_eq!(decode_vlc(&mut br, MBA_TABLE).unwrap(), MbaSym::Diff(1));
    }

    #[test]
    fn mba_diff_2_is_011() {
        let data = [0b0110_0000u8];
        let mut br = BitReader::new(&data);
        assert_eq!(decode_vlc(&mut br, MBA_TABLE).unwrap(), MbaSym::Diff(2));
    }

    #[test]
    fn mtype_intra_1bit_vs_inter() {
        let data = [0b0001_0000u8];
        let mut br = BitReader::new(&data);
        let v = decode_vlc(&mut br, MTYPE_TABLE).unwrap();
        assert_eq!(v.prediction, Prediction::Intra);
        assert!(!v.mquant);
        let data = [0b1000_0000u8];
        let mut br = BitReader::new(&data);
        let v = decode_vlc(&mut br, MTYPE_TABLE).unwrap();
        assert_eq!(v.prediction, Prediction::Inter);
        assert!(v.cbp);
    }

    #[test]
    fn cbp_basic() {
        // CBP 60 -> 111.
        let data = [0b1110_0000u8];
        let mut br = BitReader::new(&data);
        assert_eq!(decode_vlc(&mut br, CBP_TABLE).unwrap(), 60);
    }

    #[test]
    fn mvd_zero() {
        // MVD code `1` -> (0, 0).
        let data = [0b1000_0000u8];
        let mut br = BitReader::new(&data);
        let v = decode_vlc(&mut br, MVD_TABLE).unwrap();
        assert_eq!(v.a, 0);
    }

    #[test]
    fn tcoeff_first_one() {
        // "1s" first-position code: read a `1` bit — gives (0,1). Caller reads sign.
        let data = [0b1_000_0000u8];
        let mut br = BitReader::new(&data);
        assert_eq!(
            decode_tcoeff(&mut br, true).unwrap(),
            TcoeffSym::RunLevel {
                run: 0,
                level_abs: 1
            }
        );
    }

    #[test]
    fn tcoeff_eob() {
        // "10" EOB.
        let data = [0b1000_0000u8];
        let mut br = BitReader::new(&data);
        assert_eq!(decode_tcoeff(&mut br, false).unwrap(), TcoeffSym::Eob);
    }

    #[test]
    fn tcoeff_subsequent_one() {
        // "11s" — subsequent (0,1).
        let data = [0b1100_0000u8];
        let mut br = BitReader::new(&data);
        assert_eq!(
            decode_tcoeff(&mut br, false).unwrap(),
            TcoeffSym::RunLevel {
                run: 0,
                level_abs: 1
            }
        );
    }

    #[test]
    fn tcoeff_0_2() {
        // "0100 s" — (0,2).
        let data = [0b0100_0000u8];
        let mut br = BitReader::new(&data);
        assert_eq!(
            decode_tcoeff(&mut br, false).unwrap(),
            TcoeffSym::RunLevel {
                run: 0,
                level_abs: 2
            }
        );
    }

    #[test]
    fn tcoeff_escape() {
        // "000001" — escape.
        let data = [0b0000_0100u8];
        let mut br = BitReader::new(&data);
        assert_eq!(decode_tcoeff(&mut br, false).unwrap(), TcoeffSym::Escape);
    }

    #[test]
    fn zigzag_starts_natural() {
        assert_eq!(ZIGZAG[0], 0);
        assert_eq!(ZIGZAG[1], 1);
        assert_eq!(ZIGZAG[2], 8);
    }
}
