//! H.261 RTP payload-format packing and unpacking — RFC 4587.
//!
//! RFC 4587 specifies how to carry an H.261 elementary bitstream over RTP.
//! Each RTP packet's payload starts with a 4-byte H.261 payload header
//! that allows independent decoding of the macroblocks (MBs) inside,
//! followed by an arbitrary-bit-length slice of the H.261 bitstream that
//! starts and ends on an MB boundary.
//!
//! This module implements the **payload format itself** — packing an
//! H.261 elementary stream into a sequence of RTP-shaped payloads and
//! unpacking it back. The lower-level `packetize_gob_aligned` /
//! `depacketize` pair handles just the §4.1 4-byte H.261 payload header
//! and the GOB-aligned payload slice. The higher-level [`RtpPacketizer`]
//! glue wraps each payload in a full RFC 3550 §5.1 RTP fixed header (V,
//! P, X, CC, M, PT, sequence number, timestamp, SSRC) so callers can
//! hand its output straight to UDP / DTLS / SRTP. UDP/TCP transport,
//! RTCP, and SDP are still caller-side concerns.
//!
//! ```text
//!  0                   1                   2                   3
//!   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//!  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//!  |SBIT |EBIT |I|V| GOBN  |   MBAP  |  QUANT  |  HMVD   |  VMVD   |
//!  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! ```
//!
//! Header fields (RFC 4587 §4.1):
//!
//! * **SBIT** (3 bits): number of MSB ignored in the first payload byte.
//! * **EBIT** (3 bits): number of LSB ignored in the last payload byte.
//! * **I**    (1 bit):  hint — set if the stream contains only INTRA MBs.
//! * **V**    (1 bit):  hint — set if motion vectors may be used.
//! * **GOBN** (4 bits): GOB number in effect at the start of the packet;
//!   `0` if the packet begins with a GOB header.
//! * **MBAP** (5 bits): MBA predictor, biased -1 (`0` ⇒ predictor=1, `1` ⇒
//!   predictor=2, …); `0` if packet begins with a GOB header.
//! * **QUANT** (5 bits): QUANT value in effect prior to the start of this
//!   packet (`0` if the packet begins with a GOB).
//! * **HMVD** (5 bits): reference horizontal MVD, two's complement;
//!   `'10000'` (-16) is forbidden.
//! * **VMVD** (5 bits): reference vertical MVD, two's complement;
//!   `'10000'` (-16) MUST NOT be used.
//!
//! ## What the module provides
//!
//! * [`H261RtpHeader`] — typed view of the 4-byte payload header.
//! * [`pack_header`] / [`unpack_header`] — bit-exact (de)serialisation.
//! * [`packetize_gob_aligned`] — split an H.261 elementary stream into
//!   RTP payloads at GOB boundaries (the cheap, hardware-friendly path
//!   per §4.2 of RFC 4587). Each payload starts at a GOB header
//!   (GOBN=0, MBAP=0, QUANT=0, HMVD=VMVD=0); SBIT/EBIT are zero because
//!   GOB headers are bit-byte-aligned by construction in the elementary
//!   stream emitted by this crate's encoder.
//! * [`packetize_mb_fragmented`] — the §4.2 RECOMMENDED MB-level
//!   fragmenting packetizer. GOBs that exceed the payload budget are
//!   parsed at the Huffman layer ("it is not necessary to decompress
//!   the stream fully") to locate macroblock boundaries; each
//!   continuation packet starts on an MB boundary with non-zero
//!   SBIT/EBIT and carries the §4.1 GOBN / MBAP / QUANT / HMVD / VMVD
//!   context in effect at its first bit, so a receiver can resume
//!   decoding mid-GOB after losing the preceding packet. Per §3.2 "an
//!   MB cannot be split across multiple packets" and "the bit stream
//!   cannot be fragmented between a GOB header and MB 1 of that GOB".
//! * [`depacketize`] — reassemble an elementary stream from a sequence
//!   of payloads, honouring per-packet SBIT/EBIT and concatenating the
//!   inner bits to a single byte buffer.
//!
//! ## What it does **not** do
//!
//! * RTCP, SDP offer/answer, payload-type negotiation. The PT, SSRC,
//!   initial sequence number, and per-frame RTP timestamp are passed in
//!   by the caller from its SDP / clock state.
//!
//! ## RFC 4587 §4.1 caveat
//!
//! RFC 4587 says explicitly: *"The H.261 stream SHALL be used without BCH
//! error correction and without error correction framing."* The
//! [`crate::bch`] module is for direct ISDN p×64 kbit/s transport, not
//! for RTP. The two modules are mutually exclusive consumers of an H.261
//! elementary stream.

use oxideav_core::bits::BitReader;

use crate::gob::parse_gob_header;
use crate::mb::{decode_mba_diff, reconstruct_mv, MbContext};
use crate::picture::parse_picture_header;
use crate::start_code::{find_next_start_code_bits, iter_start_codes, GN_PICTURE};
use crate::tables::{
    decode_tcoeff, decode_vlc, MtypeInfo, MvdSym, Prediction, TcoeffSym, CBP_TABLE, MTYPE_TABLE,
    MVD_TABLE,
};

/// Length of the H.261 RTP payload header in bytes (RFC 4587 §4.1).
pub const HEADER_LEN: usize = 4;

/// Typed view of the 4-byte H.261 RTP payload header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H261RtpHeader {
    /// Start-bit position: most-significant bits to ignore in the first
    /// payload byte. 3-bit field, valid range 0..=7.
    pub sbit: u8,
    /// End-bit position: least-significant bits to ignore in the last
    /// payload byte. 3-bit field, valid range 0..=7.
    pub ebit: u8,
    /// `I` flag — stream contains only INTRA-coded MBs.
    pub intra_only: bool,
    /// `V` flag — motion vectors may be used somewhere in the stream.
    pub motion_vectors: bool,
    /// GOB number in effect at the start of the packet, or `0` if the
    /// packet begins with a GOB header. 4-bit field, valid 0..=15.
    pub gobn: u8,
    /// Macroblock-address predictor, biased -1 (`0` ⇒ predictor=1, `31`
    /// ⇒ predictor=32). `0` if the packet begins with a GOB header.
    /// 5-bit field.
    pub mbap: u8,
    /// QUANT value (GQUANT or MQUANT) in effect prior to the start of
    /// this packet; `0` if the packet begins with a GOB header. 5-bit
    /// field, legal range otherwise 1..=31.
    pub quant: u8,
    /// Reference horizontal MVD (2's complement, 5-bit field).
    /// `-16` (`0b10000`) is forbidden by RFC 4587 §4.1.
    pub hmvd: i8,
    /// Reference vertical MVD (2's complement, 5-bit field).
    /// `-16` (`0b10000`) is forbidden by RFC 4587 §4.1.
    pub vmvd: i8,
}

impl H261RtpHeader {
    /// Build a header for a packet that begins with a GOB header.
    ///
    /// RFC 4587 §4.1 requires `GOBN = MBAP = QUANT = HMVD = VMVD = 0`
    /// when the packet starts on a GOB boundary. The encoded payload is
    /// also expected to begin on a byte boundary (GBSC is byte-aligned
    /// in the H.261 elementary stream our encoder emits), so `SBIT = 0`.
    /// `EBIT` is set by the packetizer when it knows how many trailing
    /// padding bits this packet carries.
    pub fn gob_aligned(ebit: u8, intra_only: bool, motion_vectors: bool) -> Self {
        Self {
            sbit: 0,
            ebit,
            intra_only,
            motion_vectors,
            gobn: 0,
            mbap: 0,
            quant: 0,
            hmvd: 0,
            vmvd: 0,
        }
    }
}

/// Encoding / decoding error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RtpError {
    /// Header buffer is shorter than 4 bytes.
    ShortHeader,
    /// SBIT or EBIT is outside the 0..=7 range encodable in 3 bits.
    BadBitOffset { field: &'static str, value: u8 },
    /// HMVD or VMVD encoded the forbidden `-16` value (`'10000'`),
    /// which RFC 4587 §4.1 says SHALL NOT be used.
    ForbiddenMvd { field: &'static str },
    /// A field exceeds its valid bit-width.
    FieldOverflow { field: &'static str, value: u32 },
    /// Payload is empty (the H.261 payload after the 4-byte header
    /// must be at least 1 byte; otherwise the encoder shouldn't have
    /// emitted the packet at all).
    EmptyPayload,
    /// The depacketizer found no GOB or PSC start codes in the
    /// reconstructed stream — the packet sequence is non-recoverable.
    NoStartCodes,
    /// The MB-level fragmenter could not parse the elementary stream's
    /// Huffman layer. RFC 4587 §4.2 requires parsing the variable-length
    /// codes (not a full decode) to locate macroblock boundaries; a
    /// stream that fails that parse cannot be MB-fragmented.
    MalformedStream {
        /// Parse-failure detail from the VLC layer.
        detail: String,
    },
    /// A single unfragmentable unit — a picture header, a GOB header +
    /// MB 1, or one macroblock run — needs more bytes than the per-packet
    /// payload budget allows, so no MB-boundary split exists (RFC 4587
    /// §3.2: "an MB cannot be split across multiple packets").
    FragmentTooLarge {
        /// Bytes the smallest legal fragment would need.
        needed: usize,
        /// Maximum data bytes available per packet after the 4-byte header.
        max: usize,
    },
}

impl core::fmt::Display for RtpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RtpError::ShortHeader => write!(f, "h261 RTP header < 4 bytes"),
            RtpError::BadBitOffset { field, value } => {
                write!(f, "h261 RTP: {field}={value} out of 0..=7")
            }
            RtpError::ForbiddenMvd { field } => {
                write!(
                    f,
                    "h261 RTP: {field} encodes the forbidden -16 (RFC 4587 §4.1)"
                )
            }
            RtpError::FieldOverflow { field, value } => {
                write!(f, "h261 RTP: {field}={value} exceeds field width")
            }
            RtpError::EmptyPayload => write!(f, "h261 RTP: empty payload"),
            RtpError::NoStartCodes => {
                write!(f, "h261 RTP: depacketized stream contains no start codes")
            }
            RtpError::MalformedStream { detail } => {
                write!(f, "h261 RTP: MB fragmenter VLC parse failed: {detail}")
            }
            RtpError::FragmentTooLarge { needed, max } => {
                write!(
                    f,
                    "h261 RTP: smallest legal fragment needs {needed} bytes but the payload budget is {max}"
                )
            }
        }
    }
}

impl std::error::Error for RtpError {}

/// Pack an [`H261RtpHeader`] into 4 bytes (RFC 4587 §4.1 wire layout).
///
/// Returns `Err` if any field overflows its bit-width, or if HMVD/VMVD
/// encodes `-16` (the spec's forbidden value).
pub fn pack_header(h: &H261RtpHeader) -> Result<[u8; HEADER_LEN], RtpError> {
    if h.sbit > 7 {
        return Err(RtpError::BadBitOffset {
            field: "SBIT",
            value: h.sbit,
        });
    }
    if h.ebit > 7 {
        return Err(RtpError::BadBitOffset {
            field: "EBIT",
            value: h.ebit,
        });
    }
    if h.gobn > 15 {
        return Err(RtpError::FieldOverflow {
            field: "GOBN",
            value: h.gobn as u32,
        });
    }
    if h.mbap > 31 {
        return Err(RtpError::FieldOverflow {
            field: "MBAP",
            value: h.mbap as u32,
        });
    }
    if h.quant > 31 {
        return Err(RtpError::FieldOverflow {
            field: "QUANT",
            value: h.quant as u32,
        });
    }
    // MVD is a 5-bit 2's-complement field, range -15..=15, with -16
    // (`0b10000`) explicitly forbidden by RFC 4587 §4.1.
    if h.hmvd < -15 || h.hmvd > 15 {
        return Err(RtpError::ForbiddenMvd { field: "HMVD" });
    }
    if h.vmvd < -15 || h.vmvd > 15 {
        return Err(RtpError::ForbiddenMvd { field: "VMVD" });
    }

    // Pack as a single 32-bit big-endian word for clarity, then split.
    let hmvd5 = (h.hmvd as i32 & 0x1F) as u32;
    let vmvd5 = (h.vmvd as i32 & 0x1F) as u32;
    let word: u32 = ((h.sbit as u32) << 29)
        | ((h.ebit as u32) << 26)
        | ((h.intra_only as u32) << 25)
        | ((h.motion_vectors as u32) << 24)
        | ((h.gobn as u32) << 20)
        | ((h.mbap as u32) << 15)
        | ((h.quant as u32) << 10)
        | (hmvd5 << 5)
        | vmvd5;
    Ok(word.to_be_bytes())
}

/// Decode the 4-byte H.261 RTP payload header from the front of `buf`.
///
/// Returns the parsed header and a slice pointing to the remaining
/// payload bytes (the H.261 stream itself).
pub fn unpack_header(buf: &[u8]) -> Result<(H261RtpHeader, &[u8]), RtpError> {
    if buf.len() < HEADER_LEN {
        return Err(RtpError::ShortHeader);
    }
    let word = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let sbit = ((word >> 29) & 0x7) as u8;
    let ebit = ((word >> 26) & 0x7) as u8;
    let intra_only = ((word >> 25) & 0x1) != 0;
    let motion_vectors = ((word >> 24) & 0x1) != 0;
    let gobn = ((word >> 20) & 0xF) as u8;
    let mbap = ((word >> 15) & 0x1F) as u8;
    let quant = ((word >> 10) & 0x1F) as u8;
    let hmvd5 = ((word >> 5) & 0x1F) as u8;
    let vmvd5 = (word & 0x1F) as u8;

    // Sign-extend 5-bit 2's complement to i8.
    let sign_extend5 = |v: u8| -> i8 {
        if v & 0x10 != 0 {
            (v as i8) | !0x1F
        } else {
            v as i8
        }
    };
    let hmvd = sign_extend5(hmvd5);
    let vmvd = sign_extend5(vmvd5);

    // §4.1: -16 is forbidden in either MV field. We accept it on
    // unpack (so a tolerant decoder can still inspect a malformed
    // packet) but pack_header refuses to emit it. Callers wanting a
    // strict check can compare `hmvd == -16 || vmvd == -16` themselves.

    Ok((
        H261RtpHeader {
            sbit,
            ebit,
            intra_only,
            motion_vectors,
            gobn,
            mbap,
            quant,
            hmvd,
            vmvd,
        },
        &buf[HEADER_LEN..],
    ))
}

/// One RTP-shaped payload as emitted by [`packetize_gob_aligned`].
///
/// `bytes` is the full payload: the 4-byte H.261 RTP header followed by
/// the inner H.261 bitstream slice. Pass `bytes` straight to your RTP
/// transport layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H261RtpPayload {
    /// The 4-byte H.261 RTP payload header.
    pub header: H261RtpHeader,
    /// Full payload (header + data) as a single byte buffer.
    pub bytes: Vec<u8>,
    /// Marker-bit hint: `true` on the last packet of a video frame. The
    /// caller is responsible for setting this on the RTP fixed header.
    pub marker: bool,
}

impl H261RtpPayload {
    /// Length of the inner H.261 data after the 4-byte header.
    pub fn data_len(&self) -> usize {
        self.bytes.len().saturating_sub(HEADER_LEN)
    }
}

/// Packetize an H.261 elementary stream at GOB boundaries (RFC 4587
/// §4.2 "cheap packetization").
///
/// `data` is the H.261 elementary stream as emitted by
/// [`crate::encoder::encode_intra_picture`] / `encode_inter_picture` —
/// PSC, GOBs, MB data, all byte-aligned at the start by construction.
/// `max_payload` is the maximum size, in bytes, of each emitted payload
/// **including** the 4-byte RTP header (use your effective MTU minus
/// the RTP fixed-header size of typically 12 bytes).
///
/// The packetizer scans `data` for start codes (PSC or GBSC); every
/// payload it emits starts at a start-code byte boundary and ends just
/// before the next start code, the end of the data, or the
/// `max_payload` cap — whichever is first. The last payload of a video
/// frame is flagged with `marker = true`.
///
/// Returns an empty vector if `data` contains no start codes.
///
/// # Panics
///
/// Panics if `max_payload < HEADER_LEN + 1`. A meaningful packet must
/// carry at least one byte of H.261 data after its 4-byte header.
pub fn packetize_gob_aligned(
    data: &[u8],
    max_payload: usize,
    intra_only: bool,
    motion_vectors: bool,
) -> Vec<H261RtpPayload> {
    assert!(
        max_payload > HEADER_LEN,
        "max_payload must accommodate the 4-byte H.261 RTP header + at least 1 data byte"
    );

    let mut payloads = Vec::new();

    // Find every start code in the stream. The boundaries between
    // payloads sit at the byte-position of each start code; we always
    // emit a fresh payload aligned to a start code so the receiver can
    // re-sync from any packet on its own.
    let starts: Vec<usize> = iter_start_codes(data)
        .filter_map(|sc| {
            // Only byte-aligned start codes are usable as packet
            // boundaries. The encoder always emits byte-aligned GBSC/PSC,
            // but if a caller feeds in concatenated streams the scanner
            // might pick up a bit-misaligned tail. Skip those.
            if sc.bit_pos % 8 == 0 {
                Some(sc.byte_pos)
            } else {
                None
            }
        })
        .collect();
    if starts.is_empty() {
        return payloads;
    }

    // Track picture boundaries to set the RTP marker bit on the last
    // payload of each frame. A start code with gn == GN_PICTURE (0) is
    // a PSC.
    let psc_byte_positions: Vec<usize> = iter_start_codes(data)
        .filter(|sc| sc.bit_pos % 8 == 0 && sc.gn == GN_PICTURE)
        .map(|sc| sc.byte_pos)
        .collect();

    let max_data = max_payload - HEADER_LEN;
    let total = data.len();

    // For each start-code-aligned boundary, build payload(s) up to but
    // not including the next boundary; if the inter-boundary chunk is
    // larger than `max_data`, split it at byte boundaries (this is a
    // GOB-too-large fallback; SBIT/EBIT are still zero because we slice
    // at byte boundaries only).
    for (idx, &begin) in starts.iter().enumerate() {
        let end = starts.get(idx + 1).copied().unwrap_or(total);
        let mut p = begin;
        let mut first_chunk_of_gob = true;
        while p < end {
            let chunk_len = (end - p).min(max_data);
            let chunk_end = p + chunk_len;

            // GOBN/MBAP/QUANT/HMVD/VMVD are 0 only at the start of a GOB.
            // Sub-chunks of a fragmented GOB don't get those zeros —
            // they'd need MBA-state continuation that the cheap
            // packetizer doesn't track. For now we surface a flag for
            // the caller to know not to expect independent decodability
            // of those sub-packets. The receiver still gets all the
            // bytes back through depacketize().
            let mut hdr = if first_chunk_of_gob {
                H261RtpHeader::gob_aligned(0, intra_only, motion_vectors)
            } else {
                // Continuation chunk — set GOBN=15 (reserved sentinel
                // per §4.1 valid GOBN range) only if the caller wants
                // to detect it. We instead leave GOBN=0 and rely on the
                // marker bit + RTP sequence number for reassembly. A
                // real MB-fragmenting implementation would parse MBA
                // state and fill these fields properly.
                H261RtpHeader::gob_aligned(0, intra_only, motion_vectors)
            };

            // The last chunk of any GOB may have trailing bits beyond
            // the end-of-payload byte boundary. Our slicer only cuts at
            // byte boundaries, so EBIT remains 0 — except for the very
            // last chunk of the entire stream, which inherits whatever
            // trailing padding bits the encoder appended.
            if chunk_end == total {
                hdr.ebit = 0;
            }

            // Marker bit: set on the last chunk of the frame, i.e. when
            // chunk_end falls strictly before the next PSC (or hits
            // the stream end with no further PSC).
            let next_psc_after = psc_byte_positions.iter().copied().find(|&pos| pos > p);
            let marker = match next_psc_after {
                Some(np) => chunk_end == np || (idx + 1 == starts.len() && chunk_end == total),
                None => chunk_end == total,
            };

            let mut bytes = Vec::with_capacity(HEADER_LEN + chunk_len);
            bytes.extend_from_slice(&pack_header(&hdr).expect("hdr packs"));
            bytes.extend_from_slice(&data[p..chunk_end]);
            payloads.push(H261RtpPayload {
                header: hdr,
                bytes,
                marker,
            });

            p = chunk_end;
            first_chunk_of_gob = false;
        }
    }

    payloads
}

/// RFC 4587 §4.1 context at a mid-GOB packet split point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FragCtx {
    /// GN of the GOB being continued (1..=12).
    pub(crate) gobn: u8,
    /// MBA of the macroblock that just ended (1..=32). This is the §4.1
    /// "macroblock address predictor" — the last MBA encoded in the
    /// previous packet when a fragment starts here (the header carries
    /// it biased -1).
    pub(crate) last_mba: u8,
    /// Quantizer (GQUANT or MQUANT) in effect after this MB (1..=31).
    pub(crate) quant: u8,
    /// Reference MVD for the next MB's §4.2.3.4 prediction: the MV of
    /// the MB that just ended if it was MC-coded, `(0, 0)` otherwise
    /// (§4.1: HMVD/VMVD are zero "when the MTYPE of the last MB encoded
    /// in the previous packet was not motion compensation").
    pub(crate) mv: (i32, i32),
}

/// One legal packet split point inside an H.261 elementary stream, as
/// recorded by [`walk_mb_split_points`].
///
/// A split point is either the first bit of a start code (PSC or GBSC —
/// a packet starting there begins with a header, so its §4.1 context
/// fields are all zero per the RFC) or the first bit after a macroblock
/// (a mid-GOB continuation carrying `ctx`). The encoder bit-packs GOB
/// headers, so split points sit at arbitrary bit offsets — SBIT/EBIT
/// carry the sub-byte alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SplitPoint {
    /// Absolute bit offset from the start of the elementary stream.
    pub(crate) bit: u64,
    /// True when this split point is the first bit of a PSC. Packets
    /// from different video frames must not share an RTP packet (their
    /// timestamps differ per §4.1), so a PSC is a *mandatory* cut.
    pub(crate) is_psc: bool,
    /// Mid-GOB continuation context, or `None` when the split point is
    /// a start code (all §4.1 context fields zero).
    pub(crate) ctx: Option<FragCtx>,
}

fn walk_err(e: oxideav_core::Error) -> RtpError {
    RtpError::MalformedStream {
        detail: e.to_string(),
    }
}

/// Skip the TCOEFF run/level + EOB sequence of one 8x8 block without
/// dequantising or transforming anything — RFC 4587 §4.2: "only the
/// Huffman encoding must be parsed and ... it is not necessary to
/// decompress the stream fully". Mirrors the bit consumption of
/// `crate::block::decode_ac_coeffs` exactly (including the run-overflow
/// and forbidden-escape-level checks, so a stream the decoder would
/// reject also fails the walk instead of desyncing it).
fn skip_ac_coeffs(
    br: &mut BitReader<'_>,
    start: usize,
    is_first_inter: bool,
) -> oxideav_core::Result<()> {
    let mut idx = start;
    let mut first = is_first_inter;
    loop {
        let sym = decode_tcoeff(br, first)?;
        first = false;
        match sym {
            TcoeffSym::Eob => return Ok(()),
            TcoeffSym::RunLevel { run, .. } => {
                let _sign = br.read_u1()?;
                idx = idx.saturating_add(run as usize);
                if idx > 63 {
                    return Err(oxideav_core::Error::invalid(format!(
                        "h261 block: AC run overflow (idx={idx}, run={run})"
                    )));
                }
                idx += 1;
            }
            TcoeffSym::Escape => {
                let run = br.read_u32(6)? as u8;
                let raw = br.read_u32(8)?;
                if raw == 0 || raw == 0x80 {
                    return Err(oxideav_core::Error::invalid(
                        "h261 escape: forbidden level FLC",
                    ));
                }
                idx = idx.saturating_add(run as usize);
                if idx > 63 {
                    return Err(oxideav_core::Error::invalid(format!(
                        "h261 escape: run overflow (idx={idx}, run={run})"
                    )));
                }
                idx += 1;
            }
        }
    }
}

/// Parse one macroblock at the Huffman layer only (no pixel work),
/// tracking the same `quant` / MV-predictor state the real decoder
/// keeps (`crate::mb::decode_macroblock`). Bit consumption is
/// bit-identical to the decoder's by construction.
fn skip_macroblock(
    br: &mut BitReader<'_>,
    mba: u8,
    quant: &mut u32,
    ctx: &mut MbContext,
) -> oxideav_core::Result<()> {
    let mtype: MtypeInfo = decode_vlc(br, MTYPE_TABLE)?;

    if mtype.mquant {
        let q = br.read_u32(5)?;
        if q == 0 {
            return Err(oxideav_core::Error::invalid("h261 MB: MQUANT == 0"));
        }
        *quant = q;
    }

    let mut mv = (0i32, 0i32);
    if mtype.mvd {
        let pred = if ctx.prev_was_mc && ctx.prev_mba != 0 && mba == ctx.prev_mba + 1 {
            ctx.mv
        } else {
            (0, 0)
        };
        let sym_x: MvdSym = decode_vlc(br, MVD_TABLE)?;
        let sym_y: MvdSym = decode_vlc(br, MVD_TABLE)?;
        mv = (reconstruct_mv(pred.0, sym_x), reconstruct_mv(pred.1, sym_y));
    }

    let cbp: u8 = if mtype.cbp {
        decode_vlc(br, CBP_TABLE)?
    } else if mtype.prediction == Prediction::Intra {
        0b111111
    } else {
        0
    };

    if mtype.prediction == Prediction::Intra {
        for _ in 0..6 {
            // INTRA DC FLC (Table 6) — validated like the decoder so a
            // forbidden value fails the walk instead of desyncing it.
            let dc = br.read_u32(8)?;
            if dc == 0x00 || dc == 0x80 {
                return Err(oxideav_core::Error::invalid(
                    "h261 INTRA DC: forbidden bitstream value",
                ));
            }
            skip_ac_coeffs(br, 1, false)?;
        }
        ctx.mv = (0, 0);
        ctx.prev_was_mc = false;
        ctx.prev_mba = mba;
        return Ok(());
    }

    let block_coded = [
        (cbp >> 5) & 1 != 0,
        (cbp >> 4) & 1 != 0,
        (cbp >> 3) & 1 != 0,
        (cbp >> 2) & 1 != 0,
        (cbp >> 1) & 1 != 0,
        cbp & 1 != 0,
    ];
    for coded in block_coded {
        if coded && mtype.tcoeff {
            skip_ac_coeffs(br, 0, true)?;
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

/// Walk an entire H.261 elementary stream at the Huffman layer and
/// record every legal packet split point with its RFC 4587 §4.1
/// context: each start code (PSC / GBSC, zero context) and the first
/// bit after every macroblock whose MBA is 1..=32 (mid-GOB continuation
/// context; MBAP cannot encode a predictor of 33, and nothing but the
/// next start code follows MB 33 anyway).
///
/// The walk consumes exactly the bits the real decoder consumes (same
/// VLC tables, same validation), so each recorded `bit` is a legal
/// packet split point per §3.2 ("Packets must start and end on an MB
/// boundary"; "the bit stream cannot be fragmented between a GOB header
/// and MB 1" holds because no point is recorded between a GBSC and the
/// end of MB 1).
pub(crate) fn walk_mb_split_points(data: &[u8]) -> Result<Vec<SplitPoint>, RtpError> {
    let mut br = BitReader::new(data);
    let mut points = Vec::new();

    loop {
        // Seek to the next start code (the reader sits just past the
        // previous picture header / GOB; only stuffing or byte-align
        // padding separates it from the next start code).
        let pos = br.bit_position();
        let Some(sc) = find_next_start_code_bits(data, pos) else {
            break;
        };
        if sc.bit_pos > pos {
            br.skip((sc.bit_pos - pos) as u32).map_err(walk_err)?;
        }

        if sc.gn == GN_PICTURE {
            points.push(SplitPoint {
                bit: sc.bit_pos,
                is_psc: true,
                ctx: None,
            });
            parse_picture_header(&mut br).map_err(walk_err)?;
            continue;
        }

        points.push(SplitPoint {
            bit: sc.bit_pos,
            is_psc: false,
            ctx: None,
        });
        let gob_hdr = parse_gob_header(&mut br).map_err(walk_err)?;
        let mut quant = gob_hdr.gquant as u32;
        let mut ctx = MbContext::reset();
        let mut current_mba: i32 = 0;
        loop {
            let remaining = br.bits_remaining();
            if remaining == 0 {
                break;
            }
            // GOBs are bit-contiguous, so the next start code begins
            // exactly where the last MB (plus any stuffing consumed by
            // `decode_mba_diff`) ends; mirror the decoder's peek-16
            // guard.
            if remaining >= 16 && br.peek_u32(16).map_err(walk_err)? == 0x0001 {
                break;
            }
            let diff = match decode_mba_diff(&mut br).map_err(walk_err)? {
                Some(d) => d as i32,
                None => break,
            };
            let new_mba = current_mba + diff;
            if !(1..=33).contains(&new_mba) {
                return Err(RtpError::MalformedStream {
                    detail: format!(
                        "MBA out of range {new_mba} (GN={}, prev_mba={current_mba})",
                        gob_hdr.gn
                    ),
                });
            }
            current_mba = new_mba;
            skip_macroblock(&mut br, new_mba as u8, &mut quant, &mut ctx).map_err(walk_err)?;
            if new_mba <= 32 {
                points.push(SplitPoint {
                    bit: br.bit_position(),
                    is_psc: false,
                    ctx: Some(FragCtx {
                        gobn: gob_hdr.gn,
                        last_mba: new_mba as u8,
                        quant: quant as u8,
                        mv: if ctx.prev_was_mc { ctx.mv } else { (0, 0) },
                    }),
                });
            }
        }
    }
    Ok(points)
}

/// Packetize an H.261 elementary stream with MB-level fragmentation
/// (RFC 4587 §4.2 RECOMMENDED packetization).
///
/// The stream is parsed once at the Huffman layer (no pixel decode —
/// §4.2: "it is not necessary to decompress the stream fully") to
/// collect every legal split point: each start code and each macroblock
/// boundary. Packets are then filled greedily — as many GOBs / MBs as
/// fit per packet, per §3.2 ("Multiple MBs may be carried in a single
/// packet ... This practice is recommended to reduce the packet send
/// rate") — under three rules:
///
/// * every payload is at most `max_payload` bytes;
/// * every packet starts and ends on an MB or start-code boundary
///   (§3.2: "an MB cannot be split across multiple packets"; "the bit
///   stream cannot be fragmented between a GOB header and MB 1");
/// * no packet crosses a PSC (packets of different video frames must
///   carry different RTP timestamps per §4.1).
///
/// A packet that starts with a PSC or GBSC has all-zero context fields;
/// a mid-GOB continuation packet carries the full §4.1 context:
///
/// * `SBIT` / `EBIT` — the split bit's offset within its byte
///   (consecutive fragments share the split byte, since the encoder
///   bit-packs GOB headers and macroblocks);
/// * `GOBN` — the GN of the GOB being continued;
/// * `MBAP` — the last MBA encoded in the previous packet, biased -1;
/// * `QUANT` — the GQUANT/MQUANT in effect prior to the packet;
/// * `HMVD` / `VMVD` — the reference MVD of the last MB of the previous
///   packet, or zero when that MB was not MC-coded.
///
/// The RTP marker hint is set on the last payload of each video frame,
/// as in [`packetize_gob_aligned`]. [`depacketize`] reassembles the
/// output byte-exactly (the SBIT/EBIT bit-walk handles the shared split
/// bytes).
///
/// Returns an empty vector if `data` contains no start codes.
///
/// # Errors
///
/// * [`RtpError::MalformedStream`] — the Huffman-layer walk failed (the
///   stream would not decode either).
/// * [`RtpError::FragmentTooLarge`] — some unfragmentable unit (picture
///   header + GOB header + MB 1, or a single MB run) needs more than
///   `max_payload - 4` bytes. Callers wanting a never-fails path can
///   fall back to [`packetize_gob_aligned`], which splits at byte
///   boundaries without per-packet decodability.
///
/// # Panics
///
/// Panics if `max_payload < HEADER_LEN + 1`, like [`packetize_gob_aligned`].
pub fn packetize_mb_fragmented(
    data: &[u8],
    max_payload: usize,
    intra_only: bool,
    motion_vectors: bool,
) -> Result<Vec<H261RtpPayload>, RtpError> {
    assert!(
        max_payload > HEADER_LEN,
        "max_payload must accommodate the 4-byte H.261 RTP header + at least 1 data byte"
    );

    let mut payloads = Vec::new();
    let points = walk_mb_split_points(data)?;
    if points.is_empty() {
        return Ok(payloads);
    }

    let max_data = max_payload - HEADER_LEN;
    let total_bits = (data.len() as u64) * 8;

    // Current fragment start: bit position + §4.1 context (None = the
    // fragment begins with a start code, all context fields zero).
    let mut s = points[0].bit;
    let mut s_ctx: Option<FragCtx> = None;
    // Index of the first candidate point not yet behind `s`.
    let mut search_from = 0usize;

    loop {
        let start_byte = (s / 8) as usize;
        while search_from < points.len() && points[search_from].bit <= s {
            search_from += 1;
        }

        // The current frame ends at the next PSC, or at the stream end.
        let frame_end = points[search_from..]
            .iter()
            .find(|p| p.is_psc)
            .map(|p| p.bit)
            .unwrap_or(total_bits);

        // Furthest split point (bounded by the frame end) whose byte
        // span still fits the budget. Spans are monotone in `bit`, so
        // the scan can stop at the first overflow. When two points share
        // a bit (an MB boundary that coincides with the next GBSC), the
        // later one — the start code, with zero context — wins.
        let mut chosen: Option<(u64, Option<FragCtx>)> = None;
        let mut first_span: Option<usize> = None;
        for p in points[search_from..]
            .iter()
            .take_while(|p| p.bit <= frame_end)
        {
            let span = (p.bit.div_ceil(8) as usize) - start_byte;
            if first_span.is_none() {
                first_span = Some(span);
            }
            if span > max_data {
                break;
            }
            chosen = Some((p.bit, p.ctx));
        }
        if frame_end == total_bits {
            // The stream end is a candidate too (it is not in `points`).
            let span = data.len() - start_byte;
            if first_span.is_none() {
                first_span = Some(span);
            }
            if span <= max_data {
                chosen = Some((total_bits, None));
            }
        }

        let Some((frag_end_bit, next_ctx)) = chosen else {
            // Even the nearest split point overflows the budget: the
            // unit between `s` and it cannot be fragmented legally.
            return Err(RtpError::FragmentTooLarge {
                needed: first_span.unwrap_or(data.len() - start_byte),
                max: max_data,
            });
        };

        let hdr = H261RtpHeader {
            sbit: (s % 8) as u8,
            ebit: ((8 - (frag_end_bit % 8)) % 8) as u8,
            intra_only,
            motion_vectors,
            gobn: s_ctx.map_or(0, |c| c.gobn),
            mbap: s_ctx.map_or(0, |c| c.last_mba - 1),
            quant: s_ctx.map_or(0, |c| c.quant),
            hmvd: s_ctx.map_or(0, |c| c.mv.0 as i8),
            vmvd: s_ctx.map_or(0, |c| c.mv.1 as i8),
        };
        let end_byte = frag_end_bit.div_ceil(8) as usize;
        let mut bytes = Vec::with_capacity(HEADER_LEN + (end_byte - start_byte));
        bytes.extend_from_slice(&pack_header(&hdr).expect("hdr packs"));
        bytes.extend_from_slice(&data[start_byte..end_byte]);
        payloads.push(H261RtpPayload {
            header: hdr,
            bytes,
            // Last packet of its video frame: ends exactly at the next
            // PSC or at the stream end.
            marker: frag_end_bit == frame_end,
        });

        if frag_end_bit >= total_bits {
            break;
        }
        s = frag_end_bit;
        s_ctx = next_ctx;
    }

    Ok(payloads)
}

/// Reassemble an H.261 elementary stream from a sequence of
/// [`H261RtpPayload`]s.
///
/// The depacketizer concatenates the inner H.261 data bytes, honouring
/// each packet's SBIT/EBIT (in the GOB-aligned case both are 0, so the
/// concatenation is a simple `extend_from_slice`; the general case uses
/// a bitwise concatenation that costs O(N) per packet boundary).
///
/// Returns the reassembled elementary stream and a count of any packets
/// whose payload was zero-length (those are valid but conventionally
/// indicate stuffing; they don't contribute data).
pub fn depacketize(payloads: &[H261RtpPayload]) -> Result<Vec<u8>, RtpError> {
    let mut out_bits = BitBuf::new();
    for p in payloads {
        if p.bytes.len() < HEADER_LEN {
            return Err(RtpError::ShortHeader);
        }
        let (hdr, data) = unpack_header(&p.bytes)?;
        if data.is_empty() {
            continue;
        }
        let total_bits = data.len() * 8;
        if (hdr.sbit as usize) + (hdr.ebit as usize) >= total_bits {
            return Err(RtpError::EmptyPayload);
        }
        let usable_bits = total_bits - hdr.sbit as usize - hdr.ebit as usize;
        out_bits.append_msb_bits(data, hdr.sbit as usize, usable_bits);
    }

    let bytes = out_bits.finish();
    // Sanity: the reassembled stream MUST contain at least one start
    // code, otherwise the depacketizer's input was corrupt.
    if iter_start_codes(&bytes).next().is_none() {
        return Err(RtpError::NoStartCodes);
    }
    Ok(bytes)
}

/// Length of the RTP fixed header (RFC 3550 §5.1), in bytes, for the
/// no-CSRC, no-extension case that an H.261 source produces.
pub const RTP_FIXED_HEADER_LEN: usize = 12;

/// One fully-framed RFC 3550 RTP packet ready to put on the wire.
///
/// `bytes` is laid out as:
///
/// ```text
///   [RTP fixed header (12 B, RFC 3550 §5.1)]
///   [H.261 RTP payload header (4 B,  RFC 4587 §4.1)]
///   [H.261 elementary-stream slice (1..N B)]
/// ```
///
/// The packet is self-contained: feed `bytes` straight to UDP / DTLS /
/// SRTP. The `marker`, `sequence_number`, `timestamp`, and `ssrc`
/// fields are exposed for receivers and for unit-test inspection but
/// they are also already encoded in `bytes`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RtpPacket {
    /// Full wire bytes (fixed header + H.261 payload header + payload).
    pub bytes: Vec<u8>,
    /// Marker bit — `true` on the last packet of a video frame
    /// (RFC 4587 §4.1).
    pub marker: bool,
    /// 16-bit RTP sequence number assigned to this packet.
    pub sequence_number: u16,
    /// 32-bit RTP timestamp (90 kHz clock per RFC 4587 §4.1 — same on
    /// every packet of one video frame).
    pub timestamp: u32,
    /// 32-bit RTP synchronisation-source identifier.
    pub ssrc: u32,
}

impl RtpPacket {
    /// Length of the inner H.261 elementary-stream bytes after the
    /// 12-byte RTP header and the 4-byte H.261 payload header.
    pub fn data_len(&self) -> usize {
        self.bytes
            .len()
            .saturating_sub(RTP_FIXED_HEADER_LEN + HEADER_LEN)
    }
}

/// RFC 3550 §5.1 RTP fixed-header builder for the simplest case
/// (no padding, no extension, no CSRC) used by [`RtpPacketizer`].
///
/// Layout per RFC 3550 §5.1:
///
/// ```text
///   0                   1                   2                   3
///    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
///   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
///   |V=2|P|X|  CC   |M|     PT      |       sequence number         |
///   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
///   |                           timestamp                           |
///   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
///   |           synchronization source (SSRC) identifier            |
///   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
/// ```
///
/// `pt` is a 7-bit payload type (RFC 4587 §4.1 leaves this to the SDP /
/// profile layer); the high bit is masked away if the caller passes a
/// value out of range. `marker` is the M bit. `seq`, `ts`, `ssrc` are
/// the next sequence number, RTP timestamp, and SSRC respectively.
fn write_rtp_fixed_header(buf: &mut Vec<u8>, pt: u8, marker: bool, seq: u16, ts: u32, ssrc: u32) {
    // Byte 0: V=2 (bits 0..2 high), P=0, X=0, CC=0 → 0b10_000000 = 0x80.
    buf.push(0x80);
    // Byte 1: M bit (top) + PT (low 7).
    buf.push(((marker as u8) << 7) | (pt & 0x7F));
    buf.extend_from_slice(&seq.to_be_bytes());
    buf.extend_from_slice(&ts.to_be_bytes());
    buf.extend_from_slice(&ssrc.to_be_bytes());
}

/// Encoder-side RFC 4587 RTP packetiser.
///
/// `RtpPacketizer` is the high-level glue between [`crate::encoder`]'s
/// elementary-stream output and the RTP wire format. Construct one per
/// RTP session (one SSRC) with the payload type negotiated for H.261 in
/// your SDP — typically a value in the dynamic range (96..=127) per
/// RFC 4587 §4.1 — and call [`Self::pack_frame`] once per coded picture
/// with the picture's bytes and its 90-kHz RTP timestamp.
///
/// Internally each frame is GOB-aligned-split via
/// [`packetize_gob_aligned`] then wrapped in an RFC 3550 §5.1 fixed
/// header. The marker bit is set on the last packet of each frame per
/// RFC 4587 §4.1. The sequence number auto-advances mod 2^16 across
/// packets and across frames; the initial value is whatever the caller
/// passes to [`Self::new`] (RFC 3550 §5.1 recommends a random initial
/// seed for security against known-plaintext attacks).
///
/// The RTP timestamp is **not** auto-advanced — RTP timestamps are
/// supplied per frame by the caller because (a) RTP allows frame
/// dropping that breaks a simple `+= clock_ticks_per_frame` invariant,
/// and (b) the 90-kHz clock per RFC 4587 §4.1 is a multiple of 30 / 1.001
/// fps so a strict integer-rational walk has to live in the caller's
/// frame-rate state anyway.
#[derive(Debug, Clone)]
pub struct RtpPacketizer {
    pt: u8,
    ssrc: u32,
    next_seq: u16,
    max_payload: usize,
    intra_only: bool,
    motion_vectors: bool,
    /// Use the RFC 4587 §4.2 RECOMMENDED MB-level fragmentation for GOBs
    /// that exceed the payload budget (with a GOB-aligned byte-split
    /// fallback when the Huffman-layer walk fails). Off by default.
    mb_fragmentation: bool,
    /// Running count of RTP data packets emitted by `pack_frame`, for the
    /// SR "sender's packet count" field (RFC 3550 §6.4.1). Wraps mod 2^32.
    packet_count: u32,
    /// Running count of payload octets (everything after the 12-byte RTP
    /// fixed header) emitted by `pack_frame`, for the SR "sender's octet
    /// count" field (RFC 3550 §6.4.1). Wraps mod 2^32.
    octet_count: u32,
    /// RTP timestamp of the most recently packed frame, surfaced into the
    /// SR sender-info section. `None` until the first `pack_frame` call.
    last_rtp_timestamp: Option<u32>,
}

impl RtpPacketizer {
    /// Construct a new packetiser.
    ///
    /// * `payload_type` — 7-bit RTP PT for this stream. High bit is
    ///   silently masked.
    /// * `ssrc` — 32-bit synchronisation source.
    /// * `initial_sequence_number` — first sequence number emitted by
    ///   [`Self::pack_frame`]; subsequent packets increment mod 2^16.
    /// * `max_rtp_packet_size` — maximum size in bytes of each emitted
    ///   RTP packet, **including** the 12-byte RTP fixed header and the
    ///   4-byte H.261 payload header. Pick `MTU - lower-layer overhead`.
    ///
    /// # Panics
    /// Panics if `max_rtp_packet_size <= RTP_FIXED_HEADER_LEN + HEADER_LEN`
    /// (the packet must carry at least one byte of H.261 data after both
    /// headers).
    pub fn new(
        payload_type: u8,
        ssrc: u32,
        initial_sequence_number: u16,
        max_rtp_packet_size: usize,
    ) -> Self {
        assert!(
            max_rtp_packet_size > RTP_FIXED_HEADER_LEN + HEADER_LEN,
            "max_rtp_packet_size must accommodate the 12-byte RTP fixed header + 4-byte H.261 header + ≥ 1 data byte"
        );
        Self {
            pt: payload_type & 0x7F,
            ssrc,
            next_seq: initial_sequence_number,
            max_payload: max_rtp_packet_size,
            intra_only: false,
            motion_vectors: true,
            mb_fragmentation: false,
            packet_count: 0,
            octet_count: 0,
            last_rtp_timestamp: None,
        }
    }

    /// Set the I (INTRA-only) hint surfaced in the H.261 RTP payload
    /// header for **every** packet from this packetiser. Per RFC 4587
    /// §4.1, the meaning of this bit must not change across the RTP
    /// session, so wire it once at session setup.
    pub fn with_intra_only(mut self, intra_only: bool) -> Self {
        self.intra_only = intra_only;
        self
    }

    /// Set the V (motion-vectors-may-be-used) hint. Per RFC 4587 §4.1
    /// the meaning of this bit must not change across the session.
    pub fn with_motion_vectors(mut self, motion_vectors: bool) -> Self {
        self.motion_vectors = motion_vectors;
        self
    }

    /// Enable the RFC 4587 §4.2 RECOMMENDED MB-level fragmentation.
    ///
    /// When enabled, [`Self::pack_frame`] routes each frame through
    /// [`packetize_mb_fragmented`]: a GOB exceeding the per-packet
    /// budget is split on macroblock boundaries and every continuation
    /// packet carries the §4.1 GOBN / MBAP / QUANT / HMVD / VMVD
    /// context, so a receiver can resume decoding mid-GOB after packet
    /// loss. If the MB-level walk fails (malformed stream, or a single
    /// MB run larger than the budget), `pack_frame` falls back to the
    /// GOB-aligned byte-split path so it never drops a frame.
    pub fn with_mb_fragmentation(mut self, mb_fragmentation: bool) -> Self {
        self.mb_fragmentation = mb_fragmentation;
        self
    }

    /// Next sequence number that will be assigned to a packet.
    pub fn next_sequence_number(&self) -> u16 {
        self.next_seq
    }

    /// 32-bit SSRC identifier the packetiser stamps into every packet.
    pub fn ssrc(&self) -> u32 {
        self.ssrc
    }

    /// 7-bit RTP payload type the packetiser stamps into every packet.
    pub fn payload_type(&self) -> u8 {
        self.pt
    }

    /// Pack one coded H.261 picture into a sequence of RTP packets.
    ///
    /// `frame_bytes` is the H.261 elementary stream for exactly one
    /// picture — as returned by [`crate::encoder::H261Encoder::encode_frame`]
    /// or by [`crate::encoder::encode_intra_picture`] /
    /// `encode_inter_picture`. `rtp_timestamp_90khz` is the RTP
    /// timestamp (RFC 4587 §4.1: 90-kHz clock); the same value is
    /// written to every packet of this frame (RFC 3550 §5.1).
    ///
    /// The marker bit is set on the LAST emitted packet of the frame
    /// (RFC 4587 §4.1: "MUST be set to one in the last packet of a
    /// video frame; otherwise, it MUST be zero").
    ///
    /// Returns an empty vector if `frame_bytes` contains no start codes
    /// (e.g. it was empty, or it was raw padding only — neither is a
    /// well-formed H.261 picture).
    pub fn pack_frame(&mut self, frame_bytes: &[u8], rtp_timestamp_90khz: u32) -> Vec<RtpPacket> {
        let inner_budget = self.max_payload - RTP_FIXED_HEADER_LEN;
        let h261_payloads = if self.mb_fragmentation {
            // §4.2 RECOMMENDED path; fall back to the GOB-aligned
            // byte-split when no MB-boundary split exists so a frame is
            // never dropped on the floor.
            packetize_mb_fragmented(
                frame_bytes,
                inner_budget,
                self.intra_only,
                self.motion_vectors,
            )
            .unwrap_or_else(|_| {
                packetize_gob_aligned(
                    frame_bytes,
                    inner_budget,
                    self.intra_only,
                    self.motion_vectors,
                )
            })
        } else {
            packetize_gob_aligned(
                frame_bytes,
                inner_budget,
                self.intra_only,
                self.motion_vectors,
            )
        };

        if h261_payloads.is_empty() {
            return Vec::new();
        }

        // RFC 4587 §4.1: "The marker bit of the RTP header MUST be set
        // to one in the last packet of a video frame; otherwise, it
        // MUST be zero." Override whatever per-payload hints
        // packetize_gob_aligned might have flagged: the only marker
        // that matters at the RTP layer is "is this the last packet of
        // the frame I'm packing right now".
        let last_idx = h261_payloads.len() - 1;
        let mut out = Vec::with_capacity(h261_payloads.len());
        for (i, p) in h261_payloads.into_iter().enumerate() {
            let marker = i == last_idx;
            let seq = self.next_seq;
            self.next_seq = self.next_seq.wrapping_add(1);

            let mut bytes = Vec::with_capacity(RTP_FIXED_HEADER_LEN + p.bytes.len());
            write_rtp_fixed_header(
                &mut bytes,
                self.pt,
                marker,
                seq,
                rtp_timestamp_90khz,
                self.ssrc,
            );
            bytes.extend_from_slice(&p.bytes);

            // RFC 3550 §6.4.1: "sender's octet count" is the total number
            // of *payload* octets — everything after the RTP fixed header,
            // excluding header and padding. For H.261 that's the 4-byte
            // payload header plus the bitstream slice (== p.bytes.len()).
            self.packet_count = self.packet_count.wrapping_add(1);
            self.octet_count = self.octet_count.wrapping_add(p.bytes.len() as u32);

            out.push(RtpPacket {
                bytes,
                marker,
                sequence_number: seq,
                timestamp: rtp_timestamp_90khz,
                ssrc: self.ssrc,
            });
        }
        if !out.is_empty() {
            self.last_rtp_timestamp = Some(rtp_timestamp_90khz);
        }
        out
    }

    /// Total RTP data packets this packetiser has emitted so far — the
    /// "sender's packet count" field of an RTCP SR (RFC 3550 §6.4.1).
    pub fn packet_count(&self) -> u32 {
        self.packet_count
    }

    /// Total payload octets this packetiser has emitted so far (everything
    /// after each RTP fixed header) — the "sender's octet count" field of
    /// an RTCP SR (RFC 3550 §6.4.1).
    pub fn octet_count(&self) -> u32 {
        self.octet_count
    }

    /// Build the [`crate::rtcp::SenderInfo`] section for an RTCP Sender
    /// Report from this packetiser's running counters (RFC 3550 §6.4.1).
    ///
    /// `ntp_timestamp` is the 64-bit NTP wallclock at which the SR is being
    /// sent (the caller's clock; pass `0` if it has no notion of wallclock,
    /// per §6.4.1). The RTP timestamp written into the sender info is the
    /// timestamp of the most recently packed frame; if no frame has been
    /// packed yet it is `0`.
    ///
    /// The returned `SenderInfo` is fed to
    /// [`crate::rtcp::build_sender_report`] together with this packetiser's
    /// [`Self::ssrc`] and any reception report blocks the endpoint has.
    pub fn sender_info(&self, ntp_timestamp: u64) -> crate::rtcp::SenderInfo {
        crate::rtcp::SenderInfo {
            ntp_timestamp,
            rtp_timestamp: self.last_rtp_timestamp.unwrap_or(0),
            packet_count: self.packet_count,
            octet_count: self.octet_count,
        }
    }

    /// Build a complete RTCP Sender Report (PT = 200) for this session from
    /// the packetiser's running counters plus caller-supplied reception
    /// report blocks (RFC 3550 §6.4.1).
    ///
    /// `ntp_timestamp` is the wallclock NTP value for the report instant;
    /// `blocks` are the reception report blocks for sources this endpoint
    /// has heard (0..=31). Returns the wire bytes, or
    /// [`crate::rtcp::RtcpError::TooManyReportBlocks`] if more than 31
    /// blocks are supplied.
    pub fn sender_report(
        &self,
        ntp_timestamp: u64,
        blocks: &[crate::rtcp::ReceptionReportBlock],
    ) -> Result<Vec<u8>, crate::rtcp::RtcpError> {
        crate::rtcp::build_sender_report(self.ssrc, &self.sender_info(ntp_timestamp), blocks)
    }
}

/// Parsed RTP fixed header (RFC 3550 §5.1) — the fields the H.261
/// receive path actually inspects. Padding / extension / CSRC handling
/// is the caller's job; this struct only models the V/P/X/CC/M/PT/seq/
/// timestamp/SSRC fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RtpFixedHeader {
    /// RTP version (must equal 2 for this codec; the parser rejects
    /// other values).
    pub version: u8,
    /// Padding (P) bit.
    pub padding: bool,
    /// Extension (X) bit.
    pub extension: bool,
    /// CSRC count (CC field).
    pub csrc_count: u8,
    /// Marker (M) bit.
    pub marker: bool,
    /// 7-bit payload type.
    pub payload_type: u8,
    /// 16-bit RTP sequence number.
    pub sequence_number: u16,
    /// 32-bit RTP timestamp.
    pub timestamp: u32,
    /// 32-bit synchronisation-source identifier.
    pub ssrc: u32,
}

/// Parse one RFC 3550 §5.1 RTP fixed header from `buf`. Returns the
/// parsed header and a slice pointing past the fixed header **and**
/// past any CSRC identifiers (so the next byte the caller looks at is
/// either a header extension or the H.261 RTP payload header).
///
/// Returns:
/// * [`RtpError::ShortHeader`] if `buf` is shorter than the fixed
///   12 bytes or shorter than 12 + 4 * CC.
/// * [`RtpError::FieldOverflow`] if the version field is not 2.
pub fn parse_rtp_fixed_header(buf: &[u8]) -> Result<(RtpFixedHeader, &[u8]), RtpError> {
    if buf.len() < RTP_FIXED_HEADER_LEN {
        return Err(RtpError::ShortHeader);
    }
    let b0 = buf[0];
    let b1 = buf[1];
    let version = (b0 >> 6) & 0x3;
    if version != 2 {
        return Err(RtpError::FieldOverflow {
            field: "RTP-V",
            value: version as u32,
        });
    }
    let padding = (b0 & 0x20) != 0;
    let extension = (b0 & 0x10) != 0;
    let csrc_count = b0 & 0x0F;
    let marker = (b1 & 0x80) != 0;
    let payload_type = b1 & 0x7F;
    let sequence_number = u16::from_be_bytes([buf[2], buf[3]]);
    let timestamp = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let ssrc = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]);

    let csrc_bytes = (csrc_count as usize) * 4;
    let need = RTP_FIXED_HEADER_LEN + csrc_bytes;
    if buf.len() < need {
        return Err(RtpError::ShortHeader);
    }

    Ok((
        RtpFixedHeader {
            version,
            padding,
            extension,
            csrc_count,
            marker,
            payload_type,
            sequence_number,
            timestamp,
            ssrc,
        },
        &buf[need..],
    ))
}

/// Minimal MSB-first bit-append buffer used by [`depacketize`] to
/// handle non-zero SBIT/EBIT in the general case. The fast path
/// (`sbit == 0 && ebit == 0`) reduces to byte-aligned appends.
struct BitBuf {
    bytes: Vec<u8>,
    /// Number of valid bits already written into `bytes` (0..=8 * bytes.len()).
    bit_len: usize,
}

impl BitBuf {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_len: 0,
        }
    }

    fn append_msb_bits(&mut self, src: &[u8], src_skip_bits: usize, count: usize) {
        // Fast path: byte-aligned source, byte-aligned dest.
        if src_skip_bits == 0 && self.bit_len % 8 == 0 && count == src.len() * 8 {
            self.bytes.extend_from_slice(src);
            self.bit_len += count;
            return;
        }
        // Slow path: one bit at a time.
        for i in 0..count {
            let bit_index = src_skip_bits + i;
            let byte = src[bit_index / 8];
            let shift = 7 - (bit_index % 8);
            let bit = (byte >> shift) & 1;
            self.push_bit(bit);
        }
    }

    fn push_bit(&mut self, bit: u8) {
        let byte_index = self.bit_len / 8;
        let bit_index_in_byte = 7 - (self.bit_len % 8);
        if byte_index == self.bytes.len() {
            self.bytes.push(0);
        }
        self.bytes[byte_index] |= (bit & 1) << bit_index_in_byte;
        self.bit_len += 1;
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_then_unpack_round_trip_zero_header() {
        let h = H261RtpHeader::gob_aligned(0, false, false);
        let bytes = pack_header(&h).unwrap();
        // GOB-aligned header with no flags packs to four zero bytes.
        assert_eq!(bytes, [0, 0, 0, 0]);
        let (h2, rest) = unpack_header(&bytes).unwrap();
        assert_eq!(h, h2);
        assert!(rest.is_empty());
    }

    #[test]
    fn pack_then_unpack_round_trip_typical_values() {
        let h = H261RtpHeader {
            sbit: 3,
            ebit: 5,
            intra_only: true,
            motion_vectors: true,
            gobn: 7,
            mbap: 12,
            quant: 17,
            hmvd: -7,
            vmvd: 11,
        };
        let bytes = pack_header(&h).unwrap();
        // Sanity-check the SBIT field landed in the top 3 bits.
        assert_eq!((bytes[0] >> 5) & 0x7, 3);
        let mut buf = vec![0u8; 4];
        buf.copy_from_slice(&bytes);
        buf.push(0xAB); // tail data
        let (h2, rest) = unpack_header(&buf).unwrap();
        assert_eq!(h, h2);
        assert_eq!(rest, &[0xAB]);
    }

    #[test]
    fn pack_then_unpack_round_trip_negative_mvds() {
        // Sweep every legal 5-bit MVD (-15..=15) to confirm the
        // sign-extension is symmetric.
        for hmvd in -15i8..=15 {
            for vmvd in -15i8..=15 {
                let h = H261RtpHeader {
                    sbit: 0,
                    ebit: 0,
                    intra_only: false,
                    motion_vectors: true,
                    gobn: 0,
                    mbap: 0,
                    quant: 0,
                    hmvd,
                    vmvd,
                };
                let bytes = pack_header(&h).unwrap();
                let (h2, _) = unpack_header(&bytes).unwrap();
                assert_eq!(h, h2, "round-trip failed for ({hmvd},{vmvd})");
            }
        }
    }

    #[test]
    fn pack_rejects_forbidden_mvd_value() {
        let mut h = H261RtpHeader::gob_aligned(0, false, true);
        h.hmvd = -16;
        assert_eq!(
            pack_header(&h),
            Err(RtpError::ForbiddenMvd { field: "HMVD" })
        );
        let mut h2 = H261RtpHeader::gob_aligned(0, false, true);
        h2.vmvd = -16;
        assert_eq!(
            pack_header(&h2),
            Err(RtpError::ForbiddenMvd { field: "VMVD" })
        );
    }

    #[test]
    fn pack_rejects_oversized_sbit_ebit() {
        let mut h = H261RtpHeader::gob_aligned(0, false, false);
        h.sbit = 8;
        assert_eq!(
            pack_header(&h),
            Err(RtpError::BadBitOffset {
                field: "SBIT",
                value: 8
            })
        );
        let mut h2 = H261RtpHeader::gob_aligned(0, false, false);
        h2.ebit = 9;
        assert_eq!(
            pack_header(&h2),
            Err(RtpError::BadBitOffset {
                field: "EBIT",
                value: 9
            })
        );
    }

    #[test]
    fn pack_rejects_oversized_quant() {
        let mut h = H261RtpHeader::gob_aligned(0, false, false);
        h.quant = 32;
        assert_eq!(
            pack_header(&h),
            Err(RtpError::FieldOverflow {
                field: "QUANT",
                value: 32
            })
        );
    }

    #[test]
    fn unpack_rejects_short_header() {
        for n in 0..HEADER_LEN {
            let buf = vec![0u8; n];
            assert_eq!(unpack_header(&buf), Err(RtpError::ShortHeader));
        }
    }

    #[test]
    fn unpack_field_widths_match_rfc4587_layout() {
        // Hand-craft a known bit pattern and confirm the field decomposition
        // matches RFC 4587 §4.1's layout exactly. Word laid out MSB-first:
        //   SBIT  = 111  (7)
        //   EBIT  = 110  (6)
        //   I     = 1
        //   V     = 0
        //   GOBN  = 1010 (10)
        //   MBAP  = 11000 (24)
        //   QUANT = 00011 (3)
        //   HMVD  = 01111 (15)
        //   VMVD  = 10001 (-15 via 2's complement)
        // = (7<<29)|(6<<26)|(1<<25)|(0<<24)|(10<<20)|(24<<15)|(3<<10)|(15<<5)|17
        // V bit is bit 24; intentionally cleared in this fixture.
        let word: u32 = (7u32 << 29)
            | (6 << 26)
            | (1 << 25)
            | (10 << 20)
            | (24 << 15)
            | (3 << 10)
            | (15 << 5)
            | 17;
        let bytes = word.to_be_bytes();
        let (h, rest) = unpack_header(&bytes).unwrap();
        assert_eq!(h.sbit, 7);
        assert_eq!(h.ebit, 6);
        assert!(h.intra_only);
        assert!(!h.motion_vectors);
        assert_eq!(h.gobn, 10);
        assert_eq!(h.mbap, 24);
        assert_eq!(h.quant, 3);
        assert_eq!(h.hmvd, 15);
        assert_eq!(h.vmvd, -15);
        assert!(rest.is_empty());
    }

    #[test]
    fn packetize_returns_empty_for_input_with_no_start_codes() {
        let data = vec![0xFFu8; 100];
        let pkts = packetize_gob_aligned(&data, 256, true, false);
        assert!(pkts.is_empty());
    }

    #[test]
    fn packetize_splits_at_gob_boundaries_for_small_stream() {
        // Two byte-aligned start codes back-to-back; each fits well
        // inside the cap. We expect exactly two payloads.
        // Byte 0..=2: PSC (20 bits) — `00 00 00 01 0...` (20 zeros + 1)
        //   actually PSC = 0x00010 (20 bits including the trailing 0).
        //   We emit 0x00 0x01 0x00 then arbitrary tail = byte 3.
        // Byte 4..=6: GBSC GN=1 → 0x00 0x01 0x1F (the `1F` byte starts
        //   with `0001 1111` ⇒ the 4 high bits encode `0001` after the
        //   16-bit prefix's `1` sync, i.e. GN=1).
        // We pad with a couple of bytes to give each payload some data.
        let mut data = vec![0x00, 0x01, 0x00, 0xAA, 0xBB];
        // Start of second start code at byte 5.
        data.extend_from_slice(&[0x00, 0x01, 0x1F, 0xCC, 0xDD]);
        let pkts = packetize_gob_aligned(&data, 1024, false, true);
        assert_eq!(pkts.len(), 2, "expected one payload per start code");
        // The first payload covers bytes 0..5 (5 bytes inner data + 4-byte header).
        assert_eq!(pkts[0].data_len(), 5);
        assert_eq!(pkts[1].data_len(), 5);
        // Marker on second packet only (last of frame; there's no
        // subsequent PSC, so it's the tail of the stream).
        assert!(pkts[1].marker);
    }

    #[test]
    fn packetize_fragments_large_gob_at_byte_boundaries() {
        // Construct a single GBSC followed by 500 bytes of data. With
        // max_payload = 100 (4 header + 96 data), we expect ceil(500/96)
        // = 6 payloads.
        let mut data = vec![0x00, 0x01, 0x1F];
        data.extend(std::iter::repeat(0xAA).take(500 - 3));
        let pkts = packetize_gob_aligned(&data, 100, false, true);
        assert_eq!(pkts.len(), 500_usize.div_ceil(96));
        // All but the last carry max-size payloads.
        for p in &pkts[..pkts.len() - 1] {
            assert_eq!(p.data_len(), 96);
        }
        // Marker on the last payload of the frame.
        assert!(pkts.last().unwrap().marker);
    }

    #[test]
    fn packetize_then_depacketize_round_trip() {
        // Build a synthetic stream: PSC + 10 bytes + GBSC + 10 bytes.
        let mut data = vec![0x00, 0x01, 0x00];
        data.extend(std::iter::repeat(0xA5).take(10));
        data.extend_from_slice(&[0x00, 0x01, 0x1F]);
        data.extend(std::iter::repeat(0x5A).take(10));
        let pkts = packetize_gob_aligned(&data, 256, false, false);
        let recovered = depacketize(&pkts).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn packetize_then_depacketize_round_trip_with_fragmentation() {
        // Force fragmentation by picking a tiny max_payload.
        let mut data = vec![0x00, 0x01, 0x00];
        data.extend(std::iter::repeat(0xA5).take(50));
        data.extend_from_slice(&[0x00, 0x01, 0x1F]);
        data.extend(std::iter::repeat(0x5A).take(50));
        let pkts = packetize_gob_aligned(&data, 20, false, false);
        assert!(pkts.len() > 4, "expected fragmentation");
        let recovered = depacketize(&pkts).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn depacketize_rejects_short_header_payload() {
        let bad = H261RtpPayload {
            header: H261RtpHeader::gob_aligned(0, false, false),
            bytes: vec![0u8; 2],
            marker: true,
        };
        assert_eq!(depacketize(&[bad]), Err(RtpError::ShortHeader));
    }

    #[test]
    fn depacketize_rejects_payload_with_no_start_codes() {
        // 4 header bytes + 4 data bytes of all-ones — no GBSC pattern
        // present.
        let hdr = pack_header(&H261RtpHeader::gob_aligned(0, false, false)).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&hdr);
        bytes.extend_from_slice(&[0xFFu8; 4]);
        let p = H261RtpPayload {
            header: H261RtpHeader::gob_aligned(0, false, false),
            bytes,
            marker: true,
        };
        assert_eq!(depacketize(&[p]), Err(RtpError::NoStartCodes));
    }

    #[test]
    fn packetize_marker_bit_set_only_on_last_packet_of_frame() {
        // Two pictures, one GOB each. We expect the marker bit on the
        // payload that holds the tail of each picture.
        // Picture 1: PSC + 5 bytes.
        // Picture 2: PSC + 5 bytes.
        let mut data = vec![0x00, 0x01, 0x00];
        data.extend(std::iter::repeat(0x11).take(5));
        data.extend_from_slice(&[0x00, 0x01, 0x00]);
        data.extend(std::iter::repeat(0x22).take(5));
        let pkts = packetize_gob_aligned(&data, 256, false, false);
        // With max_payload large, each picture is one payload ⇒ two
        // total payloads, both holding a "tail of frame" position.
        assert_eq!(pkts.len(), 2);
        // First packet's marker should be true (it's the last of
        // picture 1, since picture 2's PSC comes immediately after).
        assert!(pkts[0].marker, "first packet should be tail of picture 1");
        // Second packet's marker should also be true (tail of stream).
        assert!(pkts[1].marker, "second packet should be tail of picture 2");
    }

    #[test]
    fn empty_input_packetizes_to_empty_output() {
        let pkts = packetize_gob_aligned(&[], 256, false, false);
        assert!(pkts.is_empty());
    }

    #[test]
    #[should_panic(expected = "max_payload must accommodate")]
    fn packetize_panics_on_tiny_max_payload() {
        let _ = packetize_gob_aligned(&[0x00, 0x01, 0x00], HEADER_LEN, false, false);
    }

    // ------------------------------------------------------------------
    // RtpPacketizer / parse_rtp_fixed_header tests
    // ------------------------------------------------------------------

    fn synthetic_two_picture_stream() -> Vec<u8> {
        // Two pictures back-to-back: PSC + 5 bytes, PSC + 5 bytes.
        let mut data = vec![0x00, 0x01, 0x00];
        data.extend(std::iter::repeat(0x11).take(5));
        data.extend_from_slice(&[0x00, 0x01, 0x00]);
        data.extend(std::iter::repeat(0x22).take(5));
        data
    }

    fn synthetic_one_picture_stream() -> Vec<u8> {
        let mut data = vec![0x00, 0x01, 0x00];
        data.extend(std::iter::repeat(0xA5).take(20));
        data
    }

    #[test]
    fn rtp_packet_stamps_fixed_header_fields_and_marker_on_last() {
        let mut pk = RtpPacketizer::new(96, 0xDEAD_BEEF, 1000, 1500);
        let frame = synthetic_one_picture_stream();
        let packets = pk.pack_frame(&frame, 90_000);
        assert_eq!(packets.len(), 1);
        let p = &packets[0];
        assert_eq!(
            p.bytes.len(),
            RTP_FIXED_HEADER_LEN + HEADER_LEN + frame.len()
        );
        assert_eq!(p.sequence_number, 1000);
        assert_eq!(p.timestamp, 90_000);
        assert_eq!(p.ssrc, 0xDEAD_BEEF);
        assert!(p.marker, "single-packet frame must carry M=1");
        // The wire bytes' marker bit lives in byte 1 bit 7.
        assert_eq!(p.bytes[1] & 0x80, 0x80);
        assert_eq!(p.bytes[1] & 0x7F, 96);
        // V=2, P=0, X=0, CC=0 ⇒ byte 0 == 0x80.
        assert_eq!(p.bytes[0], 0x80);
        // Sequence number advanced.
        assert_eq!(pk.next_sequence_number(), 1001);
    }

    #[test]
    fn rtp_packet_marker_set_only_on_last_of_frame_when_fragmented() {
        // Tiny MTU forces fragmentation: 20-byte payload window.
        let mut pk = RtpPacketizer::new(96, 1, 0, 20)
            .with_intra_only(true)
            .with_motion_vectors(false);
        let frame = synthetic_one_picture_stream();
        let packets = pk.pack_frame(&frame, 12345);
        assert!(packets.len() >= 2, "expected fragmentation");
        for (i, p) in packets.iter().enumerate() {
            if i + 1 == packets.len() {
                assert!(p.marker, "last packet must have M=1");
                assert_eq!(p.bytes[1] & 0x80, 0x80);
            } else {
                assert!(!p.marker, "non-last packet must have M=0");
                assert_eq!(p.bytes[1] & 0x80, 0);
            }
            assert_eq!(p.timestamp, 12345);
            assert_eq!(p.ssrc, 1);
        }
        // Sequence numbers are dense.
        for w in packets.windows(2) {
            assert_eq!(w[1].sequence_number, w[0].sequence_number.wrapping_add(1));
        }
    }

    #[test]
    fn rtp_packet_two_frames_share_timestamp_per_frame_only() {
        let mut pk = RtpPacketizer::new(96, 7, 0, 1500);
        let one = synthetic_one_picture_stream();
        let pkts1 = pk.pack_frame(&one, 100);
        let pkts2 = pk.pack_frame(&one, 200);
        assert_eq!(pkts1.len(), 1);
        assert_eq!(pkts2.len(), 1);
        assert_eq!(pkts1[0].timestamp, 100);
        assert_eq!(pkts2[0].timestamp, 200);
        // Sequence number kept advancing across frame boundary.
        assert_eq!(pkts2[0].sequence_number, pkts1[0].sequence_number + 1);
        // Both frames are tail-of-frame ⇒ both marker = 1.
        assert!(pkts1[0].marker);
        assert!(pkts2[0].marker);
    }

    #[test]
    fn rtp_packet_two_gobs_in_one_frame_only_last_has_marker() {
        // Two GOBs in one frame: PSC + GOB1 + GOB2. Only the GOB2
        // packet should carry M=1 (it's the tail of the frame).
        let mut pk = RtpPacketizer::new(96, 1, 0, 1500);
        let frame = synthetic_two_picture_stream();
        // Caveat: synthetic_two_picture_stream actually has two PSCs.
        // Packetizer treats each PSC as starting a new frame, but the
        // RtpPacketizer-level marker is "last packet of the call to
        // pack_frame", so both should have marker=true ONLY on the
        // last packet of the pack_frame call. The packet stream order
        // is still "GOB1 then GOB2", so only GOB2's packet has M=1.
        let packets = pk.pack_frame(&frame, 90_000);
        assert!(packets.len() >= 2, "expected ≥ 2 packets");
        assert!(!packets[0].marker, "first packet must not be M=1");
        assert!(packets.last().unwrap().marker, "last packet must be M=1");
    }

    #[test]
    fn rtp_packet_payload_type_is_masked_to_7_bits() {
        // Pass PT=0xFF, expect the wire byte to carry only the low 7
        // bits (0x7F), so byte 1 with M=0 is 0x7F.
        let mut pk = RtpPacketizer::new(0xFF, 0, 0, 1500);
        let pkts = pk.pack_frame(&synthetic_one_picture_stream(), 1);
        assert_eq!(pkts.len(), 1);
        assert_eq!(pk.payload_type(), 0x7F);
        // Marker bit IS set on the only packet (last of frame), so the
        // wire byte is 0xFF; mask it back out to verify PT.
        let pt = pkts[0].bytes[1] & 0x7F;
        assert_eq!(pt, 0x7F);
    }

    #[test]
    fn rtp_packet_sequence_number_wraps() {
        let mut pk = RtpPacketizer::new(96, 0, u16::MAX, 1500);
        let p1 = pk.pack_frame(&synthetic_one_picture_stream(), 0);
        let p2 = pk.pack_frame(&synthetic_one_picture_stream(), 1);
        assert_eq!(p1[0].sequence_number, u16::MAX);
        assert_eq!(p2[0].sequence_number, 0);
    }

    #[test]
    fn rtp_packet_empty_input_emits_no_packets() {
        let mut pk = RtpPacketizer::new(96, 0, 0, 1500);
        let pkts = pk.pack_frame(&[], 0);
        assert!(pkts.is_empty());
        // No-start-code input also emits nothing.
        let pkts = pk.pack_frame(&[0xFFu8; 8], 0);
        assert!(pkts.is_empty());
    }

    #[test]
    #[should_panic(expected = "max_rtp_packet_size must accommodate")]
    fn rtp_packetizer_panics_on_tiny_mtu() {
        let _ = RtpPacketizer::new(96, 0, 0, RTP_FIXED_HEADER_LEN + HEADER_LEN);
    }

    #[test]
    fn parse_rtp_fixed_header_round_trips_against_packetizer_output() {
        let mut pk = RtpPacketizer::new(101, 0x1234_5678, 42, 1500);
        let pkts = pk.pack_frame(&synthetic_one_picture_stream(), 0xCAFE_BABE);
        let pkt = &pkts[0];
        let (hdr, rest) = parse_rtp_fixed_header(&pkt.bytes).unwrap();
        assert_eq!(hdr.version, 2);
        assert!(!hdr.padding);
        assert!(!hdr.extension);
        assert_eq!(hdr.csrc_count, 0);
        assert!(hdr.marker);
        assert_eq!(hdr.payload_type, 101);
        assert_eq!(hdr.sequence_number, 42);
        assert_eq!(hdr.timestamp, 0xCAFE_BABE);
        assert_eq!(hdr.ssrc, 0x1234_5678);
        // The first 4 bytes of `rest` are the H.261 RTP payload header.
        assert!(rest.len() >= HEADER_LEN);
        let (h261_hdr, _payload) = unpack_header(rest).unwrap();
        assert_eq!(h261_hdr.sbit, 0);
        assert_eq!(h261_hdr.ebit, 0);
    }

    #[test]
    fn parse_rtp_fixed_header_rejects_short_buffer() {
        for n in 0..RTP_FIXED_HEADER_LEN {
            let buf = vec![0x80u8; n]; // V=2 in the first byte but truncated
            assert_eq!(parse_rtp_fixed_header(&buf), Err(RtpError::ShortHeader));
        }
    }

    #[test]
    fn parse_rtp_fixed_header_rejects_wrong_version() {
        // V=1, P=0, X=0, CC=0 ⇒ 0x40.
        let mut buf = vec![0u8; RTP_FIXED_HEADER_LEN];
        buf[0] = 0x40;
        assert_eq!(
            parse_rtp_fixed_header(&buf),
            Err(RtpError::FieldOverflow {
                field: "RTP-V",
                value: 1,
            })
        );
    }

    #[test]
    fn parse_rtp_fixed_header_consumes_csrc_block() {
        // V=2, P=0, X=0, CC=2 ⇒ byte 0 = 0x82. We feed 12 + 2*4 = 20 B.
        let mut buf = vec![0u8; RTP_FIXED_HEADER_LEN + 8 + 3];
        buf[0] = 0x82;
        buf[1] = 0x00; // M=0, PT=0
                       // Fill seq/ts/ssrc with sentinel zeros, then two CSRC ids.
        buf[12..16].copy_from_slice(&0xAABB_CCDDu32.to_be_bytes());
        buf[16..20].copy_from_slice(&0x1122_3344u32.to_be_bytes());
        buf[20] = 0xFE; // sentinel tail byte
        buf[21] = 0xED;
        buf[22] = 0xBE;
        let (hdr, rest) = parse_rtp_fixed_header(&buf).unwrap();
        assert_eq!(hdr.csrc_count, 2);
        assert_eq!(rest, &[0xFE, 0xED, 0xBE]);
    }

    #[test]
    fn parse_rtp_fixed_header_short_when_csrc_count_overruns() {
        // Claim CC=15 but only carry 12 bytes ⇒ ShortHeader.
        let mut buf = vec![0u8; RTP_FIXED_HEADER_LEN];
        buf[0] = 0x8F; // V=2, CC=15
        assert_eq!(parse_rtp_fixed_header(&buf), Err(RtpError::ShortHeader));
    }

    #[test]
    fn packetizer_tracks_packet_and_octet_counts() {
        use crate::rtcp::{parse_report, PT_SR};
        let mut pk = RtpPacketizer::new(96, 0xABCD, 0, 1500);
        assert_eq!(pk.packet_count(), 0);
        assert_eq!(pk.octet_count(), 0);

        let frame = synthetic_one_picture_stream();
        let pkts = pk.pack_frame(&frame, 9000);
        assert_eq!(pkts.len(), 1);
        // One packet emitted; octet count is everything after the 12-byte
        // fixed header = HEADER_LEN + inner data.
        assert_eq!(pk.packet_count(), 1);
        let expected_octets = (pkts[0].bytes.len() - RTP_FIXED_HEADER_LEN) as u32;
        assert_eq!(pk.octet_count(), expected_octets);

        // A second frame accumulates.
        let pkts2 = pk.pack_frame(&frame, 12000);
        assert_eq!(pk.packet_count(), 2);
        let expected2 = expected_octets + (pkts2[0].bytes.len() - RTP_FIXED_HEADER_LEN) as u32;
        assert_eq!(pk.octet_count(), expected2);

        // Build an SR from the packetiser state and verify the counters
        // round-trip and the RTP timestamp is the last frame's.
        let sr = pk
            .sender_report(SenderInfoTestClock::NTP, &[])
            .expect("sr builds");
        let parsed = parse_report(&sr).unwrap();
        assert_eq!(parsed.packet_type, PT_SR);
        assert_eq!(parsed.ssrc, 0xABCD);
        let info = parsed.sender_info.expect("SR carries sender info");
        assert_eq!(info.packet_count, 2);
        assert_eq!(info.octet_count, expected2);
        assert_eq!(info.rtp_timestamp, 12000);
        assert_eq!(info.ntp_timestamp, SenderInfoTestClock::NTP);
    }

    struct SenderInfoTestClock;
    impl SenderInfoTestClock {
        const NTP: u64 = 0xB44D_B705_2000_0000;
    }

    #[test]
    fn sender_info_rtp_timestamp_zero_before_any_frame() {
        let pk = RtpPacketizer::new(96, 1, 0, 1500);
        let info = pk.sender_info(0);
        assert_eq!(info.rtp_timestamp, 0);
        assert_eq!(info.packet_count, 0);
        assert_eq!(info.octet_count, 0);
    }

    #[test]
    fn empty_pack_frame_does_not_advance_counters() {
        let mut pk = RtpPacketizer::new(96, 1, 0, 1500);
        let _ = pk.pack_frame(&[], 5);
        let _ = pk.pack_frame(&[0xFFu8; 8], 5); // no start codes
        assert_eq!(pk.packet_count(), 0);
        assert_eq!(pk.octet_count(), 0);
        // last_rtp_timestamp stays unset.
        assert_eq!(pk.sender_info(0).rtp_timestamp, 0);
    }

    #[test]
    fn rtp_packet_round_trip_recovers_elementary_stream() {
        // Pack a frame, then strip the RTP fixed header + reuse
        // depacketize on the inner H.261 payloads to recover bytes.
        // Use a deliberately tiny MTU so the synthetic stream is
        // fragmented across packets.
        let mut pk = RtpPacketizer::new(96, 0, 0, RTP_FIXED_HEADER_LEN + HEADER_LEN + 4);
        // Stream: PSC + 30 bytes ⇒ inner budget is 4 ⇒ ≥ 8 chunks.
        let mut frame = vec![0x00, 0x01, 0x00];
        frame.extend(std::iter::repeat(0xA5).take(30));
        let pkts = pk.pack_frame(&frame, 0);
        assert!(
            pkts.len() >= 2,
            "expected fragmentation, got {}",
            pkts.len()
        );
        let mut inner = Vec::new();
        for p in &pkts {
            let (_h, rest) = parse_rtp_fixed_header(&p.bytes).unwrap();
            // Re-wrap as H261RtpPayload so depacketize can consume it.
            inner.push(H261RtpPayload {
                header: unpack_header(rest).unwrap().0,
                bytes: rest.to_vec(),
                marker: p.marker,
            });
        }
        let recovered = depacketize(&inner).unwrap();
        assert_eq!(recovered, frame);
    }

    // ------------------------------------------------------------------
    // MB-level fragmentation (RFC 4587 §4.2 RECOMMENDED packetization)
    // ------------------------------------------------------------------

    /// Deterministic textured QCIF planes — enough AC energy that the
    /// coded picture is several KiB at quant 8 (forces MB-level splits).
    fn textured_qcif() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (w, h) = (176usize, 144usize);
        let mut y = vec![0u8; w * h];
        let mut state = 0x2545_F491u32;
        for j in 0..h {
            for i in 0..w {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let base = 40 + ((i * 160) / w + (j * 40) / h) as u32;
                y[j * w + i] = (base + (state & 0x3F)).min(255) as u8;
            }
        }
        let mut cb = vec![128u8; (w / 2) * (h / 2)];
        let mut cr = vec![128u8; (w / 2) * (h / 2)];
        for (idx, v) in cb.iter_mut().enumerate() {
            *v = 100 + ((idx * 7) % 56) as u8;
        }
        for (idx, v) in cr.iter_mut().enumerate() {
            *v = 110 + ((idx * 11) % 36) as u8;
        }
        (y, cb, cr)
    }

    fn encode_textured_qcif_intra(quant: u32) -> Vec<u8> {
        let (y, cb, cr) = textured_qcif();
        crate::encoder::encode_intra_picture(
            crate::picture::SourceFormat::Qcif,
            &y,
            176,
            &cb,
            88,
            &cr,
            88,
            quant,
            0,
        )
        .expect("encode_intra_picture")
    }

    /// Re-walk an elementary stream with the REAL decoder
    /// (`decode_macroblock`, full pixel reconstruction) recording the
    /// same split-point list `walk_mb_split_points` produces. This is
    /// the independent oracle the Huffman-layer skip walk is checked
    /// against: if the skip path consumed even one bit more or less
    /// than the decode path, the positions (and thus every §4.1 context
    /// field) would diverge.
    fn decoder_walk_oracle(data: &[u8]) -> Vec<SplitPoint> {
        use crate::gob::{cif_gob_origin_luma, qcif_gob_origin_luma};
        use crate::mb::{decode_macroblock, Picture};
        use crate::picture::SourceFormat;

        let mut br = BitReader::new(data);
        let mut points = Vec::new();
        let mut fmt = SourceFormat::Qcif;
        let mut pic = Picture::new(176, 144);
        loop {
            let pos = br.bit_position();
            let Some(sc) = find_next_start_code_bits(data, pos) else {
                break;
            };
            if sc.bit_pos > pos {
                br.skip((sc.bit_pos - pos) as u32).unwrap();
            }
            if sc.gn == GN_PICTURE {
                points.push(SplitPoint {
                    bit: sc.bit_pos,
                    is_psc: true,
                    ctx: None,
                });
                let hdr = parse_picture_header(&mut br).unwrap();
                fmt = hdr.source_format;
                pic = Picture::new(hdr.width as usize, hdr.height as usize);
                continue;
            }
            points.push(SplitPoint {
                bit: sc.bit_pos,
                is_psc: false,
                ctx: None,
            });
            let gob_hdr = parse_gob_header(&mut br).unwrap();
            let (gob_x, gob_y) = match fmt {
                SourceFormat::Cif => cif_gob_origin_luma(gob_hdr.gn),
                SourceFormat::Qcif => qcif_gob_origin_luma(gob_hdr.gn),
            };
            let mut quant = gob_hdr.gquant as u32;
            let mut ctx = MbContext::reset();
            let mut current_mba: i32 = 0;
            loop {
                let remaining = br.bits_remaining();
                if remaining == 0 {
                    break;
                }
                if remaining >= 16 && br.peek_u32(16).unwrap() == 0x0001 {
                    break;
                }
                let Some(diff) = decode_mba_diff(&mut br).unwrap() else {
                    break;
                };
                let new_mba = current_mba + diff as i32;
                current_mba = new_mba;
                decode_macroblock(
                    &mut br,
                    new_mba as u8,
                    gob_x,
                    gob_y,
                    &mut quant,
                    &mut ctx,
                    &mut pic,
                    None,
                )
                .unwrap();
                if new_mba <= 32 {
                    points.push(SplitPoint {
                        bit: br.bit_position(),
                        is_psc: false,
                        ctx: Some(FragCtx {
                            gobn: gob_hdr.gn,
                            last_mba: new_mba as u8,
                            quant: quant as u8,
                            mv: if ctx.prev_was_mc { ctx.mv } else { (0, 0) },
                        }),
                    });
                }
            }
        }
        points
    }

    #[test]
    fn mb_walker_matches_real_decoder_bit_for_bit() {
        // The fragmenter's Huffman-layer walk must consume EXACTLY the
        // bits the real decoder consumes, or every recorded split point
        // (and thus every §4.1 context field) would be garbage. Check a
        // real encoded picture against the decode-path oracle across a
        // quantiser sweep (different quants exercise different TCOEFF
        // code lengths and escape usage).
        for quant in [2u32, 8, 31] {
            let bits = encode_textured_qcif_intra(quant);
            let walked = walk_mb_split_points(&bits).expect("walk");
            let oracle = decoder_walk_oracle(&bits);
            assert_eq!(
                walked, oracle,
                "walker desynced from decoder at quant {quant}"
            );
            // Structure: 1 PSC + 3 GBSCs (QCIF GOBs 1, 3, 5), and an
            // INTRA picture codes all 33 MBs of every GOB — 32 mid-GOB
            // split points each (MB 33's end is the next start code).
            assert_eq!(walked.iter().filter(|p| p.is_psc).count(), 1);
            assert_eq!(
                walked
                    .iter()
                    .filter(|p| !p.is_psc && p.ctx.is_none())
                    .count(),
                3,
                "QCIF transmits GOBs 1, 3, 5"
            );
            assert_eq!(walked.iter().filter(|p| p.ctx.is_some()).count(), 3 * 32);
        }
    }

    #[test]
    fn mb_walker_matches_real_decoder_on_p_picture_with_motion() {
        // P-picture coverage: shifted+textured content makes the encoder
        // emit INTER+MC (and FIL) macroblocks with non-zero MVDs, so the
        // walker's §4.2.3.4 MV-predictor tracking is exercised against
        // the decode-path oracle too.
        let (y, cb, cr) = textured_qcif();
        let mut enc = crate::encoder::H261Encoder::new(crate::picture::SourceFormat::Qcif, 6);
        let _i = enc.encode_frame(&y, 176, &cb, 88, &cr, 88).expect("I");
        // Shift the luma 4 right / 2 down for clean integer-pel motion.
        let mut y2 = vec![0u8; 176 * 144];
        for j in 0..144usize {
            for i in 0..176usize {
                let sj = j.saturating_sub(2);
                let si = i.saturating_sub(4);
                y2[j * 176 + i] = y[sj * 176 + si];
            }
        }
        let p_bits = enc.encode_frame(&y2, 176, &cb, 88, &cr, 88).expect("P");

        let walked = walk_mb_split_points(&p_bits).expect("walk");
        let oracle = decoder_walk_oracle(&p_bits);
        assert_eq!(walked, oracle, "walker desynced from decoder on P-picture");
        assert!(
            walked.iter().any(|p| p.ctx.is_some_and(|c| c.mv != (0, 0))),
            "shifted content should produce at least one MC-coded MB boundary"
        );
    }

    #[test]
    fn mb_fragment_round_trip_is_byte_exact_and_respects_budget() {
        let bits = encode_textured_qcif_intra(8);
        for &budget in &[128usize, 256, 512] {
            let pkts = packetize_mb_fragmented(&bits, budget, true, false).expect("fragment");
            assert!(
                pkts.len() > 3,
                "expected several packets at budget {budget}"
            );
            let mut saw_continuation = false;
            for p in &pkts {
                assert!(
                    p.bytes.len() <= budget,
                    "payload {} exceeds budget {budget}",
                    p.bytes.len()
                );
                assert!(p.bytes.len() > HEADER_LEN);
                if p.header.gobn != 0 {
                    saw_continuation = true;
                    // Continuation context invariants (RFC 4587 §4.1).
                    assert!((1..=12).contains(&p.header.gobn));
                    assert!((1..=31).contains(&p.header.quant));
                    assert!(p.header.mbap <= 31);
                    assert!((-15..=15).contains(&p.header.hmvd));
                    assert!((-15..=15).contains(&p.header.vmvd));
                }
            }
            assert!(
                saw_continuation,
                "budget {budget} should force at least one mid-GOB packet"
            );
            // Exactly one marker (single picture), on the last packet.
            assert_eq!(pkts.iter().filter(|p| p.marker).count(), 1);
            assert!(pkts.last().unwrap().marker);
            let recovered = depacketize(&pkts).expect("depacketize");
            assert_eq!(recovered, bits, "round trip at budget {budget}");
        }
    }

    #[test]
    fn mb_fragment_boundaries_are_bit_contiguous() {
        // Fragments tile the stream with no gaps: every consecutive pair
        // must agree on the split bit (next.sbit == (8 - prev.ebit) % 8)
        // and share the split byte when it lands mid-byte.
        let bits = encode_textured_qcif_intra(8);
        let pkts = packetize_mb_fragmented(&bits, 128, true, false).expect("fragment");
        let mut misaligned_splits = 0;
        for pair in pkts.windows(2) {
            let (prev, next) = (&pair[0], &pair[1]);
            assert_eq!(
                next.header.sbit,
                (8 - prev.header.ebit) % 8,
                "SBIT/EBIT must describe the same split bit"
            );
            if next.header.sbit != 0 {
                misaligned_splits += 1;
                assert_eq!(
                    prev.bytes.last().unwrap(),
                    &next.bytes[HEADER_LEN],
                    "fragments must share the split byte"
                );
            }
        }
        assert!(
            misaligned_splits > 0,
            "expected at least one non-byte-aligned MB split"
        );
    }

    #[test]
    fn mb_fragment_continuation_context_matches_walker() {
        // Every continuation packet's §4.1 context must equal a walker
        // split point (which the oracle test above ties bit-for-bit to
        // the real decoder).
        let bits = encode_textured_qcif_intra(8);
        let pkts = packetize_mb_fragmented(&bits, 128, true, false).expect("fragment");
        let points = walk_mb_split_points(&bits).expect("walk");

        let mut checked = 0;
        for p in pkts.iter().filter(|p| p.header.gobn != 0) {
            let matched = points.iter().any(|sp| {
                sp.ctx.is_some_and(|c| {
                    c.gobn == p.header.gobn
                        && c.last_mba == p.header.mbap + 1
                        && c.quant == p.header.quant
                        && c.mv == (p.header.hmvd as i32, p.header.vmvd as i32)
                        && (sp.bit % 8) as u8 == p.header.sbit
                })
            });
            assert!(
                matched,
                "continuation header {:?} matches no walker split point",
                p.header
            );
            checked += 1;
        }
        assert!(checked > 0, "expected continuation packets to verify");
    }

    #[test]
    fn mb_fragment_errors_when_no_split_fits() {
        // With max_data = 1 even the picture-header-to-GBSC unit (4
        // bytes: PSC + TR + PTYPE + PEI = 32 bits) cannot be emitted;
        // the fragmenter must surface FragmentTooLarge instead of
        // emitting an undecodable packet.
        let bits = encode_textured_qcif_intra(8);
        match packetize_mb_fragmented(&bits, HEADER_LEN + 1, true, false) {
            Err(RtpError::FragmentTooLarge { needed, max }) => {
                assert_eq!(max, 1);
                assert!(needed > 1);
            }
            other => panic!("expected FragmentTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn mb_fragment_emits_single_packet_when_frame_fits() {
        // §3.2 efficiency rule: multiple MBs (here, the whole picture —
        // picture header plus all three GOBs) ride one packet when they
        // fit, unlike the one-payload-per-start-code cheap packetizer.
        let bits = encode_textured_qcif_intra(16);
        let pkts =
            packetize_mb_fragmented(&bits, bits.len() + HEADER_LEN, true, false).expect("fragment");
        assert_eq!(pkts.len(), 1, "whole frame fits one packet");
        let p = &pkts[0];
        assert!(p.marker);
        assert_eq!(p.header.sbit, 0);
        assert_eq!(p.header.ebit, 0);
        assert_eq!(p.header.gobn, 0);
        assert_eq!(p.header.quant, 0);
        assert_eq!(depacketize(&pkts).unwrap(), bits);
    }

    #[test]
    fn mb_fragment_never_spans_a_psc() {
        // Two coded pictures in one buffer: every frame's tail packet
        // carries the marker, and the next packet starts at the PSC
        // (zero context, byte-aligned SBIT=0) — packets of different
        // video frames must not share an RTP packet (§4.1 timestamps).
        let (y, cb, cr) = textured_qcif();
        let mut enc = crate::encoder::H261Encoder::new(crate::picture::SourceFormat::Qcif, 8);
        let f0 = enc.encode_frame(&y, 176, &cb, 88, &cr, 88).expect("I");
        let f1 = enc.encode_frame(&y, 176, &cb, 88, &cr, 88).expect("P");
        let mut stream = f0.clone();
        stream.extend_from_slice(&f1);

        let pkts = packetize_mb_fragmented(&stream, 128, true, false).expect("fragment");
        assert_eq!(
            pkts.iter().filter(|p| p.marker).count(),
            2,
            "one marker per picture"
        );
        // The packet after each marker starts the next frame at its PSC.
        for pair in pkts.windows(2) {
            if pair[0].marker {
                assert_eq!(pair[1].header.gobn, 0);
                assert_eq!(pair[1].header.sbit, 0);
            }
        }
        assert_eq!(depacketize(&pkts).unwrap(), stream);
    }

    #[test]
    fn rtp_packetizer_mb_fragmentation_emits_context_and_round_trips() {
        let bits = encode_textured_qcif_intra(8);
        let mut pk = RtpPacketizer::new(96, 0xABCD_EF01, 0, 200)
            .with_intra_only(true)
            .with_mb_fragmentation(true);
        let packets = pk.pack_frame(&bits, 0);
        assert!(!packets.is_empty());
        let mut inner = Vec::new();
        let mut saw_continuation = false;
        for p in &packets {
            assert!(p.bytes.len() <= 200);
            let (_h, rest) = parse_rtp_fixed_header(&p.bytes).unwrap();
            let (h261, _) = unpack_header(rest).unwrap();
            if h261.gobn != 0 {
                saw_continuation = true;
            }
            inner.push(H261RtpPayload {
                header: h261,
                bytes: rest.to_vec(),
                marker: p.marker,
            });
        }
        assert!(saw_continuation, "MTU 200 should force mid-GOB packets");
        assert!(packets.last().unwrap().marker);
        assert_eq!(packets.iter().filter(|p| p.marker).count(), 1);
        assert_eq!(depacketize(&inner).unwrap(), bits);
    }

    #[test]
    fn rtp_packetizer_mb_fragmentation_falls_back_when_unsplittable() {
        // Inner budget of 5 bytes (max_data = 1) cannot hold the picture
        // header; pack_frame must fall back to the byte-split path
        // instead of dropping the frame.
        let bits = encode_textured_qcif_intra(16);
        let mtu = RTP_FIXED_HEADER_LEN + HEADER_LEN + 1;
        let mut pk = RtpPacketizer::new(96, 1, 0, mtu).with_mb_fragmentation(true);
        let packets = pk.pack_frame(&bits, 0);
        assert!(!packets.is_empty(), "fallback must still emit packets");
        let mut inner = Vec::new();
        for p in &packets {
            assert!(p.bytes.len() <= mtu);
            let (_h, rest) = parse_rtp_fixed_header(&p.bytes).unwrap();
            inner.push(H261RtpPayload {
                header: unpack_header(rest).unwrap().0,
                bytes: rest.to_vec(),
                marker: p.marker,
            });
        }
        assert_eq!(depacketize(&inner).unwrap(), bits);
    }
}
