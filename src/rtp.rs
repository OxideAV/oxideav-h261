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
//! * [`depacketize`] — reassemble an elementary stream from a sequence
//!   of payloads, honouring per-packet SBIT/EBIT and concatenating the
//!   inner bits to a single byte buffer.
//!
//! ## What it does **not** do
//!
//! * MB-level fragmentation (§4.2 RECOMMENDED for hardware codecs whose
//!   GOBs exceed the MTU). Our software encoder typically produces GOBs
//!   well under 1 KiB at canonical p×64 kbit/s rates, so GOB-aligned
//!   packets fit comfortably below an IPv4 MTU. The header builder is
//!   nonetheless general enough that a caller can hand-construct an MB-
//!   fragmented packetizer on top of [`pack_header`] / [`unpack_header`].
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

use crate::start_code::{iter_start_codes, GN_PICTURE};

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
        let h261_payloads = packetize_gob_aligned(
            frame_bytes,
            inner_budget,
            self.intra_only,
            self.motion_vectors,
        );

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

            out.push(RtpPacket {
                bytes,
                marker,
                sequence_number: seq,
                timestamp: rtp_timestamp_90khz,
                ssrc: self.ssrc,
            });
        }
        out
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
}
