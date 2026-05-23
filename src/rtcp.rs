//! RTCP control-channel packets — RFC 3550 §6.
//!
//! The [`crate::rtp::RtpPacketizer`] puts H.261 media on the wire as RFC 3550
//! §5.1 RTP data packets. Every RTP session also runs an RTCP control channel
//! (RFC 3550 §6) on which each participant periodically reports transmission
//! and reception statistics and identifies itself. This module builds and
//! parses the RTCP packet types an H.261 endpoint emits:
//!
//! * **SR** (Sender Report, PT = 200, §6.4.1) — sent by a participant that
//!   has transmitted data during the interval. It carries a 20-byte *sender
//!   info* section (NTP + RTP timestamps, sender's packet & octet counts)
//!   plus zero or more reception report blocks.
//! * **RR** (Receiver Report, PT = 201, §6.4.2) — identical to an SR minus
//!   the sender-info section. An empty RR (RC = 0) is the canonical "I have
//!   nothing to report" packet a pure receiver puts at the head of its
//!   compound RTCP packet.
//! * **SDES** (Source Description, PT = 202, §6.5) — one or more *chunks*,
//!   each binding an SSRC/CSRC to a list of textual items. The **CNAME**
//!   item (§6.5.1) is mandatory: §6.1 requires every compound RTCP packet to
//!   carry an SDES CNAME so a new receiver can bind the (randomly chosen,
//!   collision-mutable) SSRC to a stable source identifier. NAME / EMAIL /
//!   PHONE / LOC / TOOL / NOTE / PRIV (§6.5.2–§6.5.8) are also supported.
//! * **BYE** (Goodbye, PT = 203, §6.6) — announces that one or more SSRC/CSRC
//!   sources have left the session, with an optional free-text reason.
//! * **APP** (Application-Defined, PT = 204, §6.7) — a four-octet ASCII name
//!   plus an arbitrary application-dependent payload (a multiple of 32 bits
//!   long). Intended for experimental extensions and per-application control
//!   that does not warrant its own IANA-registered packet type.
//!
//! ## Compound packets (§6.1)
//!
//! RFC 3550 §6.1 requires every transmitted RTCP packet to be a *compound*
//! packet of at least two stacked sub-packets: a report (SR or RR — an empty
//! RR if nothing was sent or received) **first**, followed by an SDES packet
//! carrying at least the CNAME item, then optionally BYE / APP. The
//! sub-packets are concatenated with no separators; each is self-delimiting
//! via its 16-bit `length` field (32-bit words minus one). [`compound`]
//! concatenates pre-built sub-packets into one datagram body; [`parse_compound`]
//! walks a received datagram back into the individual packets.
//!
//! Both reports share the same 8-byte RTCP header and the same 24-byte
//! [`ReceptionReportBlock`] format. The only structural difference is the
//! PT code and the presence of the sender-info section, so the two builders
//! delegate to a common core.
//!
//! ## Wire layouts (RFC 3550 §6.4.1 / §6.4.2)
//!
//! SR:
//!
//! ```text
//!  0                   1                   2                   3
//!   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//!  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//!  |V=2|P|    RC   |   PT=SR=200   |             length            | header
//!  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//!  |                         SSRC of sender                        |
//!  +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
//!  |              NTP timestamp, most significant word             | sender
//!  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ info
//!  |             NTP timestamp, least significant word             |
//!  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//!  |                         RTP timestamp                         |
//!  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//!  |                     sender's packet count                     |
//!  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//!  |                      sender's octet count                     |
//!  +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
//!  |                  reception report block(s) …                  |
//!  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! ```
//!
//! Each reception report block is 24 bytes:
//!
//! ```text
//!  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//!  |                    SSRC_n (source SSRC)                       |
//!  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//!  | fraction lost |       cumulative number of packets lost       |
//!  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//!  |           extended highest sequence number received           |
//!  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//!  |                       interarrival jitter                     |
//!  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//!  |                          last SR (LSR)                        |
//!  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//!  |                    delay since last SR (DLSR)                 |
//!  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! ```
//!
//! RR is byte-for-byte the same except PT = 201 and the 20-byte sender-info
//! section is omitted.
//!
//! ## What this module is for
//!
//! H.261 over RTP (RFC 4587) is a thin payload format on top of RFC 3550;
//! it does not redefine RTCP. These builders therefore live here so an
//! H.261 sender driven by [`crate::rtp::RtpPacketizer`] can emit a
//! conformant compound packet (SR/RR + SDES CNAME, optionally followed by
//! BYE) using the packet/octet counters the packetiser already tracks, and a
//! receiver can parse the same. Scheduling (the §6.2 transmission-interval
//! algorithm) and the reception-statistics estimators (§A.3 loss fraction,
//! §A.8 jitter) are deliberately out of scope — those are session-management
//! concerns above the codec, and RFC 3550 specifies them independently of
//! the wire formats implemented here. The builders accept pre-computed
//! statistics; the parsers surface them back.

/// PT code for an RTCP Sender Report (RFC 3550 §6.4.1).
pub const PT_SR: u8 = 200;
/// PT code for an RTCP Receiver Report (RFC 3550 §6.4.2).
pub const PT_RR: u8 = 201;
/// PT code for an RTCP Source Description packet (RFC 3550 §6.5).
pub const PT_SDES: u8 = 202;
/// PT code for an RTCP Goodbye packet (RFC 3550 §6.6).
pub const PT_BYE: u8 = 203;
/// PT code for an RTCP Application-defined packet (RFC 3550 §6.7).
pub const PT_APP: u8 = 204;

/// Length in octets of the APP `name` field (RFC 3550 §6.7:
/// "name: 4 octets … a sequence of four ASCII characters").
pub const APP_NAME_LEN: usize = 4;

/// Maximum value the 5-bit APP `subtype` field can hold (RFC 3550 §6.7).
pub const APP_SUBTYPE_MAX: u8 = 31;

/// Maximum number of chunks (SDES) or SSRC/CSRC identifiers (BYE) that fit in
/// one packet — the SC field is 5 bits (RFC 3550 §6.5 / §6.6).
pub const MAX_SOURCE_COUNT: usize = 31;

/// Maximum length in octets of one SDES item's text / BYE reason string — the
/// length byte is 8 bits (RFC 3550 §6.5).
pub const MAX_TEXT_LEN: usize = 255;

/// Length in bytes of the common RTCP report header (V/P/RC, PT, length,
/// SSRC of sender). RFC 3550 §6.4.1: "the header, is 8 octets long".
pub const RTCP_HEADER_LEN: usize = 8;

/// Length in bytes of the fixed RTCP header common to *all* packet types
/// (the V/P/count byte, PT, and 16-bit length) — RFC 3550 §6.4.1. SR/RR
/// follow it with the sender's SSRC (making [`RTCP_HEADER_LEN`] = 8); SDES /
/// BYE do not, so their bodies begin right after these 4 bytes.
pub const RTCP_HEADER_FIXED_LEN: usize = 4;

/// Length in bytes of the SR sender-info section (RFC 3550 §6.4.1:
/// "the sender information, is 20 octets long").
pub const SENDER_INFO_LEN: usize = 20;

/// Length in bytes of one reception report block (RFC 3550 §6.4.1).
pub const REPORT_BLOCK_LEN: usize = 24;

/// Maximum number of reception report blocks that fit in a single SR/RR
/// packet — the RC field is 5 bits (RFC 3550 §6.4.1).
pub const MAX_REPORT_BLOCKS: usize = 31;

/// RTCP build / parse error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RtcpError {
    /// More than 31 reception report blocks were supplied — they cannot be
    /// encoded into the 5-bit RC field of a single packet (RFC 3550 §6.4.1
    /// says additional packets should be stacked instead).
    TooManyReportBlocks { count: usize },
    /// Buffer is shorter than a complete RTCP report header.
    ShortHeader,
    /// The version field in the RTCP header is not 2.
    BadVersion { value: u8 },
    /// The PT field is neither 200 (SR) nor 201 (RR).
    UnexpectedPacketType { value: u8 },
    /// The buffer is shorter than the `length`-field-implied packet size,
    /// or shorter than `RC` report blocks demand.
    Truncated,
    /// The `length` field (32-bit words minus one) does not match the
    /// number of bytes the RC count + report type imply.
    LengthMismatch {
        stated_words: u16,
        actual_words: u16,
    },
    /// More than 31 SDES chunks / BYE sources were supplied — they cannot be
    /// encoded into the 5-bit SC field (RFC 3550 §6.5 / §6.6).
    TooManySources { count: usize },
    /// An SDES item text or BYE reason exceeds the 255-octet limit imposed by
    /// the 8-bit length byte (RFC 3550 §6.5).
    TextTooLong { len: usize },
    /// A `PRIV` SDES item's prefix + value would exceed the 255-octet item
    /// length budget once the 1-byte prefix-length is included (RFC 3550 §6.5.8).
    PrivTooLong { len: usize },
    /// An APP `name` field is not exactly 4 octets (RFC 3550 §6.7).
    AppNameWrongLength { len: usize },
    /// An APP `application-dependent data` blob is not a multiple of 32 bits
    /// (RFC 3550 §6.7: "It must be a multiple of 32 bits long.").
    AppDataNotAligned { len: usize },
    /// An APP `subtype` exceeds the 5-bit field maximum of 31 (RFC 3550 §6.7).
    AppSubtypeOutOfRange { value: u8 },
}

impl core::fmt::Display for RtcpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RtcpError::TooManyReportBlocks { count } => {
                write!(f, "rtcp: {count} report blocks exceeds the 31-block limit")
            }
            RtcpError::ShortHeader => write!(f, "rtcp: buffer < 8-byte report header"),
            RtcpError::BadVersion { value } => write!(f, "rtcp: version {value} != 2"),
            RtcpError::UnexpectedPacketType { value } => {
                write!(f, "rtcp: PT {value} is neither SR (200) nor RR (201)")
            }
            RtcpError::Truncated => write!(f, "rtcp: buffer shorter than length/RC implies"),
            RtcpError::LengthMismatch {
                stated_words,
                actual_words,
            } => write!(
                f,
                "rtcp: length field says {stated_words} words, body is {actual_words}"
            ),
            RtcpError::TooManySources { count } => {
                write!(f, "rtcp: {count} sources exceeds the 31-source SC limit")
            }
            RtcpError::TextTooLong { len } => {
                write!(f, "rtcp: text/reason {len} octets exceeds the 255 limit")
            }
            RtcpError::PrivTooLong { len } => {
                write!(f, "rtcp: PRIV item {len} octets exceeds the 255 limit")
            }
            RtcpError::AppNameWrongLength { len } => {
                write!(f, "rtcp: APP name is {len} octets, must be exactly 4")
            }
            RtcpError::AppDataNotAligned { len } => {
                write!(f, "rtcp: APP data is {len} octets, must be a multiple of 4")
            }
            RtcpError::AppSubtypeOutOfRange { value } => {
                write!(f, "rtcp: APP subtype {value} exceeds the 5-bit max of 31")
            }
        }
    }
}

impl std::error::Error for RtcpError {}

/// The 20-byte SR sender-info section (RFC 3550 §6.4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct SenderInfo {
    /// Wallclock time when the report was sent, as a 64-bit NTP timestamp
    /// (seconds since 1900-01-01 in the high 32 bits, fractional seconds in
    /// the low 32). A sender with no notion of wallclock may use zero.
    pub ntp_timestamp: u64,
    /// RTP timestamp corresponding to the same instant as `ntp_timestamp`,
    /// in the stream's media clock units (90 kHz for H.261 per RFC 4587).
    pub rtp_timestamp: u32,
    /// Total RTP data packets transmitted by the sender since transmission
    /// started, up to the moment this SR was generated.
    pub packet_count: u32,
    /// Total payload octets (excluding the RTP fixed header and padding)
    /// transmitted by the sender, up to this SR.
    pub octet_count: u32,
}

impl SenderInfo {
    /// Build an NTP 64-bit timestamp from whole seconds + a 32-bit fraction
    /// of a second (RFC 3550 §4 format). Convenience for callers that hold
    /// the two halves separately.
    pub fn ntp_from_parts(seconds: u32, fraction: u32) -> u64 {
        ((seconds as u64) << 32) | (fraction as u64)
    }
}

/// One reception report block (RFC 3550 §6.4.1), 24 bytes on the wire.
///
/// All statistics are supplied by the caller — this module does not run the
/// §A.1/§A.3/§A.8 estimators; it only (de)serialises the block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ReceptionReportBlock {
    /// SSRC of the source these statistics describe.
    pub ssrc: u32,
    /// Fraction of packets lost since the previous report, as an 8-bit
    /// fixed-point value with the binary point at the left edge (loss
    /// fraction × 256, integer part). RFC 3550 §6.4.1.
    pub fraction_lost: u8,
    /// Cumulative number of packets lost since reception began. 24-bit
    /// signed two's-complement (loss can be negative when duplicates
    /// arrive); the field is masked to 24 bits on encode.
    pub cumulative_lost: i32,
    /// Extended highest sequence number received: low 16 bits the highest
    /// sequence number seen, high 16 bits the count of sequence-number
    /// cycles.
    pub extended_highest_seq: u32,
    /// Interarrival jitter estimate, in media-clock units (RFC 3550 §6.4.1
    /// / §A.8). Caller supplies the value; the field is opaque here.
    pub jitter: u32,
    /// Middle 32 bits of the NTP timestamp from the last SR received from
    /// this source (LSR); zero if none received yet.
    pub last_sr: u32,
    /// Delay since the last SR from this source, in units of 1/65536 s
    /// (DLSR); zero if no SR received yet.
    pub delay_since_last_sr: u32,
}

impl ReceptionReportBlock {
    fn write(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.ssrc.to_be_bytes());
        // fraction lost (8) | cumulative lost (24, two's complement).
        let cum = (self.cumulative_lost as u32) & 0x00FF_FFFF;
        let word = ((self.fraction_lost as u32) << 24) | cum;
        buf.extend_from_slice(&word.to_be_bytes());
        buf.extend_from_slice(&self.extended_highest_seq.to_be_bytes());
        buf.extend_from_slice(&self.jitter.to_be_bytes());
        buf.extend_from_slice(&self.last_sr.to_be_bytes());
        buf.extend_from_slice(&self.delay_since_last_sr.to_be_bytes());
    }

    fn read(buf: &[u8]) -> Self {
        // Caller guarantees buf.len() >= REPORT_BLOCK_LEN.
        let ssrc = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let word = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let fraction_lost = (word >> 24) as u8;
        // Sign-extend the 24-bit cumulative-lost field to i32.
        let cum24 = word & 0x00FF_FFFF;
        let cumulative_lost = if cum24 & 0x0080_0000 != 0 {
            (cum24 | 0xFF00_0000) as i32
        } else {
            cum24 as i32
        };
        let extended_highest_seq = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let jitter = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]);
        let last_sr = u32::from_be_bytes([buf[16], buf[17], buf[18], buf[19]]);
        let delay_since_last_sr = u32::from_be_bytes([buf[20], buf[21], buf[22], buf[23]]);
        Self {
            ssrc,
            fraction_lost,
            cumulative_lost,
            extended_highest_seq,
            jitter,
            last_sr,
            delay_since_last_sr,
        }
    }
}

/// A parsed RTCP report (SR or RR), as produced by [`parse_report`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RtcpReport {
    /// PT code: [`PT_SR`] (200) or [`PT_RR`] (201).
    pub packet_type: u8,
    /// SSRC of the report's originator.
    pub ssrc: u32,
    /// Sender-info section — `Some` for SR, `None` for RR.
    pub sender_info: Option<SenderInfo>,
    /// Reception report blocks (0..=31).
    pub report_blocks: Vec<ReceptionReportBlock>,
}

/// Common header writer. `rc` is the report-block count (already validated
/// to be <= 31), `pt` the packet type, `total_len_bytes` the full packet
/// length in bytes (must be a multiple of 4).
fn write_header(buf: &mut Vec<u8>, rc: u8, pt: u8, ssrc: u32, total_len_bytes: usize) {
    debug_assert!(rc <= 31);
    debug_assert_eq!(total_len_bytes % 4, 0);
    // Byte 0: V=2 (top 2 bits), P=0, RC (low 5 bits) → 0b10_000000 | rc.
    buf.push(0x80 | (rc & 0x1F));
    buf.push(pt);
    // length: total length in 32-bit words minus one (RFC 3550 §6.4.1).
    let words = (total_len_bytes / 4) as u16;
    let length_field = words.saturating_sub(1);
    buf.extend_from_slice(&length_field.to_be_bytes());
    buf.extend_from_slice(&ssrc.to_be_bytes());
}

/// Fixed 4-byte header writer for packet types whose count field (SC) is *not*
/// followed by a sender SSRC in the header — SDES (§6.5) and BYE (§6.6). `cnt`
/// is the 5-bit source/chunk count, `pt` the packet type, `total_len_bytes`
/// the whole packet length (must be a multiple of 4).
fn write_count_header(buf: &mut Vec<u8>, cnt: u8, pt: u8, total_len_bytes: usize) {
    debug_assert!(cnt <= 31);
    debug_assert_eq!(total_len_bytes % 4, 0);
    buf.push(0x80 | (cnt & 0x1F)); // V=2, P=0, count
    buf.push(pt);
    let words = (total_len_bytes / 4) as u16;
    buf.extend_from_slice(&words.saturating_sub(1).to_be_bytes());
}

/// Build an RTCP Sender Report (PT = 200, RFC 3550 §6.4.1).
///
/// `ssrc` is the sender's own SSRC; `info` is the 20-byte sender-info
/// section; `blocks` are the reception report blocks for sources this
/// participant has heard (0..=31).
///
/// Returns the wire bytes (a multiple of 4 in length, no padding), or
/// [`RtcpError::TooManyReportBlocks`] if more than 31 blocks are supplied.
pub fn build_sender_report(
    ssrc: u32,
    info: &SenderInfo,
    blocks: &[ReceptionReportBlock],
) -> Result<Vec<u8>, RtcpError> {
    if blocks.len() > MAX_REPORT_BLOCKS {
        return Err(RtcpError::TooManyReportBlocks {
            count: blocks.len(),
        });
    }
    let total = RTCP_HEADER_LEN + SENDER_INFO_LEN + blocks.len() * REPORT_BLOCK_LEN;
    let mut buf = Vec::with_capacity(total);
    write_header(&mut buf, blocks.len() as u8, PT_SR, ssrc, total);
    // Sender info.
    buf.extend_from_slice(&info.ntp_timestamp.to_be_bytes());
    buf.extend_from_slice(&info.rtp_timestamp.to_be_bytes());
    buf.extend_from_slice(&info.packet_count.to_be_bytes());
    buf.extend_from_slice(&info.octet_count.to_be_bytes());
    for b in blocks {
        b.write(&mut buf);
    }
    debug_assert_eq!(buf.len(), total);
    Ok(buf)
}

/// Build an RTCP Receiver Report (PT = 201, RFC 3550 §6.4.2).
///
/// `ssrc` is the reporting participant's own SSRC; `blocks` are the
/// reception report blocks (0..=31). An empty `blocks` slice produces the
/// canonical empty RR (RC = 0) that RFC 3550 §6.4.2 says MUST head a
/// compound packet when there is nothing to report.
pub fn build_receiver_report(
    ssrc: u32,
    blocks: &[ReceptionReportBlock],
) -> Result<Vec<u8>, RtcpError> {
    if blocks.len() > MAX_REPORT_BLOCKS {
        return Err(RtcpError::TooManyReportBlocks {
            count: blocks.len(),
        });
    }
    let total = RTCP_HEADER_LEN + blocks.len() * REPORT_BLOCK_LEN;
    let mut buf = Vec::with_capacity(total);
    write_header(&mut buf, blocks.len() as u8, PT_RR, ssrc, total);
    for b in blocks {
        b.write(&mut buf);
    }
    debug_assert_eq!(buf.len(), total);
    Ok(buf)
}

/// Parse a single RTCP SR or RR packet from the front of `buf`.
///
/// Validates the version (must be 2), the PT (must be 200 or 201), and that
/// `buf` is long enough for the header + (sender info, if SR) + `RC` report
/// blocks. The `length` field is cross-checked against the RC-implied body
/// size; a mismatch is reported as [`RtcpError::LengthMismatch`].
///
/// Padding (P bit) and any profile-specific extension after the report
/// blocks are ignored by the parser — it returns only the SR/RR fields.
pub fn parse_report(buf: &[u8]) -> Result<RtcpReport, RtcpError> {
    if buf.len() < RTCP_HEADER_LEN {
        return Err(RtcpError::ShortHeader);
    }
    let b0 = buf[0];
    let version = (b0 >> 6) & 0x3;
    if version != 2 {
        return Err(RtcpError::BadVersion { value: version });
    }
    let rc = (b0 & 0x1F) as usize;
    let pt = buf[1];
    if pt != PT_SR && pt != PT_RR {
        return Err(RtcpError::UnexpectedPacketType { value: pt });
    }
    let stated_words = u16::from_be_bytes([buf[2], buf[3]]);
    let ssrc = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);

    let mut off = RTCP_HEADER_LEN;
    let sender_info = if pt == PT_SR {
        if buf.len() < off + SENDER_INFO_LEN {
            return Err(RtcpError::Truncated);
        }
        let ntp_timestamp = u64::from_be_bytes([
            buf[off],
            buf[off + 1],
            buf[off + 2],
            buf[off + 3],
            buf[off + 4],
            buf[off + 5],
            buf[off + 6],
            buf[off + 7],
        ]);
        let rtp_timestamp =
            u32::from_be_bytes([buf[off + 8], buf[off + 9], buf[off + 10], buf[off + 11]]);
        let packet_count =
            u32::from_be_bytes([buf[off + 12], buf[off + 13], buf[off + 14], buf[off + 15]]);
        let octet_count =
            u32::from_be_bytes([buf[off + 16], buf[off + 17], buf[off + 18], buf[off + 19]]);
        off += SENDER_INFO_LEN;
        Some(SenderInfo {
            ntp_timestamp,
            rtp_timestamp,
            packet_count,
            octet_count,
        })
    } else {
        None
    };

    let need_blocks = rc * REPORT_BLOCK_LEN;
    if buf.len() < off + need_blocks {
        return Err(RtcpError::Truncated);
    }
    let mut report_blocks = Vec::with_capacity(rc);
    for i in 0..rc {
        let start = off + i * REPORT_BLOCK_LEN;
        report_blocks.push(ReceptionReportBlock::read(&buf[start..]));
    }
    let body_end = off + need_blocks;

    // Cross-check the length field against the body we just parsed. The
    // stated length covers header + sender info + report blocks (+ any
    // padding/extension we don't model). We require it to be at least the
    // body size we computed; an exact-match check would reject valid
    // packets carrying a profile-specific extension, so we only flag the
    // case where the stated length is *smaller* than the parsed body.
    let actual_words = (body_end / 4) as u16;
    let stated_total_words = stated_words.saturating_add(1);
    if stated_total_words < actual_words {
        return Err(RtcpError::LengthMismatch {
            stated_words: stated_total_words,
            actual_words,
        });
    }

    Ok(RtcpReport {
        packet_type: pt,
        ssrc,
        sender_info,
        report_blocks,
    })
}

// ---------------------------------------------------------------------------
// SDES — Source Description (RFC 3550 §6.5)
// ---------------------------------------------------------------------------

/// SDES item type codes (RFC 3550 §6.5). Type 0 is the END marker that
/// terminates a chunk's item list.
pub mod sdes_type {
    /// END: item-list terminator (RFC 3550 §6.5 — "an item type of zero").
    pub const END: u8 = 0;
    /// CNAME: canonical end-point identifier (RFC 3550 §6.5.1) — mandatory.
    pub const CNAME: u8 = 1;
    /// NAME: user name (RFC 3550 §6.5.2).
    pub const NAME: u8 = 2;
    /// EMAIL: electronic mail address (RFC 3550 §6.5.3).
    pub const EMAIL: u8 = 3;
    /// PHONE: phone number (RFC 3550 §6.5.4).
    pub const PHONE: u8 = 4;
    /// LOC: geographic user location (RFC 3550 §6.5.5).
    pub const LOC: u8 = 5;
    /// TOOL: application or tool name (RFC 3550 §6.5.6).
    pub const TOOL: u8 = 6;
    /// NOTE: notice / status (RFC 3550 §6.5.7).
    pub const NOTE: u8 = 7;
    /// PRIV: private extensions (RFC 3550 §6.5.8).
    pub const PRIV: u8 = 8;
}

/// One SDES item (RFC 3550 §6.5): an 8-bit type, an 8-bit text-length, and the
/// text itself. The text is UTF-8 (US-ASCII is a subset); it is **not**
/// null-terminated and may be no longer than 255 octets.
///
/// `PRIV` (§6.5.8) carries a length-prefixed `prefix` string followed by a
/// `value` string that fills the rest of the item, so it is modelled as a
/// distinct variant rather than a single text blob.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SdesItem {
    /// Canonical name (`user@host` / `host`), §6.5.1. Mandatory in every
    /// compound packet (§6.1).
    Cname(String),
    /// User's real name, §6.5.2.
    Name(String),
    /// Email address, §6.5.3.
    Email(String),
    /// Phone number, §6.5.4.
    Phone(String),
    /// Geographic location, §6.5.5.
    Loc(String),
    /// Tool name / version, §6.5.6.
    Tool(String),
    /// Transient notice / status, §6.5.7.
    Note(String),
    /// Private extension (§6.5.8): a `prefix` name plus a `value` string.
    Priv { prefix: String, value: String },
}

impl SdesItem {
    /// The 8-bit SDES type code for this item (RFC 3550 §6.5).
    pub fn type_code(&self) -> u8 {
        match self {
            SdesItem::Cname(_) => sdes_type::CNAME,
            SdesItem::Name(_) => sdes_type::NAME,
            SdesItem::Email(_) => sdes_type::EMAIL,
            SdesItem::Phone(_) => sdes_type::PHONE,
            SdesItem::Loc(_) => sdes_type::LOC,
            SdesItem::Tool(_) => sdes_type::TOOL,
            SdesItem::Note(_) => sdes_type::NOTE,
            SdesItem::Priv { .. } => sdes_type::PRIV,
        }
    }

    /// Total on-wire length of this item including the 2-byte type+length
    /// header (the END terminator and 32-bit chunk padding are accounted for
    /// at the chunk level, not here).
    fn wire_len(&self) -> usize {
        2 + self.text_len()
    }

    /// Length of the item's text payload (the value of the 8-bit length byte).
    /// For `PRIV` this is `1 + prefix.len() + value.len()` (§6.5.8).
    fn text_len(&self) -> usize {
        match self {
            SdesItem::Cname(s)
            | SdesItem::Name(s)
            | SdesItem::Email(s)
            | SdesItem::Phone(s)
            | SdesItem::Loc(s)
            | SdesItem::Tool(s)
            | SdesItem::Note(s) => s.len(),
            SdesItem::Priv { prefix, value } => 1 + prefix.len() + value.len(),
        }
    }

    /// Validate the item's text fits the 8-bit length byte (RFC 3550 §6.5).
    fn validate(&self) -> Result<(), RtcpError> {
        match self {
            SdesItem::Priv { .. } => {
                let len = self.text_len();
                if len > MAX_TEXT_LEN {
                    return Err(RtcpError::PrivTooLong { len });
                }
            }
            _ => {
                let len = self.text_len();
                if len > MAX_TEXT_LEN {
                    return Err(RtcpError::TextTooLong { len });
                }
            }
        }
        Ok(())
    }

    fn write(&self, buf: &mut Vec<u8>) {
        buf.push(self.type_code());
        match self {
            SdesItem::Priv { prefix, value } => {
                // text_len() == 1 (prefix-len byte) + prefix + value.
                buf.push((1 + prefix.len() + value.len()) as u8);
                buf.push(prefix.len() as u8);
                buf.extend_from_slice(prefix.as_bytes());
                buf.extend_from_slice(value.as_bytes());
            }
            SdesItem::Cname(s)
            | SdesItem::Name(s)
            | SdesItem::Email(s)
            | SdesItem::Phone(s)
            | SdesItem::Loc(s)
            | SdesItem::Tool(s)
            | SdesItem::Note(s) => {
                buf.push(s.len() as u8);
                buf.extend_from_slice(s.as_bytes());
            }
        }
    }
}

/// One SDES chunk (RFC 3550 §6.5): an SSRC/CSRC identifier plus a list of
/// items. On the wire the item list is terminated by a zero type byte and the
/// chunk is padded with null octets to the next 32-bit boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SdesChunk {
    /// The SSRC/CSRC these items describe.
    pub ssrc: u32,
    /// The item list (a CNAME should be present per §6.1).
    pub items: Vec<SdesItem>,
}

impl SdesChunk {
    /// On-wire length of this chunk: 4 (SSRC) + items + the END byte, rounded
    /// up to the next multiple of 4 (RFC 3550 §6.5 32-bit chunk alignment).
    fn padded_len(&self) -> usize {
        let raw = 4 + self.items.iter().map(SdesItem::wire_len).sum::<usize>() + 1; // +1 END
        raw.div_ceil(4) * 4
    }

    fn write(&self, buf: &mut Vec<u8>) {
        let start = buf.len();
        buf.extend_from_slice(&self.ssrc.to_be_bytes());
        for item in &self.items {
            item.write(buf);
        }
        // END marker (item type 0) terminates the list, then pad with null
        // octets to the next 32-bit boundary (§6.5).
        buf.push(sdes_type::END);
        while (buf.len() - start) % 4 != 0 {
            buf.push(0);
        }
    }
}

/// Build an RTCP SDES packet (PT = 202, RFC 3550 §6.5).
///
/// `chunks` are the per-source descriptions (0..=31). A typical H.261
/// endpoint sends a single chunk for its own SSRC carrying at least a CNAME
/// item (§6.1). Each item's text must fit the 8-bit length byte (≤ 255
/// octets); each chunk is independently 32-bit-aligned with a trailing END
/// item-type-0 byte and null padding.
///
/// Returns the wire bytes (a multiple of 4 in length), or
/// [`RtcpError::TooManySources`] / [`RtcpError::TextTooLong`] /
/// [`RtcpError::PrivTooLong`].
pub fn build_sdes(chunks: &[SdesChunk]) -> Result<Vec<u8>, RtcpError> {
    if chunks.len() > MAX_SOURCE_COUNT {
        return Err(RtcpError::TooManySources {
            count: chunks.len(),
        });
    }
    for chunk in chunks {
        for item in &chunk.items {
            item.validate()?;
        }
    }
    let body: usize = chunks.iter().map(SdesChunk::padded_len).sum();
    let total = RTCP_HEADER_FIXED_LEN + body;
    let mut buf = Vec::with_capacity(total);
    write_count_header(&mut buf, chunks.len() as u8, PT_SDES, total);
    for chunk in chunks {
        chunk.write(&mut buf);
    }
    debug_assert_eq!(buf.len(), total);
    debug_assert_eq!(buf.len() % 4, 0);
    Ok(buf)
}

/// A parsed RTCP SDES packet (RFC 3550 §6.5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SdesPacket {
    /// The chunks, in wire order.
    pub chunks: Vec<SdesChunk>,
}

/// Parse a single RTCP SDES packet from the front of `buf`.
///
/// Validates V=2 and PT=202, then walks `SC` chunks. Each chunk's item list
/// is read until a zero type byte (END), then advanced to the next 32-bit
/// boundary. The 16-bit `length` field bounds how far the walk may read;
/// chunks may not cross it. Unknown item types are surfaced via
/// [`RtcpError::UnexpectedPacketType`]-free tolerance: an item type outside
/// 1..=8 is skipped over (consuming its declared length) so a forward-
/// compatible parser doesn't reject profile extensions, but its content is
/// not retained.
pub fn parse_sdes(buf: &[u8]) -> Result<SdesPacket, RtcpError> {
    if buf.len() < RTCP_HEADER_FIXED_LEN {
        return Err(RtcpError::ShortHeader);
    }
    let b0 = buf[0];
    if (b0 >> 6) & 0x3 != 2 {
        return Err(RtcpError::BadVersion {
            value: (b0 >> 6) & 0x3,
        });
    }
    if buf[1] != PT_SDES {
        return Err(RtcpError::UnexpectedPacketType { value: buf[1] });
    }
    let sc = (b0 & 0x1F) as usize;
    let stated_words = u16::from_be_bytes([buf[2], buf[3]]);
    // Body bytes the length field claims (header is fixed 4 bytes here; the
    // length covers the whole packet in words minus one, §6.4.1).
    let stated_total = (stated_words as usize + 1) * 4;
    let body_end = stated_total.min(buf.len());

    let mut off = RTCP_HEADER_FIXED_LEN;
    let mut chunks = Vec::with_capacity(sc);
    for _ in 0..sc {
        if off + 4 > body_end {
            return Err(RtcpError::Truncated);
        }
        let ssrc = u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        off += 4;
        let mut items = Vec::new();
        loop {
            if off >= body_end {
                return Err(RtcpError::Truncated);
            }
            let ty = buf[off];
            off += 1;
            if ty == sdes_type::END {
                break;
            }
            if off >= body_end {
                return Err(RtcpError::Truncated);
            }
            let len = buf[off] as usize;
            off += 1;
            if off + len > body_end {
                return Err(RtcpError::Truncated);
            }
            let text = &buf[off..off + len];
            off += len;
            match ty {
                sdes_type::CNAME => items.push(SdesItem::Cname(utf8_lossy(text))),
                sdes_type::NAME => items.push(SdesItem::Name(utf8_lossy(text))),
                sdes_type::EMAIL => items.push(SdesItem::Email(utf8_lossy(text))),
                sdes_type::PHONE => items.push(SdesItem::Phone(utf8_lossy(text))),
                sdes_type::LOC => items.push(SdesItem::Loc(utf8_lossy(text))),
                sdes_type::TOOL => items.push(SdesItem::Tool(utf8_lossy(text))),
                sdes_type::NOTE => items.push(SdesItem::Note(utf8_lossy(text))),
                // A PRIV item with a zero-length body has no prefix-length
                // byte and so cannot be decoded — skip it (it was already
                // consumed by the length advance above).
                sdes_type::PRIV if len >= 1 => {
                    let plen = (text[0] as usize).min(len - 1);
                    let prefix = utf8_lossy(&text[1..1 + plen]);
                    let value = utf8_lossy(&text[1 + plen..]);
                    items.push(SdesItem::Priv { prefix, value });
                }
                _ => { /* unknown item type: already skipped its length */ }
            }
        }
        // Advance over null padding to the next 32-bit boundary.
        while (off - RTCP_HEADER_FIXED_LEN) % 4 != 0 {
            if off >= body_end {
                break;
            }
            off += 1;
        }
        chunks.push(SdesChunk { ssrc, items });
    }
    Ok(SdesPacket { chunks })
}

/// Convenience: build a single-chunk SDES packet carrying only a CNAME item —
/// the minimal SDES every compound RTCP packet must include (RFC 3550 §6.1).
pub fn build_cname_sdes(ssrc: u32, cname: &str) -> Result<Vec<u8>, RtcpError> {
    build_sdes(&[SdesChunk {
        ssrc,
        items: vec![SdesItem::Cname(cname.to_string())],
    }])
}

// ---------------------------------------------------------------------------
// BYE — Goodbye (RFC 3550 §6.6)
// ---------------------------------------------------------------------------

/// A parsed RTCP BYE packet (RFC 3550 §6.6).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByePacket {
    /// The SSRC/CSRC identifiers leaving the session (1..=31; zero is valid
    /// but useless per §6.6).
    pub sources: Vec<u32>,
    /// Optional free-text reason for leaving (§6.6), e.g. "camera malfunction".
    pub reason: Option<String>,
}

/// Build an RTCP BYE packet (PT = 203, RFC 3550 §6.6).
///
/// `sources` are the SSRC/CSRC identifiers leaving (0..=31). `reason` is an
/// optional free-text explanation; it is prefixed by an 8-bit length byte and
/// the whole packet is padded with null octets to the next 32-bit boundary
/// (§6.6 — this padding is separate from the header P bit). The reason text
/// must be ≤ 255 octets.
pub fn build_bye(sources: &[u32], reason: Option<&str>) -> Result<Vec<u8>, RtcpError> {
    if sources.len() > MAX_SOURCE_COUNT {
        return Err(RtcpError::TooManySources {
            count: sources.len(),
        });
    }
    let reason_bytes = reason.map(str::as_bytes);
    if let Some(r) = reason_bytes {
        if r.len() > MAX_TEXT_LEN {
            return Err(RtcpError::TextTooLong { len: r.len() });
        }
    }
    // Body so far: SC * 4 (SSRC list) + optional (1 length byte + reason).
    let mut body = sources.len() * 4;
    if let Some(r) = reason_bytes {
        body += 1 + r.len();
    }
    // Pad the whole packet (header + body) to a 32-bit boundary.
    let unpadded = RTCP_HEADER_FIXED_LEN + body;
    let total = unpadded.div_ceil(4) * 4;
    let mut buf = Vec::with_capacity(total);
    write_count_header(&mut buf, sources.len() as u8, PT_BYE, total);
    for &s in sources {
        buf.extend_from_slice(&s.to_be_bytes());
    }
    if let Some(r) = reason_bytes {
        buf.push(r.len() as u8);
        buf.extend_from_slice(r);
    }
    while buf.len() < total {
        buf.push(0);
    }
    debug_assert_eq!(buf.len(), total);
    Ok(buf)
}

/// Parse a single RTCP BYE packet from the front of `buf` (RFC 3550 §6.6).
///
/// Validates V=2 and PT=203, reads `SC` SSRC/CSRC identifiers, then — if the
/// `length` field indicates bytes remain past the identifier list — reads the
/// 8-bit-prefixed reason string. Trailing null padding is ignored.
pub fn parse_bye(buf: &[u8]) -> Result<ByePacket, RtcpError> {
    if buf.len() < RTCP_HEADER_FIXED_LEN {
        return Err(RtcpError::ShortHeader);
    }
    let b0 = buf[0];
    if (b0 >> 6) & 0x3 != 2 {
        return Err(RtcpError::BadVersion {
            value: (b0 >> 6) & 0x3,
        });
    }
    if buf[1] != PT_BYE {
        return Err(RtcpError::UnexpectedPacketType { value: buf[1] });
    }
    let sc = (b0 & 0x1F) as usize;
    let stated_words = u16::from_be_bytes([buf[2], buf[3]]);
    let stated_total = (stated_words as usize + 1) * 4;
    let body_end = stated_total.min(buf.len());

    let mut off = RTCP_HEADER_FIXED_LEN;
    if off + sc * 4 > body_end {
        return Err(RtcpError::Truncated);
    }
    let mut sources = Vec::with_capacity(sc);
    for _ in 0..sc {
        sources.push(u32::from_be_bytes([
            buf[off],
            buf[off + 1],
            buf[off + 2],
            buf[off + 3],
        ]));
        off += 4;
    }
    // Optional reason string: present iff a length byte remains in the body.
    let reason = if off < body_end {
        let len = buf[off] as usize;
        off += 1;
        if off + len > body_end {
            return Err(RtcpError::Truncated);
        }
        Some(utf8_lossy(&buf[off..off + len]))
    } else {
        None
    };
    Ok(ByePacket { sources, reason })
}

// ---------------------------------------------------------------------------
// APP — Application-Defined RTCP Packet (RFC 3550 §6.7)
// ---------------------------------------------------------------------------
//
// Wire layout (§6.7):
//
// ```text
//  0                   1                   2                   3
//   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |V=2|P| subtype |   PT=APP=204  |             length            |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |                          SSRC/CSRC                            |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |                          name (ASCII)                         |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |                  application-dependent data                ...
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// ```
//
// The APP packet replaces the SR/RR's 5-bit RC field with a 5-bit `subtype`,
// adds a 4-byte ASCII `name`, and follows with optional application-defined
// data that "must be a multiple of 32 bits long" (§6.7). Names with
// unrecognised values "should be ignored" (§6.7) — this module exposes the
// raw bytes back to the caller so application-layer routing can do that.

/// A parsed RTCP APP (Application-Defined) packet (RFC 3550 §6.7).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppPacket {
    /// 5-bit subtype field. Per §6.7, "may be used as a subtype to allow a
    /// set of APP packets to be defined under one unique name, or for any
    /// application-dependent data."
    pub subtype: u8,
    /// SSRC/CSRC identifier of the source.
    pub ssrc: u32,
    /// The 4-byte ASCII name field. Held as raw bytes because §6.7 mandates
    /// "uppercase and lowercase characters treated as distinct" — case-folding
    /// would silently merge namespaces — and because we never want to panic on
    /// a malformed remote-host packet that put non-ASCII bytes here.
    pub name: [u8; APP_NAME_LEN],
    /// Application-dependent data. Always a multiple of 4 bytes long per §6.7;
    /// may be empty. The application interprets it; this module does not.
    pub data: Vec<u8>,
}

/// Build an RTCP APP (Application-Defined) packet (PT = 204, RFC 3550 §6.7).
///
/// `subtype` (0..=31) goes into the 5-bit RC-slot field. `ssrc` is the
/// source identifier. `name` is the mandatory 4-byte ASCII name (the function
/// rejects any other length). `data` is the application-dependent payload —
/// it must be a multiple of 4 bytes (§6.7); pass an empty slice for none.
///
/// Returns the wire bytes (a multiple of 4 in length, no padding), or one of:
/// * [`RtcpError::AppSubtypeOutOfRange`] — `subtype` > 31.
/// * [`RtcpError::AppNameWrongLength`] — `name.len() != 4`.
/// * [`RtcpError::AppDataNotAligned`] — `data.len() % 4 != 0`.
pub fn build_app(subtype: u8, ssrc: u32, name: &[u8], data: &[u8]) -> Result<Vec<u8>, RtcpError> {
    if subtype > APP_SUBTYPE_MAX {
        return Err(RtcpError::AppSubtypeOutOfRange { value: subtype });
    }
    if name.len() != APP_NAME_LEN {
        return Err(RtcpError::AppNameWrongLength { len: name.len() });
    }
    if data.len() % 4 != 0 {
        return Err(RtcpError::AppDataNotAligned { len: data.len() });
    }
    // Header (4) + SSRC (4) + name (4) + data.
    let total = RTCP_HEADER_FIXED_LEN + 4 + APP_NAME_LEN + data.len();
    debug_assert_eq!(total % 4, 0);
    let mut buf = Vec::with_capacity(total);
    // The header writer uses the same V/P/cnt|PT|length layout as SDES/BYE;
    // here `cnt` is the §6.7 subtype slot.
    write_count_header(&mut buf, subtype, PT_APP, total);
    buf.extend_from_slice(&ssrc.to_be_bytes());
    buf.extend_from_slice(name);
    buf.extend_from_slice(data);
    debug_assert_eq!(buf.len(), total);
    Ok(buf)
}

/// Parse a single RTCP APP packet from the front of `buf` (RFC 3550 §6.7).
///
/// Validates V=2 and PT=204, reads the subtype out of the 5-bit RC slot, then
/// reads SSRC, the 4-byte name, and the application-dependent data up to the
/// length field's stated end. The `length` field bounds how far the parser
/// will read into `buf`; any bytes past it are ignored. Returns
/// [`RtcpError::Truncated`] if the stated length runs past the buffer end,
/// [`RtcpError::ShortHeader`] if `buf` is shorter than the 12-byte
/// header + SSRC + name, [`RtcpError::BadVersion`] for V != 2, or
/// [`RtcpError::UnexpectedPacketType`] if PT is not 204.
pub fn parse_app(buf: &[u8]) -> Result<AppPacket, RtcpError> {
    // Minimum APP packet is the 4-byte header + SSRC (4) + name (4) = 12.
    const MIN_LEN: usize = RTCP_HEADER_FIXED_LEN + 4 + APP_NAME_LEN;
    if buf.len() < MIN_LEN {
        return Err(RtcpError::ShortHeader);
    }
    let b0 = buf[0];
    if (b0 >> 6) & 0x3 != 2 {
        return Err(RtcpError::BadVersion {
            value: (b0 >> 6) & 0x3,
        });
    }
    if buf[1] != PT_APP {
        return Err(RtcpError::UnexpectedPacketType { value: buf[1] });
    }
    let subtype = b0 & 0x1F;
    let stated_words = u16::from_be_bytes([buf[2], buf[3]]);
    let stated_total = (stated_words as usize + 1) * 4;
    if stated_total < MIN_LEN {
        // Length field claims fewer bytes than the mandatory APP header itself.
        return Err(RtcpError::LengthMismatch {
            stated_words: stated_total.div_ceil(4) as u16,
            actual_words: MIN_LEN.div_ceil(4) as u16,
        });
    }
    if buf.len() < stated_total {
        return Err(RtcpError::Truncated);
    }
    let ssrc = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let mut name = [0u8; APP_NAME_LEN];
    name.copy_from_slice(&buf[8..8 + APP_NAME_LEN]);
    let data_start = RTCP_HEADER_FIXED_LEN + 4 + APP_NAME_LEN;
    let data = buf[data_start..stated_total].to_vec();
    Ok(AppPacket {
        subtype,
        ssrc,
        name,
        data,
    })
}

// ---------------------------------------------------------------------------
// Compound packets (RFC 3550 §6.1)
// ---------------------------------------------------------------------------

/// One sub-packet recovered from a compound RTCP packet by [`parse_compound`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RtcpPacket {
    /// SR / RR report (PT 200 / 201).
    Report(RtcpReport),
    /// SDES source-description (PT 202).
    Sdes(SdesPacket),
    /// BYE goodbye (PT 203).
    Bye(ByePacket),
    /// APP application-defined (PT 204).
    App(AppPacket),
    /// A sub-packet whose PT this module does not model (e.g. RTPFB=205 or a
    /// profile extension). The raw bytes are returned so the caller can route
    /// or ignore it; the compound walk still advances correctly via its
    /// length field.
    Other { packet_type: u8, bytes: Vec<u8> },
}

/// Concatenate pre-built RTCP sub-packets into one compound-packet datagram
/// body (RFC 3550 §6.1). The sub-packets are emitted in the order given with
/// no separators; each must already be a self-delimiting RTCP packet (a
/// multiple of 4 bytes with a correct `length` field), which the builders in
/// this module guarantee.
///
/// §6.1 requires the first sub-packet to be a report (SR/RR) and an SDES
/// CNAME to be present; this function does not enforce that ordering — it is a
/// pure byte concatenator — but callers should respect it.
pub fn compound(packets: &[&[u8]]) -> Vec<u8> {
    let total: usize = packets.iter().map(|p| p.len()).sum();
    let mut buf = Vec::with_capacity(total);
    for p in packets {
        buf.extend_from_slice(p);
    }
    buf
}

/// Walk a received compound RTCP datagram into its individual sub-packets
/// (RFC 3550 §6.1). Each sub-packet is self-delimited by its 16-bit `length`
/// field (32-bit words minus one); the walk consumes `(length + 1) * 4` bytes
/// per sub-packet until the buffer is exhausted.
///
/// Returns [`RtcpError::ShortHeader`] if a sub-packet header is truncated and
/// [`RtcpError::Truncated`] if a stated length runs past the buffer end.
pub fn parse_compound(buf: &[u8]) -> Result<Vec<RtcpPacket>, RtcpError> {
    let mut out = Vec::new();
    let mut off = 0;
    while off < buf.len() {
        if buf.len() - off < RTCP_HEADER_FIXED_LEN {
            return Err(RtcpError::ShortHeader);
        }
        let pt = buf[off + 1];
        let words = u16::from_be_bytes([buf[off + 2], buf[off + 3]]) as usize;
        let sub_len = (words + 1) * 4;
        if off + sub_len > buf.len() {
            return Err(RtcpError::Truncated);
        }
        let sub = &buf[off..off + sub_len];
        let parsed = match pt {
            PT_SR | PT_RR => RtcpPacket::Report(parse_report(sub)?),
            PT_SDES => RtcpPacket::Sdes(parse_sdes(sub)?),
            PT_BYE => RtcpPacket::Bye(parse_bye(sub)?),
            PT_APP => RtcpPacket::App(parse_app(sub)?),
            other => RtcpPacket::Other {
                packet_type: other,
                bytes: sub.to_vec(),
            },
        };
        out.push(parsed);
        off += sub_len;
    }
    Ok(out)
}

/// Lossy UTF-8 decode for an SDES/BYE text field. RFC 3550 §6.5 mandates
/// UTF-8, but a malformed packet from the network must not panic the parser,
/// so invalid sequences are replaced with U+FFFD.
fn utf8_lossy(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sr_header_fields_and_length() {
        let info = SenderInfo {
            ntp_timestamp: 0xB44D_B705_2000_0000,
            rtp_timestamp: 90_000,
            packet_count: 7,
            octet_count: 4242,
        };
        let bytes = build_sender_report(0xDEAD_BEEF, &info, &[]).unwrap();
        // header(8) + sender info(20), no blocks.
        assert_eq!(bytes.len(), RTCP_HEADER_LEN + SENDER_INFO_LEN);
        // V=2, P=0, RC=0 → 0x80.
        assert_eq!(bytes[0], 0x80);
        assert_eq!(bytes[1], PT_SR);
        // length = words - 1 = (28/4) - 1 = 6.
        assert_eq!(u16::from_be_bytes([bytes[2], bytes[3]]), 6);
        assert_eq!(
            u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            0xDEAD_BEEF
        );
        // Sender info NTP MSW/LSW.
        assert_eq!(
            u64::from_be_bytes(bytes[8..16].try_into().unwrap()),
            0xB44D_B705_2000_0000
        );
        assert_eq!(
            u32::from_be_bytes(bytes[16..20].try_into().unwrap()),
            90_000
        );
        assert_eq!(u32::from_be_bytes(bytes[20..24].try_into().unwrap()), 7);
        assert_eq!(u32::from_be_bytes(bytes[24..28].try_into().unwrap()), 4242);
    }

    #[test]
    fn rr_empty_is_canonical() {
        let bytes = build_receiver_report(0x1234_5678, &[]).unwrap();
        assert_eq!(bytes.len(), RTCP_HEADER_LEN);
        assert_eq!(bytes[0], 0x80); // V=2, RC=0
        assert_eq!(bytes[1], PT_RR);
        // length = (8/4) - 1 = 1.
        assert_eq!(u16::from_be_bytes([bytes[2], bytes[3]]), 1);
        assert_eq!(
            u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            0x1234_5678
        );
    }

    #[test]
    fn sr_round_trip_with_blocks() {
        let info = SenderInfo {
            ntp_timestamp: SenderInfo::ntp_from_parts(0xB705_2000 >> 16, 0x2000_0000),
            rtp_timestamp: 12345,
            packet_count: 100,
            octet_count: 200_000,
        };
        let blocks = vec![
            ReceptionReportBlock {
                ssrc: 0xAAAA_AAAA,
                fraction_lost: 13,
                cumulative_lost: 4096,
                extended_highest_seq: 0x0001_2345,
                jitter: 88,
                last_sr: 0xB705_2000,
                delay_since_last_sr: 0x0005_4000,
            },
            ReceptionReportBlock {
                ssrc: 0xBBBB_BBBB,
                fraction_lost: 0,
                cumulative_lost: -3, // duplicates ⇒ negative
                extended_highest_seq: 0x0002_0000,
                jitter: 0,
                last_sr: 0,
                delay_since_last_sr: 0,
            },
        ];
        let bytes = build_sender_report(0xCAFE_F00D, &info, &blocks).unwrap();
        assert_eq!(
            bytes.len(),
            RTCP_HEADER_LEN + SENDER_INFO_LEN + 2 * REPORT_BLOCK_LEN
        );
        // RC must reflect 2 blocks.
        assert_eq!(bytes[0] & 0x1F, 2);
        let parsed = parse_report(&bytes).unwrap();
        assert_eq!(parsed.packet_type, PT_SR);
        assert_eq!(parsed.ssrc, 0xCAFE_F00D);
        assert_eq!(parsed.sender_info, Some(info));
        assert_eq!(parsed.report_blocks, blocks);
        // Negative cumulative-lost survives the 24-bit two's-complement
        // round trip.
        assert_eq!(parsed.report_blocks[1].cumulative_lost, -3);
    }

    #[test]
    fn rr_round_trip_with_blocks() {
        let blocks = vec![ReceptionReportBlock {
            ssrc: 0x0102_0304,
            fraction_lost: 255,
            cumulative_lost: 0x007F_FFFF, // max positive 24-bit
            extended_highest_seq: 0xFFFF_FFFF,
            jitter: 0xDEAD_BEEF,
            last_sr: 0x1122_3344,
            delay_since_last_sr: 0x5566_7788,
        }];
        let bytes = build_receiver_report(0x9999_9999, &blocks).unwrap();
        let parsed = parse_report(&bytes).unwrap();
        assert_eq!(parsed.packet_type, PT_RR);
        assert_eq!(parsed.ssrc, 0x9999_9999);
        assert_eq!(parsed.sender_info, None);
        assert_eq!(parsed.report_blocks, blocks);
        assert_eq!(parsed.report_blocks[0].cumulative_lost, 0x007F_FFFF);
    }

    #[test]
    fn cumulative_lost_min_negative_round_trips() {
        // Most-negative 24-bit two's-complement value: -8388608.
        let blocks = vec![ReceptionReportBlock {
            ssrc: 1,
            cumulative_lost: -8_388_608,
            ..Default::default()
        }];
        let bytes = build_receiver_report(2, &blocks).unwrap();
        let parsed = parse_report(&bytes).unwrap();
        assert_eq!(parsed.report_blocks[0].cumulative_lost, -8_388_608);
    }

    #[test]
    fn build_rejects_more_than_31_blocks() {
        let blocks = vec![ReceptionReportBlock::default(); 32];
        assert_eq!(
            build_receiver_report(0, &blocks),
            Err(RtcpError::TooManyReportBlocks { count: 32 })
        );
        let info = SenderInfo::default();
        assert_eq!(
            build_sender_report(0, &info, &blocks),
            Err(RtcpError::TooManyReportBlocks { count: 32 })
        );
    }

    #[test]
    fn exactly_31_blocks_is_allowed() {
        let blocks = vec![ReceptionReportBlock::default(); MAX_REPORT_BLOCKS];
        let bytes = build_receiver_report(0, &blocks).unwrap();
        assert_eq!(bytes[0] & 0x1F, 31);
        let parsed = parse_report(&bytes).unwrap();
        assert_eq!(parsed.report_blocks.len(), 31);
    }

    #[test]
    fn parse_rejects_short_header() {
        for n in 0..RTCP_HEADER_LEN {
            assert_eq!(parse_report(&vec![0x80u8; n]), Err(RtcpError::ShortHeader));
        }
    }

    #[test]
    fn parse_rejects_bad_version() {
        let mut bytes = build_receiver_report(0, &[]).unwrap();
        bytes[0] = 0x40; // V=1
        assert_eq!(
            parse_report(&bytes),
            Err(RtcpError::BadVersion { value: 1 })
        );
    }

    #[test]
    fn parse_rejects_unknown_pt() {
        let mut bytes = build_receiver_report(0, &[]).unwrap();
        bytes[1] = 202; // SDES, not SR/RR
        assert_eq!(
            parse_report(&bytes),
            Err(RtcpError::UnexpectedPacketType { value: 202 })
        );
    }

    #[test]
    fn parse_rejects_truncated_sender_info() {
        let info = SenderInfo::default();
        let mut bytes = build_sender_report(0, &info, &[]).unwrap();
        bytes.truncate(RTCP_HEADER_LEN + 4); // cut into sender info
        assert_eq!(parse_report(&bytes), Err(RtcpError::Truncated));
    }

    #[test]
    fn parse_rejects_truncated_report_block() {
        let blocks = vec![ReceptionReportBlock::default(); 2];
        let mut bytes = build_receiver_report(0, &blocks).unwrap();
        // Drop the last block's final 4 bytes.
        bytes.truncate(bytes.len() - 4);
        assert_eq!(parse_report(&bytes), Err(RtcpError::Truncated));
    }

    #[test]
    fn parse_tolerates_trailing_extension() {
        // A profile-specific extension after the report blocks must not
        // make the parser reject the packet; it parses the SR/RR fields and
        // ignores the tail. We bump the length field to cover the extra
        // bytes, mirroring how a real sender would set it.
        let info = SenderInfo::default();
        let mut bytes = build_sender_report(0x55, &info, &[]).unwrap();
        bytes.extend_from_slice(&[0xAB, 0xCD, 0xEF, 0x01]); // 4-byte extension
        let words = (bytes.len() / 4) as u16;
        bytes[2..4].copy_from_slice(&(words - 1).to_be_bytes());
        let parsed = parse_report(&bytes).unwrap();
        assert_eq!(parsed.ssrc, 0x55);
        assert!(parsed.report_blocks.is_empty());
    }

    #[test]
    fn parse_flags_length_too_small() {
        let blocks = vec![ReceptionReportBlock::default(); 1];
        let mut bytes = build_receiver_report(0, &blocks).unwrap();
        // Force the length field to claim a single 32-bit word (header
        // only), which is smaller than the actual body of header + 1 block.
        bytes[2..4].copy_from_slice(&0u16.to_be_bytes()); // stated total words = 1
        match parse_report(&bytes) {
            Err(RtcpError::LengthMismatch { .. }) => {}
            other => panic!("expected LengthMismatch, got {other:?}"),
        }
    }

    #[test]
    fn ntp_from_parts_packs_high_low() {
        assert_eq!(
            SenderInfo::ntp_from_parts(0xB44D_B705, 0x2000_0000),
            0xB44D_B705_2000_0000
        );
    }

    // ---- SDES (§6.5) -----------------------------------------------------

    #[test]
    fn sdes_cname_header_and_alignment() {
        let bytes = build_cname_sdes(0xDEAD_BEEF, "alice@example.com").unwrap();
        // V=2, P=0, SC=1 → 0x81.
        assert_eq!(bytes[0], 0x81);
        assert_eq!(bytes[1], PT_SDES);
        // Body: SSRC(4) + type(1) + len(1) + 17 text + END(1) = 24, padded to
        // 24 (already a multiple of 4). Packet = 4 header + 24 = 28 bytes.
        assert_eq!(bytes.len(), 28);
        assert_eq!(bytes.len() % 4, 0);
        // length = words - 1 = 28/4 - 1 = 6.
        assert_eq!(u16::from_be_bytes([bytes[2], bytes[3]]), 6);
        // SSRC follows the 4-byte fixed header.
        assert_eq!(
            u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            0xDEAD_BEEF
        );
        assert_eq!(bytes[8], sdes_type::CNAME);
        assert_eq!(bytes[9], 17);
    }

    #[test]
    fn sdes_round_trip_all_item_types() {
        let chunk = SdesChunk {
            ssrc: 0xCAFE_F00D,
            items: vec![
                SdesItem::Cname("doe@sleepy.example.com".to_string()),
                SdesItem::Name("John Doe".to_string()),
                SdesItem::Email("John.Doe@example.com".to_string()),
                SdesItem::Phone("+1 908 555 1212".to_string()),
                SdesItem::Loc("Murray Hill, New Jersey".to_string()),
                SdesItem::Tool("oxideav-h261".to_string()),
                SdesItem::Note("on the phone".to_string()),
                SdesItem::Priv {
                    prefix: "x-oxideav".to_string(),
                    value: "round101".to_string(),
                },
            ],
        };
        let bytes = build_sdes(std::slice::from_ref(&chunk)).unwrap();
        assert_eq!(bytes.len() % 4, 0);
        let parsed = parse_sdes(&bytes).unwrap();
        assert_eq!(parsed.chunks.len(), 1);
        assert_eq!(parsed.chunks[0], chunk);
    }

    #[test]
    fn sdes_multiple_chunks_each_aligned() {
        let chunks = vec![
            SdesChunk {
                ssrc: 0x1111_1111,
                items: vec![SdesItem::Cname("a@h".to_string())],
            },
            SdesChunk {
                ssrc: 0x2222_2222,
                items: vec![SdesItem::Cname("bb@hh".to_string())],
            },
        ];
        let bytes = build_sdes(&chunks).unwrap();
        assert_eq!(bytes[0] & 0x1F, 2); // SC = 2
        assert_eq!(bytes.len() % 4, 0);
        let parsed = parse_sdes(&bytes).unwrap();
        assert_eq!(parsed.chunks, chunks);
    }

    #[test]
    fn sdes_empty_chunk_is_four_null_octets() {
        // A chunk with zero items: SSRC(4) + END(1) padded to 8 bytes.
        let chunks = vec![SdesChunk {
            ssrc: 0x0102_0304,
            items: vec![],
        }];
        let bytes = build_sdes(&chunks).unwrap();
        // 4 header + 8 chunk = 12 bytes.
        assert_eq!(bytes.len(), 12);
        let parsed = parse_sdes(&bytes).unwrap();
        assert_eq!(parsed.chunks[0].ssrc, 0x0102_0304);
        assert!(parsed.chunks[0].items.is_empty());
    }

    #[test]
    fn sdes_rejects_more_than_31_chunks() {
        let chunks = vec![
            SdesChunk {
                ssrc: 0,
                items: vec![]
            };
            32
        ];
        assert_eq!(
            build_sdes(&chunks),
            Err(RtcpError::TooManySources { count: 32 })
        );
    }

    #[test]
    fn sdes_rejects_text_over_255() {
        let chunks = vec![SdesChunk {
            ssrc: 1,
            items: vec![SdesItem::Cname("x".repeat(256))],
        }];
        assert_eq!(
            build_sdes(&chunks),
            Err(RtcpError::TextTooLong { len: 256 })
        );
    }

    #[test]
    fn sdes_priv_rejects_over_255() {
        // prefix(200) + value(60) + 1 prefix-len byte = 261 > 255.
        let chunks = vec![SdesChunk {
            ssrc: 1,
            items: vec![SdesItem::Priv {
                prefix: "p".repeat(200),
                value: "v".repeat(60),
            }],
        }];
        assert_eq!(
            build_sdes(&chunks),
            Err(RtcpError::PrivTooLong { len: 261 })
        );
    }

    #[test]
    fn sdes_max_length_cname_round_trips() {
        let chunks = vec![SdesChunk {
            ssrc: 7,
            items: vec![SdesItem::Cname("z".repeat(255))],
        }];
        let bytes = build_sdes(&chunks).unwrap();
        let parsed = parse_sdes(&bytes).unwrap();
        assert_eq!(parsed.chunks[0].items[0], SdesItem::Cname("z".repeat(255)));
    }

    #[test]
    fn sdes_parse_skips_unknown_item_type() {
        // Hand-craft a chunk with an unknown item type 99 between two CNAMEs.
        let mut buf = Vec::new();
        write_count_header(&mut buf, 1, PT_SDES, 0); // placeholder length
        let chunk_start = buf.len();
        buf.extend_from_slice(&0xABCD_1234u32.to_be_bytes());
        // CNAME "hi"
        buf.push(sdes_type::CNAME);
        buf.push(2);
        buf.extend_from_slice(b"hi");
        // unknown type 99, 3 bytes
        buf.push(99);
        buf.push(3);
        buf.extend_from_slice(b"xyz");
        buf.push(sdes_type::END);
        while (buf.len() - chunk_start) % 4 != 0 {
            buf.push(0);
        }
        let words = (buf.len() / 4) as u16;
        buf[2..4].copy_from_slice(&(words - 1).to_be_bytes());
        let parsed = parse_sdes(&buf).unwrap();
        // Only the CNAME survives; the unknown item is skipped.
        assert_eq!(
            parsed.chunks[0].items,
            vec![SdesItem::Cname("hi".to_string())]
        );
    }

    #[test]
    fn sdes_parse_rejects_wrong_pt() {
        let mut bytes = build_cname_sdes(1, "a@b").unwrap();
        bytes[1] = PT_BYE;
        assert_eq!(
            parse_sdes(&bytes),
            Err(RtcpError::UnexpectedPacketType { value: PT_BYE })
        );
    }

    // ---- BYE (§6.6) ------------------------------------------------------

    #[test]
    fn bye_no_reason_header_and_length() {
        let bytes = build_bye(&[0xDEAD_BEEF, 0xCAFE_F00D], None).unwrap();
        // V=2, P=0, SC=2 → 0x82.
        assert_eq!(bytes[0], 0x82);
        assert_eq!(bytes[1], PT_BYE);
        // 4 header + 8 (two SSRCs) = 12 bytes; length = 12/4 - 1 = 2.
        assert_eq!(bytes.len(), 12);
        assert_eq!(u16::from_be_bytes([bytes[2], bytes[3]]), 2);
        let parsed = parse_bye(&bytes).unwrap();
        assert_eq!(parsed.sources, vec![0xDEAD_BEEF, 0xCAFE_F00D]);
        assert_eq!(parsed.reason, None);
    }

    #[test]
    fn bye_with_reason_round_trips_and_pads() {
        let bytes = build_bye(&[0x1234_5678], Some("camera malfunction")).unwrap();
        // 4 header + 4 SSRC + 1 len + 18 reason = 27, padded to 28.
        assert_eq!(bytes.len(), 28);
        assert_eq!(bytes.len() % 4, 0);
        let parsed = parse_bye(&bytes).unwrap();
        assert_eq!(parsed.sources, vec![0x1234_5678]);
        assert_eq!(parsed.reason.as_deref(), Some("camera malfunction"));
    }

    #[test]
    fn bye_empty_reason_is_distinguished_from_none() {
        // A zero-length reason still emits the length byte, so the parser
        // returns Some("") rather than None.
        let bytes = build_bye(&[1], Some("")).unwrap();
        let parsed = parse_bye(&bytes).unwrap();
        assert_eq!(parsed.reason.as_deref(), Some(""));
    }

    #[test]
    fn bye_rejects_more_than_31_sources() {
        let sources = vec![0u32; 32];
        assert_eq!(
            build_bye(&sources, None),
            Err(RtcpError::TooManySources { count: 32 })
        );
    }

    #[test]
    fn bye_rejects_reason_over_255() {
        assert_eq!(
            build_bye(&[1], Some(&"r".repeat(256))),
            Err(RtcpError::TextTooLong { len: 256 })
        );
    }

    #[test]
    fn bye_parse_rejects_truncated_sources() {
        let mut bytes = build_bye(&[1, 2], None).unwrap();
        bytes.truncate(RTCP_HEADER_FIXED_LEN + 4); // only one SSRC present
        assert_eq!(parse_bye(&bytes), Err(RtcpError::Truncated));
    }

    // ---- APP — Application-Defined (§6.7) -------------------------------

    #[test]
    fn app_header_layout_empty_data() {
        // The minimal APP packet has no application-dependent data; length
        // covers the 12-byte header + SSRC + name.
        let bytes = build_app(0, 0xDEAD_BEEF, b"TEST", &[]).unwrap();
        assert_eq!(bytes.len(), 12);
        assert_eq!(bytes[0], 0x80); // V=2, P=0, subtype=0
        assert_eq!(bytes[1], PT_APP);
        // length = (12 / 4) - 1 = 2.
        assert_eq!(u16::from_be_bytes([bytes[2], bytes[3]]), 2);
        assert_eq!(
            u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            0xDEAD_BEEF
        );
        assert_eq!(&bytes[8..12], b"TEST");
    }

    #[test]
    fn app_round_trip_with_data() {
        // 8 bytes of application data → packet len = 12 + 8 = 20.
        let data = [0xAA, 0xBB, 0xCC, 0xDD, 0x01, 0x02, 0x03, 0x04];
        let bytes = build_app(7, 0x1234_5678, b"oxAV", &data).unwrap();
        assert_eq!(bytes.len(), 20);
        // subtype goes into the low 5 bits of byte 0.
        assert_eq!(bytes[0] & 0x1F, 7);
        let parsed = parse_app(&bytes).unwrap();
        assert_eq!(parsed.subtype, 7);
        assert_eq!(parsed.ssrc, 0x1234_5678);
        assert_eq!(&parsed.name, b"oxAV");
        assert_eq!(parsed.data, data);
    }

    #[test]
    fn app_subtype_31_max_is_allowed() {
        // §6.7: subtype is 5 bits. 31 is the inclusive max.
        let bytes = build_app(31, 0, b"NAME", &[]).unwrap();
        assert_eq!(bytes[0] & 0x1F, 31);
        let parsed = parse_app(&bytes).unwrap();
        assert_eq!(parsed.subtype, 31);
    }

    #[test]
    fn app_rejects_subtype_over_31() {
        assert_eq!(
            build_app(32, 0, b"NAME", &[]),
            Err(RtcpError::AppSubtypeOutOfRange { value: 32 })
        );
        assert_eq!(
            build_app(255, 0, b"NAME", &[]),
            Err(RtcpError::AppSubtypeOutOfRange { value: 255 })
        );
    }

    #[test]
    fn app_rejects_name_not_4_octets() {
        for bad in [
            b"".as_ref(),
            b"x".as_ref(),
            b"xyz".as_ref(),
            b"toolong".as_ref(),
        ] {
            assert_eq!(
                build_app(0, 0, bad, &[]),
                Err(RtcpError::AppNameWrongLength { len: bad.len() })
            );
        }
    }

    #[test]
    fn app_rejects_data_not_32_bit_aligned() {
        for bad_len in [1, 2, 3, 5, 6, 7, 9] {
            let data = vec![0u8; bad_len];
            assert_eq!(
                build_app(0, 0, b"NAME", &data),
                Err(RtcpError::AppDataNotAligned { len: bad_len })
            );
        }
    }

    #[test]
    fn app_round_trip_large_data() {
        // 1024 bytes of payload is well within a single RTCP packet budget.
        let mut data = Vec::with_capacity(1024);
        for i in 0..1024 {
            data.push((i & 0xFF) as u8);
        }
        let bytes = build_app(3, 0xCAFE_F00D, b"BIG!", &data).unwrap();
        let parsed = parse_app(&bytes).unwrap();
        assert_eq!(parsed.subtype, 3);
        assert_eq!(parsed.ssrc, 0xCAFE_F00D);
        assert_eq!(&parsed.name, b"BIG!");
        assert_eq!(parsed.data, data);
    }

    #[test]
    fn app_name_is_byte_exact_not_case_folded() {
        // §6.7: "uppercase and lowercase characters treated as distinct."
        let a = build_app(0, 0, b"NaMe", &[]).unwrap();
        let b = build_app(0, 0, b"name", &[]).unwrap();
        assert_ne!(a, b);
        let pa = parse_app(&a).unwrap();
        let pb = parse_app(&b).unwrap();
        assert_ne!(pa.name, pb.name);
    }

    #[test]
    fn app_parse_rejects_short_header() {
        // 11 bytes is one byte short of header + SSRC + name (12).
        let bytes = [0x80, PT_APP, 0x00, 0x02, 0, 0, 0, 0, b'X', b'X', b'X'];
        assert_eq!(parse_app(&bytes), Err(RtcpError::ShortHeader));
    }

    #[test]
    fn app_parse_rejects_bad_version() {
        let mut bytes = build_app(0, 0, b"NAME", &[]).unwrap();
        // Clobber V to 0 (top two bits).
        bytes[0] &= 0x3F;
        assert_eq!(parse_app(&bytes), Err(RtcpError::BadVersion { value: 0 }));
    }

    #[test]
    fn app_parse_rejects_wrong_pt() {
        let mut bytes = build_app(0, 0, b"NAME", &[]).unwrap();
        bytes[1] = PT_BYE;
        assert_eq!(
            parse_app(&bytes),
            Err(RtcpError::UnexpectedPacketType { value: PT_BYE })
        );
    }

    #[test]
    fn app_parse_rejects_truncated_when_length_exceeds_buffer() {
        let mut bytes = build_app(0, 0, b"NAME", &[0u8; 8]).unwrap();
        // Lie about the length, claiming 6 words (28 bytes) when only 20 are
        // present.
        bytes[2..4].copy_from_slice(&6u16.to_be_bytes());
        assert_eq!(parse_app(&bytes), Err(RtcpError::Truncated));
    }

    #[test]
    fn app_parse_ignores_trailing_bytes_past_stated_length() {
        // §6.7 packets are self-delimited by their length field; bytes past
        // it (e.g. another stacked compound sub-packet) must not be consumed.
        let mut bytes = build_app(2, 0xAA, b"app1", &[1, 2, 3, 4]).unwrap();
        let original_len = bytes.len();
        bytes.extend_from_slice(&[0xFF; 8]);
        let parsed = parse_app(&bytes).unwrap();
        assert_eq!(parsed.subtype, 2);
        assert_eq!(parsed.data, vec![1, 2, 3, 4]);
        assert_eq!(original_len, RTCP_HEADER_FIXED_LEN + 4 + APP_NAME_LEN + 4);
    }

    // ---- Compound packets (§6.1) ----------------------------------------

    #[test]
    fn compound_rr_sdes_bye_round_trips() {
        // The canonical minimal compound: empty RR + SDES CNAME, then a BYE.
        let rr = build_receiver_report(0xAAAA_AAAA, &[]).unwrap();
        let sdes = build_cname_sdes(0xAAAA_AAAA, "me@host").unwrap();
        let bye = build_bye(&[0xAAAA_AAAA], Some("leaving")).unwrap();
        let datagram = compound(&[&rr, &sdes, &bye]);
        assert_eq!(datagram.len(), rr.len() + sdes.len() + bye.len());

        let parsed = parse_compound(&datagram).unwrap();
        assert_eq!(parsed.len(), 3);
        match &parsed[0] {
            RtcpPacket::Report(r) => {
                assert_eq!(r.packet_type, PT_RR);
                assert_eq!(r.ssrc, 0xAAAA_AAAA);
            }
            other => panic!("expected RR, got {other:?}"),
        }
        match &parsed[1] {
            RtcpPacket::Sdes(s) => {
                assert_eq!(
                    s.chunks[0].items,
                    vec![SdesItem::Cname("me@host".to_string())]
                );
            }
            other => panic!("expected SDES, got {other:?}"),
        }
        match &parsed[2] {
            RtcpPacket::Bye(b) => {
                assert_eq!(b.sources, vec![0xAAAA_AAAA]);
                assert_eq!(b.reason.as_deref(), Some("leaving"));
            }
            other => panic!("expected BYE, got {other:?}"),
        }
    }

    #[test]
    fn compound_sr_with_block_then_sdes() {
        let info = SenderInfo {
            ntp_timestamp: 0xB44D_B705_2000_0000,
            rtp_timestamp: 90_000,
            packet_count: 3,
            octet_count: 1000,
        };
        let block = ReceptionReportBlock {
            ssrc: 0xBBBB_BBBB,
            fraction_lost: 5,
            cumulative_lost: 2,
            ..Default::default()
        };
        let sr = build_sender_report(0x1357_9BDF, &info, &[block]).unwrap();
        let sdes = build_cname_sdes(0x1357_9BDF, "cam@studio").unwrap();
        let datagram = compound(&[&sr, &sdes]);
        let parsed = parse_compound(&datagram).unwrap();
        assert_eq!(parsed.len(), 2);
        match &parsed[0] {
            RtcpPacket::Report(r) => {
                assert_eq!(r.packet_type, PT_SR);
                assert_eq!(r.sender_info, Some(info));
                assert_eq!(r.report_blocks, vec![block]);
            }
            other => panic!("expected SR, got {other:?}"),
        }
    }

    #[test]
    fn compound_rr_sdes_app_round_trips() {
        // RR + SDES CNAME + APP — the APP variant must come back typed, not
        // as `Other`, now that the parser models PT=204.
        let rr = build_receiver_report(0xBBBB_BBBB, &[]).unwrap();
        let sdes = build_cname_sdes(0xBBBB_BBBB, "x@y").unwrap();
        let app = build_app(15, 0xBBBB_BBBB, b"OXAV", &[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();
        let datagram = compound(&[&rr, &sdes, &app]);
        let parsed = parse_compound(&datagram).unwrap();
        assert_eq!(parsed.len(), 3);
        match &parsed[2] {
            RtcpPacket::App(a) => {
                assert_eq!(a.subtype, 15);
                assert_eq!(a.ssrc, 0xBBBB_BBBB);
                assert_eq!(&a.name, b"OXAV");
                assert_eq!(a.data, vec![0xDE, 0xAD, 0xBE, 0xEF]);
            }
            other => panic!("expected App, got {other:?}"),
        }
    }

    #[test]
    fn compound_preserves_unknown_packet_type() {
        // A sub-packet whose PT this module does not model (e.g. RTPFB=205,
        // RFC 4585) should round-trip as `Other` with its raw bytes, and the
        // walk must still advance past it.
        let rr = build_receiver_report(1, &[]).unwrap();
        // Hand-built RTPFB-like: V=2, FMT=0, PT=205, length=1 (2 words), SSRC.
        let fb = vec![0x80, 205, 0x00, 0x01, 0, 0, 0, 9];
        let datagram = compound(&[&rr, &fb]);
        let parsed = parse_compound(&datagram).unwrap();
        assert_eq!(parsed.len(), 2);
        match &parsed[1] {
            RtcpPacket::Other { packet_type, bytes } => {
                assert_eq!(*packet_type, 205);
                assert_eq!(bytes, &fb);
            }
            other => panic!("expected Other, got {other:?}"),
        }
    }

    #[test]
    fn compound_parse_rejects_truncated_subpacket() {
        let rr = build_receiver_report(1, &[]).unwrap();
        let mut datagram = compound(&[&rr]);
        // Claim a longer length than the buffer provides.
        datagram[2..4].copy_from_slice(&5u16.to_be_bytes());
        assert_eq!(parse_compound(&datagram), Err(RtcpError::Truncated));
    }
}
