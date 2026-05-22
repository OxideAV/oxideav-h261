//! RTCP Sender Report (SR) / Receiver Report (RR) builders — RFC 3550 §6.4.
//!
//! The [`crate::rtp::RtpPacketizer`] puts H.261 media on the wire as RFC 3550
//! §5.1 RTP data packets. Every RTP session also runs an RTCP control channel
//! (RFC 3550 §6) on which each participant periodically reports transmission
//! and reception statistics. This module builds and parses the two report
//! packets an H.261 endpoint emits:
//!
//! * **SR** (Sender Report, PT = 200, §6.4.1) — sent by a participant that
//!   has transmitted data during the interval. It carries a 20-byte *sender
//!   info* section (NTP + RTP timestamps, sender's packet & octet counts)
//!   plus zero or more reception report blocks.
//! * **RR** (Receiver Report, PT = 201, §6.4.2) — identical to an SR minus
//!   the sender-info section. An empty RR (RC = 0) is the canonical "I have
//!   nothing to report" packet a pure receiver puts at the head of its
//!   compound RTCP packet.
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
//! conformant SR (using the packet/octet counters the packetiser already
//! tracks) and a receiver can emit/parse RR blocks. Scheduling (the §6.2
//! transmission-interval algorithm), the compound-packet SDES/CNAME item,
//! BYE, and the reception-statistics estimators (§A.3 loss fraction, §A.8
//! jitter) are deliberately out of scope — those are session-management
//! concerns above the codec, and RFC 3550 specifies them independently of
//! the report wire format implemented here. The builder accepts pre-computed
//! statistics; the parser surfaces them back.

/// PT code for an RTCP Sender Report (RFC 3550 §6.4.1).
pub const PT_SR: u8 = 200;
/// PT code for an RTCP Receiver Report (RFC 3550 §6.4.2).
pub const PT_RR: u8 = 201;

/// Length in bytes of the common RTCP report header (V/P/RC, PT, length,
/// SSRC of sender). RFC 3550 §6.4.1: "the header, is 8 octets long".
pub const RTCP_HEADER_LEN: usize = 8;

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
}
