//! H.261 SDP media-type and `fmtp`/`rtpmap` parameter mapping — RFC 4587
//! §6.1.1 and §6.2.
//!
//! RFC 4587 registers the `video/H261` media type and maps its three
//! optional parameters into the Session Description Protocol (SDP) lines an
//! H.261 endpoint exchanges during session setup. This module implements
//! **that signalling layer** — the wire format of the SDP `a=rtpmap` and
//! `a=fmtp` attribute lines — sitting alongside [`crate::rtp`] (the RTP
//! payload format) and [`crate::rtcp`] (the control channel). Transport,
//! the SDP offer/answer state machine itself, and the rest of the session
//! description (`v=`, `o=`, `c=`, `t=`, timing) remain caller-side.
//!
//! ## The three optional parameters (§6.1.1)
//!
//! * **CIF** — maximum supported frame rate for CIF resolution. The value
//!   is an integer 1..=4 and means "the maximum rate is `29.97 / value`
//!   frames per second". Absent ⇒ CIF is not advertised.
//! * **QCIF** — same, for QCIF resolution. Value 1..=4, rate
//!   `29.97 / value` fps. Absent ⇒ QCIF is not advertised.
//! * **D** — support for still-image graphics per H.261 Annex D. If
//!   supported the value SHALL be `"1"`; if not supported the parameter
//!   SHOULD be omitted or SHALL have the value `"0"` (§6.1.1).
//!
//! The integer 1..=4 attached to CIF/QCIF is the **minimum picture
//! interval (MPI)** in §6.2.1 terms: `CIF=2` advertises a CIF picture at
//! ≤ 15 fps (`29.97 / 2`).
//!
//! ## The fixed `a=rtpmap` line (§6.2)
//!
//! For `video/H261` the encoding name in `a=rtpmap` MUST be `H261` and the
//! clock rate MUST be `90000`; the media name in the `m=` line MUST be
//! `video`. [`ENCODING_NAME`] and [`CLOCK_RATE`] pin these constants;
//! [`format_rtpmap`] emits the attribute line for a chosen payload type.
//!
//! ## The `a=fmtp` line (§6.2)
//!
//! The optional `CIF`, `QCIF`, and `D` parameters, if any, are carried as a
//! semicolon-separated list on the `a=fmtp` line (§6.2 / §6.1.1).
//! [`H261FmtpParams`] is the typed view; [`H261FmtpParams::format_value`]
//! emits the bare parameter string (`CIF=2;QCIF=1;D=1`),
//! [`format_fmtp`] wraps it in a full `a=fmtp:<pt> …` line, and
//! [`H261FmtpParams::parse_value`] reverses the bare string.
//!
//! ## Offer/answer preference order (§6.2.1)
//!
//! "Parameters offered first are the most preferred picture mode to be
//! received." The parser preserves the order CIF / QCIF as written so a
//! caller implementing §6.2.1 can read which size the peer prefers; the
//! formatter emits whichever sizes are present, CIF before QCIF, matching
//! the spec's worked example `a=fmtp:31 CIF=2;QCIF=1;D=1`. An endpoint
//! "SHALL specify at least one supported picture size" (§6.2.1); callers
//! that need that invariant enforced use [`H261FmtpParams::validate`].
//!
//! ## RFC 2032 fallback (§6.2.1)
//!
//! If a peer specifies no picture-size parameter "it is safe to assume that
//! it is an implementation that follows RFC 2032" and "it is RECOMMENDED to
//! assume that such a receiver is able to support reception of QCIF
//! resolution with MPI=1". [`H261FmtpParams::rfc2032_fallback`] returns
//! exactly that assumption.

use crate::picture::SourceFormat;

/// Encoding name carried in the `a=rtpmap` line for `video/H261` (§6.2).
///
/// RFC 4587 §6.2: "The encoding name in the `a=rtpmap` line of SDP MUST be
/// H261 (the MIME subtype)."
pub const ENCODING_NAME: &str = "H261";

/// RTP clock rate for H.261, in Hz (§6.2).
///
/// RFC 4587 §6.2: "The clock rate in the `a=rtpmap` line MUST be 90000."
pub const CLOCK_RATE: u32 = 90_000;

/// Media name for the SDP `m=` line carrying H.261 (§6.2).
///
/// RFC 4587 §6.2: "The media name in the `m=` line of SDP MUST be video."
pub const MEDIA_NAME: &str = "video";

/// The full-rate H.261 frame rate (frames per second). §6.1.1 defines the
/// CIF/QCIF maximum rate as `29.97 / value`, i.e. this constant divided by
/// the MPI. Stored as a (numerator, denominator) rational so callers can
/// compute the bound without floating-point round-off (`2997 / 100` is the
/// exact NTSC field-rate value the spec writes as `29.97`).
pub const FULL_RATE_NUM: u32 = 2997;
/// Denominator of [`FULL_RATE_NUM`] (`29.97 = 2997 / 100`).
pub const FULL_RATE_DEN: u32 = 100;

/// Errors from parsing or validating H.261 SDP `fmtp` parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SdpError {
    /// A `CIF` or `QCIF` MPI value was outside the §6.1.1 range 1..=4.
    MpiOutOfRange {
        /// The parameter name (`"CIF"` or `"QCIF"`).
        param: &'static str,
        /// The offending value.
        value: u32,
    },
    /// The `D` (Annex D) parameter had a value other than `0` or `1`
    /// (§6.1.1: "the parameter value SHALL be `1`" / `0`).
    BadAnnexD {
        /// The raw value text as written.
        value: String,
    },
    /// A parameter value failed to parse as the expected integer.
    NotAnInteger {
        /// The parameter name as written.
        param: String,
        /// The raw value text.
        value: String,
    },
    /// A `key=value` token was missing its `=`.
    MalformedToken {
        /// The raw token.
        token: String,
    },
    /// The same picture-size parameter (`CIF` or `QCIF`) appeared twice.
    DuplicateParam {
        /// The parameter name.
        param: String,
    },
    /// [`H261FmtpParams::validate`] found neither `CIF` nor `QCIF` present.
    ///
    /// §6.2.1: "Implementations following this specification SHALL specify
    /// at least one supported picture size."
    NoPictureSize,
}

impl core::fmt::Display for SdpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SdpError::MpiOutOfRange { param, value } => {
                write!(f, "h261 sdp: {param} MPI {value} outside 1..=4")
            }
            SdpError::BadAnnexD { value } => {
                write!(f, "h261 sdp: D (Annex D) value {value:?} is not 0 or 1")
            }
            SdpError::NotAnInteger { param, value } => {
                write!(f, "h261 sdp: {param} value {value:?} is not an integer")
            }
            SdpError::MalformedToken { token } => {
                write!(f, "h261 sdp: fmtp token {token:?} is missing '='")
            }
            SdpError::DuplicateParam { param } => {
                write!(f, "h261 sdp: parameter {param} appears more than once")
            }
            SdpError::NoPictureSize => {
                write!(f, "h261 sdp: no CIF or QCIF picture size specified")
            }
        }
    }
}

impl std::error::Error for SdpError {}

/// Typed view of the H.261 `a=fmtp` optional parameters (RFC 4587 §6.1.1).
///
/// All three fields are optional, matching the wire format: a missing CIF /
/// QCIF parameter means that picture size is not advertised, and a missing
/// `D` means Annex D support is unstated (treated as unsupported per
/// §6.1.1's "SHOULD NOT be used … if not supported").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct H261FmtpParams {
    /// CIF maximum-rate MPI (1..=4), or `None` if CIF is not advertised.
    pub cif: Option<u8>,
    /// QCIF maximum-rate MPI (1..=4), or `None` if QCIF is not advertised.
    pub qcif: Option<u8>,
    /// Annex D still-image support. `Some(true)` ⇒ `D=1`, `Some(false)` ⇒
    /// `D=0`, `None` ⇒ the parameter is omitted.
    pub d: Option<bool>,
}

impl H261FmtpParams {
    /// The §6.2.1 RFC-2032 fallback: a peer that sends no picture-size
    /// parameter is assumed to support QCIF at MPI=1.
    ///
    /// RFC 4587 §6.2.1: "If the receiver does not specify the picture
    /// size/MPI parameter, then it is safe to assume that it is an
    /// implementation that follows RFC 2032. In that case, it is
    /// RECOMMENDED to assume that such a receiver is able to support
    /// reception of QCIF resolution with MPI=1."
    pub fn rfc2032_fallback() -> Self {
        H261FmtpParams {
            cif: None,
            qcif: Some(1),
            d: None,
        }
    }

    /// Whether a given source format is advertised by these parameters.
    pub fn supports(&self, fmt: SourceFormat) -> bool {
        match fmt {
            SourceFormat::Cif => self.cif.is_some(),
            SourceFormat::Qcif => self.qcif.is_some(),
        }
    }

    /// The MPI advertised for a source format, if any.
    pub fn mpi(&self, fmt: SourceFormat) -> Option<u8> {
        match fmt {
            SourceFormat::Cif => self.cif,
            SourceFormat::Qcif => self.qcif,
        }
    }

    /// Maximum frame rate for a source format as an exact `(num, den)`
    /// rational, per §6.1.1 (`29.97 / MPI`). Returns `None` if that format
    /// is not advertised.
    ///
    /// For example `mpi(Cif) == Some(2)` yields `Some((2997, 200))`
    /// ≈ 14.985 fps, the §6.2.1 "≤ 15 fps" bound for `CIF=2`.
    pub fn max_frame_rate(&self, fmt: SourceFormat) -> Option<(u32, u32)> {
        self.mpi(fmt)
            .map(|mpi| (FULL_RATE_NUM, FULL_RATE_DEN * u32::from(mpi)))
    }

    /// Enforce §6.2.1's "SHALL specify at least one supported picture size"
    /// invariant. Returns `Err(SdpError::NoPictureSize)` if neither CIF nor
    /// QCIF is present. (Field-range validity is guaranteed by construction
    /// — [`parse_value`](Self::parse_value) only stores in-range MPIs.)
    pub fn validate(&self) -> Result<(), SdpError> {
        if self.cif.is_none() && self.qcif.is_none() {
            return Err(SdpError::NoPictureSize);
        }
        Ok(())
    }

    /// Preferred picture mode advertised by these parameters per §6.2.1
    /// ("Parameters offered first are the most preferred picture mode to be
    /// received").
    ///
    /// The struct does not record the literal token order observed during
    /// parsing — it stores `cif` and `qcif` as independent
    /// `Option<u8>` fields, and [`format_value`](Self::format_value) emits
    /// CIF before QCIF to match the §6.2.1 worked example
    /// (`a=fmtp:31 CIF=2;QCIF=1;D=1`). To preserve that round trip,
    /// [`preferred_picture_size`](Self::preferred_picture_size) returns
    /// CIF when CIF is advertised (preferring it over QCIF), QCIF when
    /// only QCIF is advertised, and `None` when neither is.
    ///
    /// Callers that need to track literal offer-side preference order for a
    /// peer that emits `QCIF=…;CIF=…` (the spec uses CIF-first but the
    /// `parse_value` token order is otherwise unconstrained) should record
    /// that order separately from the parsed parameters.
    pub fn preferred_picture_size(&self) -> Option<SourceFormat> {
        if self.cif.is_some() {
            Some(SourceFormat::Cif)
        } else if self.qcif.is_some() {
            Some(SourceFormat::Qcif)
        } else {
            None
        }
    }

    /// Emit the bare semicolon-separated parameter string for the `a=fmtp`
    /// line, e.g. `"CIF=2;QCIF=1;D=1"` (§6.1.1 / §6.2).
    ///
    /// CIF is emitted before QCIF (the §6.2.1 worked-example order), then
    /// `D` last. An all-`None` params object yields the empty string. The
    /// `D=0` case is emitted explicitly when `d == Some(false)`; callers
    /// who want the §6.1.1 "SHOULD NOT be used" behaviour for unsupported
    /// Annex D set `d: None` instead.
    pub fn format_value(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(cif) = self.cif {
            parts.push(format!("CIF={cif}"));
        }
        if let Some(qcif) = self.qcif {
            parts.push(format!("QCIF={qcif}"));
        }
        if let Some(d) = self.d {
            parts.push(format!("D={}", u8::from(d)));
        }
        parts.join(";")
    }

    /// The literal §6.2.1 "Parameters offered first are the most preferred
    /// picture mode to be received" order observed in a raw `a=fmtp` value.
    ///
    /// [`parse_value`](Self::parse_value) discards token order — it stores
    /// `cif` and `qcif` as independent `Option<u8>` fields — and
    /// [`preferred_picture_size`](Self::preferred_picture_size) uses the
    /// structural CIF-over-QCIF rule of [`format_value`](Self::format_value).
    /// Neither captures a peer that advertises `QCIF=…;CIF=…` (QCIF first ⇒
    /// QCIF is the preferred mode). This helper walks the same token list
    /// `parse_value` does and returns each picture-size advertisement in the
    /// order seen, with no validation: it skips unknown / malformed / Annex-D
    /// tokens, ignores MPI range, and deduplicates so a repeated `CIF` or
    /// `QCIF` token is reported only the first time it appears.
    ///
    /// Returns an empty slice when neither `CIF` nor `QCIF` is present
    /// (the §6.2.1 RFC 2032 fallback case). Returns one entry when only one
    /// picture size is advertised; up to two entries otherwise, ordered as
    /// written. The first entry, if any, is the peer's preferred mode per
    /// §6.2.1's "offered first" rule.
    ///
    /// ```
    /// use oxideav_h261::picture::SourceFormat;
    /// use oxideav_h261::sdp::H261FmtpParams;
    /// // Spec worked example: CIF preferred over QCIF.
    /// assert_eq!(
    ///     H261FmtpParams::parse_preference_order("CIF=2;QCIF=1;D=1"),
    ///     vec![SourceFormat::Cif, SourceFormat::Qcif],
    /// );
    /// // Peer prefers QCIF: `QCIF=1;CIF=2` ⇒ QCIF leads the preference list.
    /// assert_eq!(
    ///     H261FmtpParams::parse_preference_order("QCIF=1;CIF=2"),
    ///     vec![SourceFormat::Qcif, SourceFormat::Cif],
    /// );
    /// // RFC 2032 fallback case ⇒ empty list (callers apply the QCIF=1
    /// // default themselves).
    /// assert!(H261FmtpParams::parse_preference_order("D=1").is_empty());
    /// ```
    pub fn parse_preference_order(s: &str) -> Vec<SourceFormat> {
        let mut order: Vec<SourceFormat> = Vec::new();
        let mut seen_cif = false;
        let mut seen_qcif = false;
        for raw in s.split(';') {
            let token = raw.trim();
            if token.is_empty() {
                continue;
            }
            let Some((key, _value)) = token.split_once('=') else {
                continue;
            };
            // SDP attribute token names are case-insensitive; §6.1.1 spells
            // CIF / QCIF in uppercase.
            match key.trim().to_ascii_uppercase().as_str() {
                "CIF" if !seen_cif => {
                    seen_cif = true;
                    order.push(SourceFormat::Cif);
                }
                "QCIF" if !seen_qcif => {
                    seen_qcif = true;
                    order.push(SourceFormat::Qcif);
                }
                _ => {}
            }
        }
        order
    }

    /// Parse the bare `a=fmtp` parameter string back into typed params.
    ///
    /// Accepts a semicolon-separated list of `key=value` tokens
    /// (§6.1.1 / §6.2); surrounding whitespace around each token and around
    /// the `=` is tolerated. Unknown parameter names are ignored
    /// forward-compatibly (a future RFC may add parameters this version
    /// doesn't model). Parameter names are matched case-insensitively
    /// because §6.1.1 names them in uppercase but SDP attribute matching is
    /// case-insensitive for token names.
    ///
    /// Validates that CIF/QCIF MPIs are integers in 1..=4 and that `D` is
    /// `0` or `1`; a duplicate CIF or QCIF is rejected. An empty / all-
    /// whitespace string yields `H261FmtpParams::default()` (all `None`).
    pub fn parse_value(s: &str) -> Result<Self, SdpError> {
        let mut out = H261FmtpParams::default();
        let mut seen_cif = false;
        let mut seen_qcif = false;

        for raw in s.split(';') {
            let token = raw.trim();
            if token.is_empty() {
                continue;
            }
            let (key, value) = token
                .split_once('=')
                .ok_or_else(|| SdpError::MalformedToken {
                    token: token.to_string(),
                })?;
            let key = key.trim();
            let value = value.trim();

            // SDP attribute token names are case-insensitive; §6.1.1 spells
            // CIF/QCIF/D in uppercase.
            let key_upper = key.to_ascii_uppercase();
            match key_upper.as_str() {
                "CIF" => {
                    if seen_cif {
                        return Err(SdpError::DuplicateParam {
                            param: "CIF".to_string(),
                        });
                    }
                    seen_cif = true;
                    out.cif = Some(parse_mpi("CIF", value)?);
                }
                "QCIF" => {
                    if seen_qcif {
                        return Err(SdpError::DuplicateParam {
                            param: "QCIF".to_string(),
                        });
                    }
                    seen_qcif = true;
                    out.qcif = Some(parse_mpi("QCIF", value)?);
                }
                "D" => {
                    out.d = Some(parse_annex_d(value)?);
                }
                // Forward-compatible: ignore parameters this version does
                // not model rather than rejecting the whole line.
                _ => {}
            }
        }

        Ok(out)
    }
}

/// Compute the SDP offer/answer **answer** parameters from a received
/// offer and our local capability, per RFC 4587 §6.2.1.
///
/// The two sides have to agree on a common picture size and a common
/// frame-rate bound. Each `H261FmtpParams` describes one peer's view
/// using the §6.1.1 `CIF` / `QCIF` MPI parameters:
///
/// * **Picture size intersection.** Only picture sizes both peers
///   advertise survive into the answer. A CIF advertisement from one
///   side and a QCIF-only advertisement from the other yields no
///   common size — surface as [`SdpError::NoPictureSize`] so the caller
///   can reject the offer cleanly (§6.2.1: "SHALL specify at least one
///   supported picture size").
/// * **Frame-rate bound (MPI per size).** §6.1.1's `value` parameter is
///   the *minimum picture interval*, so `29.97 / MPI` is the **upper
///   bound** on frame rate the receiver advertises. The sender must
///   satisfy **both** peers' upper bounds, i.e. the negotiated rate
///   must not exceed `29.97 / max(offer_MPI, our_MPI)`. The answer
///   therefore carries `MPI = max(offer.MPI, our.MPI)` for each shared
///   size.
/// * **Annex D (`D`).** §6.2.1: "This option MUST NOT appear unless
///   the sender of this SDP message is able to decode this option."
///   Annex D is a still-image decoder capability; the answer's `D=1`
///   requires both `offer.d == Some(true)` AND
///   `our_capability.d == Some(true)`. Otherwise the answer omits `D`
///   (matching §6.1.1's "SHOULD NOT be used … if not supported").
/// * **RFC 2032 fallback.** If `offer` advertises no picture size at
///   all, §6.2.1 says it "is safe to assume … reception of QCIF
///   resolution with MPI=1". The negotiation applies that fallback
///   automatically (equivalent to [`H261FmtpParams::rfc2032_fallback`])
///   before intersecting with our capability. The same fallback is
///   **not** applied to `our_capability` — that side is local and
///   should be supplied explicitly.
///
/// The answer is constructed so [`format_value`](H261FmtpParams::format_value)
/// emits the §6.2.1 CIF-before-QCIF order, matching the spec example
/// `a=fmtp:31 CIF=2;QCIF=1;D=1`.
///
/// ```
/// use oxideav_h261::sdp::{H261FmtpParams, negotiate_answer};
/// // Peer A offers CIF=2 (≤ 15 fps), QCIF=1 (≤ 29.97 fps), Annex D.
/// let offer = H261FmtpParams { cif: Some(2), qcif: Some(1), d: Some(true) };
/// // We can decode CIF at ≤ 14.985 fps (MPI=2) and QCIF at ≤ 14.985 fps
/// // (MPI=2). We don't support Annex D.
/// let ours = H261FmtpParams { cif: Some(2), qcif: Some(2), d: None };
/// let answer = negotiate_answer(&offer, &ours).unwrap();
/// // Both peers can do CIF at MPI=2 → answer CIF=2.
/// assert_eq!(answer.cif, Some(2));
/// // Offer caps QCIF at MPI=1 (≤ 29.97 fps) and we cap it at MPI=2
/// // (≤ 14.985 fps); the answer must satisfy both ⇒ MPI=2.
/// assert_eq!(answer.qcif, Some(2));
/// // Our side cannot decode Annex D ⇒ D omitted from the answer.
/// assert_eq!(answer.d, None);
/// ```
pub fn negotiate_answer(
    offer: &H261FmtpParams,
    our_capability: &H261FmtpParams,
) -> Result<H261FmtpParams, SdpError> {
    // §6.2.1 RFC 2032 fallback: an offer with no picture-size parameters
    // is treated as QCIF=1.
    let effective_offer = if offer.cif.is_none() && offer.qcif.is_none() {
        H261FmtpParams::rfc2032_fallback()
    } else {
        *offer
    };

    let cif = match (effective_offer.cif, our_capability.cif) {
        (Some(a), Some(b)) => Some(a.max(b)),
        _ => None,
    };
    let qcif = match (effective_offer.qcif, our_capability.qcif) {
        (Some(a), Some(b)) => Some(a.max(b)),
        _ => None,
    };

    if cif.is_none() && qcif.is_none() {
        return Err(SdpError::NoPictureSize);
    }

    let d = match (effective_offer.d, our_capability.d) {
        (Some(true), Some(true)) => Some(true),
        _ => None,
    };

    Ok(H261FmtpParams { cif, qcif, d })
}

/// Parse a CIF/QCIF MPI value, enforcing the §6.1.1 range 1..=4.
fn parse_mpi(param: &'static str, value: &str) -> Result<u8, SdpError> {
    let n: u32 = value.parse().map_err(|_| SdpError::NotAnInteger {
        param: param.to_string(),
        value: value.to_string(),
    })?;
    if !(1..=4).contains(&n) {
        return Err(SdpError::MpiOutOfRange { param, value: n });
    }
    Ok(n as u8)
}

/// Parse the §6.1.1 `D` (Annex D) value, which SHALL be `0` or `1`.
fn parse_annex_d(value: &str) -> Result<bool, SdpError> {
    match value {
        "1" => Ok(true),
        "0" => Ok(false),
        other => Err(SdpError::BadAnnexD {
            value: other.to_string(),
        }),
    }
}

/// Format the full SDP `a=rtpmap` attribute line for an H.261 payload type
/// (§6.2). The leading `a=` is included so the result drops straight into a
/// session description.
///
/// Example: `format_rtpmap(31)` → `"a=rtpmap:31 H261/90000"`.
pub fn format_rtpmap(payload_type: u8) -> String {
    format!("a=rtpmap:{payload_type} {ENCODING_NAME}/{CLOCK_RATE}")
}

/// Format the full SDP `a=fmtp` attribute line for the given parameters
/// (§6.2). Returns `None` if no optional parameters are present — §6.2
/// includes the `a=fmtp` line only "if any" parameters exist, so an empty
/// parameter set produces no line.
///
/// Example: `format_fmtp(31, &params)` → `"a=fmtp:31 CIF=2;QCIF=1;D=1"`.
pub fn format_fmtp(payload_type: u8, params: &H261FmtpParams) -> Option<String> {
    let value = params.format_value();
    if value.is_empty() {
        None
    } else {
        Some(format!("a=fmtp:{payload_type} {value}"))
    }
}

/// Parsed view of an `a=rtpmap` line, after `parse_rtpmap` validates it as
/// H.261.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RtpMap {
    /// The dynamic RTP payload type the map binds.
    pub payload_type: u8,
    /// The clock rate parsed from the line (MUST be [`CLOCK_RATE`] for a
    /// conformant H.261 map).
    pub clock_rate: u32,
}

impl RtpMap {
    /// Whether the parsed map is RFC 4587 §6.2 compliant.
    ///
    /// §6.2 mandates `clock_rate == 90000` (the encoding-name and
    /// `m=video` invariants are already enforced by [`parse_rtpmap`]). A
    /// receiver that requires strict conformance should reject any map
    /// for which this returns `false`. The lenient [`parse_rtpmap`]
    /// parser preserves the wire `clock_rate` verbatim so a misbehaving
    /// peer can be diagnosed; this accessor is the single-call
    /// "did the peer follow §6.2?" check.
    pub fn is_rfc4587_compliant(self) -> bool {
        self.clock_rate == CLOCK_RATE
    }
}

/// Parse an SDP `a=rtpmap` attribute line and confirm it describes H.261
/// (§6.2). Accepts the line with or without the leading `a=`. Returns
/// `None` if the line is not a well-formed H.261 rtpmap — wrong attribute,
/// non-`H261` encoding name, or a non-numeric payload type. The encoding
/// name match is case-insensitive (SDP attribute tokens are
/// case-insensitive); the clock rate is reported as-parsed so a caller can
/// reject a non-90000 value if it wants strict conformance. Callers that
/// always want strict §6.2 conformance can use [`parse_rtpmap_strict`] or
/// [`RtpMap::is_rfc4587_compliant`] instead.
pub fn parse_rtpmap(line: &str) -> Option<RtpMap> {
    let line = line.trim();
    let body = line.strip_prefix("a=").unwrap_or(line);
    let rest = body.strip_prefix("rtpmap:")?;
    // Format: "<pt> <encoding>/<clock>[/<params>]"
    let (pt_str, enc_clock) = rest.split_once(char::is_whitespace)?;
    let payload_type: u8 = pt_str.trim().parse().ok()?;
    let enc_clock = enc_clock.trim();
    let mut fields = enc_clock.split('/');
    let encoding = fields.next()?.trim();
    if !encoding.eq_ignore_ascii_case(ENCODING_NAME) {
        return None;
    }
    let clock_rate: u32 = fields.next()?.trim().parse().ok()?;
    Some(RtpMap {
        payload_type,
        clock_rate,
    })
}

/// Strict-conformance counterpart of [`parse_rtpmap`]: same parser, but
/// also enforces the RFC 4587 §6.2 `clock_rate == 90000` MUST. Returns
/// `None` if the line is not a well-formed H.261 rtpmap or carries any
/// other clock rate. Use this on a receiver that wants to drop a peer
/// whose advertised clock rate violates §6.2 without inspecting the
/// parsed value separately.
pub fn parse_rtpmap_strict(line: &str) -> Option<RtpMap> {
    let m = parse_rtpmap(line)?;
    if m.is_rfc4587_compliant() {
        Some(m)
    } else {
        None
    }
}

/// Parse an SDP `a=fmtp` attribute line whose payload type matches
/// `payload_type` (§6.2). Accepts the line with or without the leading
/// `a=`. Returns `None` if the line is not an `fmtp` attribute or its
/// payload type does not match; returns `Err` if the parameter list is
/// malformed (delegated to [`H261FmtpParams::parse_value`]).
pub fn parse_fmtp(line: &str, payload_type: u8) -> Option<Result<H261FmtpParams, SdpError>> {
    let line = line.trim();
    let body = line.strip_prefix("a=").unwrap_or(line);
    let rest = body.strip_prefix("fmtp:")?;
    let (pt_str, value) = rest.split_once(char::is_whitespace)?;
    let pt: u8 = pt_str.trim().parse().ok()?;
    if pt != payload_type {
        return None;
    }
    Some(H261FmtpParams::parse_value(value.trim()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtpmap_constants_match_spec() {
        // §6.2: encoding name "H261", clock rate 90000, media name "video".
        assert_eq!(ENCODING_NAME, "H261");
        assert_eq!(CLOCK_RATE, 90_000);
        assert_eq!(MEDIA_NAME, "video");
    }

    #[test]
    fn format_rtpmap_matches_spec_example() {
        // §6.2.1 worked example: "a=rtpmap:31 H261/90000".
        assert_eq!(format_rtpmap(31), "a=rtpmap:31 H261/90000");
        assert_eq!(format_rtpmap(96), "a=rtpmap:96 H261/90000");
    }

    #[test]
    fn parse_rtpmap_accepts_spec_example() {
        let m = parse_rtpmap("a=rtpmap:31 H261/90000").unwrap();
        assert_eq!(m.payload_type, 31);
        assert_eq!(m.clock_rate, 90_000);
    }

    #[test]
    fn parse_rtpmap_without_a_prefix() {
        let m = parse_rtpmap("rtpmap:96 H261/90000").unwrap();
        assert_eq!(m.payload_type, 96);
        assert_eq!(m.clock_rate, 90_000);
    }

    #[test]
    fn parse_rtpmap_is_case_insensitive_on_encoding() {
        // SDP attribute tokens are case-insensitive.
        assert!(parse_rtpmap("a=rtpmap:31 h261/90000").is_some());
        assert!(parse_rtpmap("a=rtpmap:31 H261/90000").is_some());
    }

    #[test]
    fn parse_rtpmap_rejects_other_codecs() {
        assert!(parse_rtpmap("a=rtpmap:34 H263/90000").is_none());
        assert!(parse_rtpmap("a=rtpmap:0 PCMU/8000").is_none());
    }

    #[test]
    fn parse_rtpmap_rejects_non_rtpmap_line() {
        assert!(parse_rtpmap("a=fmtp:31 CIF=2").is_none());
        assert!(parse_rtpmap("m=video 49170 RTP/AVP 31").is_none());
    }

    #[test]
    fn parse_rtpmap_tolerates_trailing_channel_field() {
        // "<encoding>/<clock>/<params>" — the optional third field is
        // ignored; H.261 doesn't use it but a line carrying it is still a
        // valid H.261 rtpmap.
        let m = parse_rtpmap("a=rtpmap:31 H261/90000/1").unwrap();
        assert_eq!(m.clock_rate, 90_000);
    }

    #[test]
    fn rtpmap_is_rfc4587_compliant_only_at_90000() {
        // §6.2: "The clock rate in the `a=rtpmap` line MUST be 90000."
        let ok = parse_rtpmap("a=rtpmap:31 H261/90000").unwrap();
        assert!(ok.is_rfc4587_compliant());
        // Lenient parser preserves the wire value, so callers can diagnose
        // a non-conformant peer without losing the parsed payload type.
        let too_low = parse_rtpmap("a=rtpmap:31 H261/8000").unwrap();
        assert_eq!(too_low.clock_rate, 8_000);
        assert!(!too_low.is_rfc4587_compliant());
        let too_high = parse_rtpmap("a=rtpmap:96 H261/180000").unwrap();
        assert_eq!(too_high.clock_rate, 180_000);
        assert!(!too_high.is_rfc4587_compliant());
        // RFC 2032's pre-§6.2 line happens to match the §6.2 MUST since
        // the clock rate was already 90000 in the predecessor doc.
        let legacy = parse_rtpmap("rtpmap:31 H261/90000").unwrap();
        assert!(legacy.is_rfc4587_compliant());
    }

    #[test]
    fn parse_rtpmap_strict_drops_non_compliant_clock_rate() {
        // §6.2 MUST → strict parser returns None for any non-90000 rate.
        assert_eq!(
            parse_rtpmap_strict("a=rtpmap:31 H261/90000"),
            Some(RtpMap {
                payload_type: 31,
                clock_rate: 90_000,
            })
        );
        assert_eq!(parse_rtpmap_strict("a=rtpmap:31 H261/8000"), None);
        assert_eq!(parse_rtpmap_strict("a=rtpmap:96 H261/45000"), None);
        // Strict parser still rejects non-H.261 encoding names and
        // malformed lines, inherited from `parse_rtpmap`.
        assert_eq!(parse_rtpmap_strict("a=rtpmap:34 H263/90000"), None);
        assert_eq!(parse_rtpmap_strict("a=fmtp:31 CIF=2"), None);
        // Tolerates the optional third "<params>" field as long as the
        // clock rate is the §6.2-mandated 90000.
        let m = parse_rtpmap_strict("a=rtpmap:31 H261/90000/1").unwrap();
        assert_eq!(m.payload_type, 31);
        assert!(m.is_rfc4587_compliant());
    }

    #[test]
    fn format_value_matches_spec_example() {
        // §6.2.1: "a=fmtp:31 CIF=2;QCIF=1;D=1".
        let p = H261FmtpParams {
            cif: Some(2),
            qcif: Some(1),
            d: Some(true),
        };
        assert_eq!(p.format_value(), "CIF=2;QCIF=1;D=1");
    }

    #[test]
    fn format_fmtp_matches_spec_example() {
        let p = H261FmtpParams {
            cif: Some(2),
            qcif: Some(1),
            d: Some(true),
        };
        assert_eq!(
            format_fmtp(31, &p).as_deref(),
            Some("a=fmtp:31 CIF=2;QCIF=1;D=1")
        );
    }

    #[test]
    fn format_fmtp_empty_params_yields_no_line() {
        // §6.2 includes the fmtp line only "if any" parameters exist.
        let p = H261FmtpParams::default();
        assert_eq!(format_fmtp(31, &p), None);
        assert_eq!(p.format_value(), "");
    }

    #[test]
    fn format_value_cif_before_qcif() {
        // The §6.2.1 example orders CIF then QCIF.
        let p = H261FmtpParams {
            cif: Some(4),
            qcif: Some(2),
            d: None,
        };
        assert_eq!(p.format_value(), "CIF=4;QCIF=2");
    }

    #[test]
    fn format_value_d_zero_is_explicit() {
        let p = H261FmtpParams {
            cif: None,
            qcif: Some(1),
            d: Some(false),
        };
        assert_eq!(p.format_value(), "QCIF=1;D=0");
    }

    #[test]
    fn parse_value_round_trips_spec_example() {
        let p = H261FmtpParams::parse_value("CIF=2;QCIF=1;D=1").unwrap();
        assert_eq!(p.cif, Some(2));
        assert_eq!(p.qcif, Some(1));
        assert_eq!(p.d, Some(true));
        assert_eq!(p.format_value(), "CIF=2;QCIF=1;D=1");
    }

    #[test]
    fn parse_value_tolerates_whitespace() {
        let p = H261FmtpParams::parse_value(" CIF = 2 ; QCIF=1 ; D = 1 ").unwrap();
        assert_eq!(p.cif, Some(2));
        assert_eq!(p.qcif, Some(1));
        assert_eq!(p.d, Some(true));
    }

    #[test]
    fn parse_value_case_insensitive_names() {
        let p = H261FmtpParams::parse_value("cif=3;qcif=4;d=1").unwrap();
        assert_eq!(p.cif, Some(3));
        assert_eq!(p.qcif, Some(4));
        assert_eq!(p.d, Some(true));
    }

    #[test]
    fn parse_value_empty_yields_default() {
        assert_eq!(
            H261FmtpParams::parse_value("").unwrap(),
            H261FmtpParams::default()
        );
        assert_eq!(
            H261FmtpParams::parse_value("   ").unwrap(),
            H261FmtpParams::default()
        );
        // Trailing/leading semicolons produce empty tokens that are skipped.
        let p = H261FmtpParams::parse_value(";CIF=1;").unwrap();
        assert_eq!(p.cif, Some(1));
    }

    #[test]
    fn parse_value_ignores_unknown_params() {
        // Forward-compatible: a future RFC parameter is skipped, not fatal.
        let p = H261FmtpParams::parse_value("CIF=2;MAXBR=256;QCIF=1").unwrap();
        assert_eq!(p.cif, Some(2));
        assert_eq!(p.qcif, Some(1));
    }

    #[test]
    fn parse_value_rejects_mpi_out_of_range() {
        // §6.1.1: permissible values are 1..=4.
        for bad in [0u32, 5, 100] {
            let err = H261FmtpParams::parse_value(&format!("CIF={bad}")).unwrap_err();
            assert_eq!(
                err,
                SdpError::MpiOutOfRange {
                    param: "CIF",
                    value: bad
                }
            );
        }
        let err = H261FmtpParams::parse_value("QCIF=9").unwrap_err();
        assert_eq!(
            err,
            SdpError::MpiOutOfRange {
                param: "QCIF",
                value: 9
            }
        );
    }

    #[test]
    fn parse_value_accepts_full_mpi_range() {
        for mpi in 1u8..=4 {
            let p = H261FmtpParams::parse_value(&format!("CIF={mpi}")).unwrap();
            assert_eq!(p.cif, Some(mpi));
        }
    }

    #[test]
    fn parse_value_rejects_non_integer_mpi() {
        let err = H261FmtpParams::parse_value("CIF=two").unwrap_err();
        assert_eq!(
            err,
            SdpError::NotAnInteger {
                param: "CIF".to_string(),
                value: "two".to_string()
            }
        );
    }

    #[test]
    fn parse_value_rejects_bad_annex_d() {
        // §6.1.1: D SHALL be "1" (or "0" when present).
        let err = H261FmtpParams::parse_value("QCIF=1;D=2").unwrap_err();
        assert_eq!(
            err,
            SdpError::BadAnnexD {
                value: "2".to_string()
            }
        );
        let err = H261FmtpParams::parse_value("QCIF=1;D=yes").unwrap_err();
        assert_eq!(
            err,
            SdpError::BadAnnexD {
                value: "yes".to_string()
            }
        );
    }

    #[test]
    fn parse_value_accepts_d_zero() {
        let p = H261FmtpParams::parse_value("QCIF=1;D=0").unwrap();
        assert_eq!(p.d, Some(false));
    }

    #[test]
    fn parse_value_rejects_malformed_token() {
        let err = H261FmtpParams::parse_value("CIF").unwrap_err();
        assert_eq!(
            err,
            SdpError::MalformedToken {
                token: "CIF".to_string()
            }
        );
    }

    #[test]
    fn parse_value_rejects_duplicate_param() {
        let err = H261FmtpParams::parse_value("CIF=2;CIF=3").unwrap_err();
        assert_eq!(
            err,
            SdpError::DuplicateParam {
                param: "CIF".to_string()
            }
        );
        let err = H261FmtpParams::parse_value("QCIF=1;qcif=2").unwrap_err();
        assert_eq!(
            err,
            SdpError::DuplicateParam {
                param: "QCIF".to_string()
            }
        );
    }

    #[test]
    fn validate_requires_a_picture_size() {
        // §6.2.1: SHALL specify at least one supported picture size.
        let none = H261FmtpParams {
            cif: None,
            qcif: None,
            d: Some(true),
        };
        assert_eq!(none.validate(), Err(SdpError::NoPictureSize));
        let cif_only = H261FmtpParams {
            cif: Some(1),
            ..Default::default()
        };
        assert_eq!(cif_only.validate(), Ok(()));
        let qcif_only = H261FmtpParams {
            qcif: Some(1),
            ..Default::default()
        };
        assert_eq!(qcif_only.validate(), Ok(()));
    }

    #[test]
    fn supports_and_mpi_reflect_fields() {
        let p = H261FmtpParams {
            cif: Some(2),
            qcif: None,
            d: None,
        };
        assert!(p.supports(SourceFormat::Cif));
        assert!(!p.supports(SourceFormat::Qcif));
        assert_eq!(p.mpi(SourceFormat::Cif), Some(2));
        assert_eq!(p.mpi(SourceFormat::Qcif), None);
    }

    #[test]
    fn max_frame_rate_is_29_97_over_mpi() {
        // §6.1.1: maximum rate is 29.97 / value. As an exact rational that
        // is 2997 / (100 * MPI).
        let p = H261FmtpParams {
            cif: Some(2),
            qcif: Some(1),
            d: None,
        };
        // CIF=2 ⇒ 2997/200 = 14.985 fps (the §6.2.1 "< 15 fps" bound).
        assert_eq!(p.max_frame_rate(SourceFormat::Cif), Some((2997, 200)));
        // QCIF=1 ⇒ 2997/100 = 29.97 fps (full rate).
        assert_eq!(p.max_frame_rate(SourceFormat::Qcif), Some((2997, 100)));
        // Unadvertised size ⇒ None.
        let q = H261FmtpParams {
            cif: None,
            qcif: Some(4),
            d: None,
        };
        assert_eq!(q.max_frame_rate(SourceFormat::Cif), None);
        // QCIF=4 ⇒ 2997/400 ≈ 7.49 fps.
        assert_eq!(q.max_frame_rate(SourceFormat::Qcif), Some((2997, 400)));
    }

    #[test]
    fn rfc2032_fallback_is_qcif_mpi_1() {
        // §6.2.1: absent picture-size params ⇒ assume QCIF MPI=1.
        let p = H261FmtpParams::rfc2032_fallback();
        assert_eq!(p.qcif, Some(1));
        assert_eq!(p.cif, None);
        assert_eq!(p.d, None);
        assert!(p.supports(SourceFormat::Qcif));
        assert!(!p.supports(SourceFormat::Cif));
    }

    #[test]
    fn parse_fmtp_line_round_trips_spec_example() {
        let parsed = parse_fmtp("a=fmtp:31 CIF=2;QCIF=1;D=1", 31)
            .unwrap()
            .unwrap();
        assert_eq!(parsed.cif, Some(2));
        assert_eq!(parsed.qcif, Some(1));
        assert_eq!(parsed.d, Some(true));
        // Re-emit identical to the spec example.
        assert_eq!(
            format_fmtp(31, &parsed).as_deref(),
            Some("a=fmtp:31 CIF=2;QCIF=1;D=1")
        );
    }

    #[test]
    fn parse_fmtp_without_a_prefix() {
        let parsed = parse_fmtp("fmtp:96 QCIF=1", 96).unwrap().unwrap();
        assert_eq!(parsed.qcif, Some(1));
    }

    #[test]
    fn parse_fmtp_rejects_payload_type_mismatch() {
        // Right line, wrong PT ⇒ None (not this PT's fmtp).
        assert!(parse_fmtp("a=fmtp:31 CIF=2", 96).is_none());
    }

    #[test]
    fn parse_fmtp_rejects_non_fmtp_line() {
        assert!(parse_fmtp("a=rtpmap:31 H261/90000", 31).is_none());
    }

    #[test]
    fn parse_fmtp_surfaces_value_errors() {
        // A matching-PT line with a bad value returns Some(Err(..)).
        let res = parse_fmtp("a=fmtp:31 CIF=9", 31).unwrap();
        assert_eq!(
            res,
            Err(SdpError::MpiOutOfRange {
                param: "CIF",
                value: 9
            })
        );
    }

    #[test]
    fn full_session_description_lines_round_trip() {
        // Build the two attribute lines for PT=31, then parse them back.
        let params = H261FmtpParams {
            cif: Some(2),
            qcif: Some(1),
            d: Some(true),
        };
        let rtpmap = format_rtpmap(31);
        let fmtp = format_fmtp(31, &params).unwrap();
        assert_eq!(rtpmap, "a=rtpmap:31 H261/90000");
        assert_eq!(fmtp, "a=fmtp:31 CIF=2;QCIF=1;D=1");

        let map = parse_rtpmap(&rtpmap).unwrap();
        assert_eq!(map.payload_type, 31);
        assert_eq!(map.clock_rate, CLOCK_RATE);
        let back = parse_fmtp(&fmtp, map.payload_type).unwrap().unwrap();
        assert_eq!(back, params);
    }

    #[test]
    fn preferred_picture_size_picks_cif_when_advertised() {
        let both = H261FmtpParams {
            cif: Some(2),
            qcif: Some(1),
            d: None,
        };
        assert_eq!(both.preferred_picture_size(), Some(SourceFormat::Cif));
        let qcif_only = H261FmtpParams {
            cif: None,
            qcif: Some(1),
            d: None,
        };
        assert_eq!(qcif_only.preferred_picture_size(), Some(SourceFormat::Qcif));
        let cif_only = H261FmtpParams {
            cif: Some(3),
            qcif: None,
            d: None,
        };
        assert_eq!(cif_only.preferred_picture_size(), Some(SourceFormat::Cif));
        let neither = H261FmtpParams::default();
        assert_eq!(neither.preferred_picture_size(), None);
    }

    #[test]
    fn negotiate_answer_intersects_picture_sizes() {
        // Offer: CIF=2, QCIF=1; ours: QCIF=1 only ⇒ answer: QCIF=1.
        let offer = H261FmtpParams {
            cif: Some(2),
            qcif: Some(1),
            d: None,
        };
        let ours = H261FmtpParams {
            cif: None,
            qcif: Some(1),
            d: None,
        };
        let ans = negotiate_answer(&offer, &ours).unwrap();
        assert_eq!(ans.cif, None);
        assert_eq!(ans.qcif, Some(1));
        assert_eq!(ans.d, None);
    }

    #[test]
    fn negotiate_answer_picks_max_mpi_per_size() {
        // §6.2.1: receiver advertises an upper bound on rate via MPI.
        // The sender must satisfy both upper bounds ⇒ answer MPI =
        // max(offer.MPI, our.MPI) = the more restrictive bound.
        let offer = H261FmtpParams {
            cif: Some(1),
            qcif: Some(1),
            d: None,
        };
        let ours = H261FmtpParams {
            cif: Some(3),
            qcif: Some(2),
            d: None,
        };
        let ans = negotiate_answer(&offer, &ours).unwrap();
        assert_eq!(ans.cif, Some(3));
        assert_eq!(ans.qcif, Some(2));
    }

    #[test]
    fn negotiate_answer_disjoint_sizes_errors() {
        // Offer: CIF-only; ours: QCIF-only ⇒ no common size.
        let offer = H261FmtpParams {
            cif: Some(2),
            qcif: None,
            d: None,
        };
        let ours = H261FmtpParams {
            cif: None,
            qcif: Some(1),
            d: None,
        };
        assert_eq!(
            negotiate_answer(&offer, &ours),
            Err(SdpError::NoPictureSize)
        );
    }

    #[test]
    fn negotiate_answer_annex_d_needs_both_sides() {
        // D=1 only when both sides signal D=1.
        let off_d = H261FmtpParams {
            cif: Some(2),
            qcif: Some(1),
            d: Some(true),
        };
        let our_d = H261FmtpParams {
            cif: Some(2),
            qcif: Some(1),
            d: Some(true),
        };
        let our_no_d = H261FmtpParams {
            cif: Some(2),
            qcif: Some(1),
            d: None,
        };
        let our_d_zero = H261FmtpParams {
            cif: Some(2),
            qcif: Some(1),
            d: Some(false),
        };
        assert_eq!(negotiate_answer(&off_d, &our_d).unwrap().d, Some(true));
        assert_eq!(negotiate_answer(&off_d, &our_no_d).unwrap().d, None);
        assert_eq!(negotiate_answer(&off_d, &our_d_zero).unwrap().d, None);
        // Offer omits D ⇒ answer omits D even when we support it.
        let off_no_d = H261FmtpParams {
            cif: Some(2),
            qcif: Some(1),
            d: None,
        };
        assert_eq!(negotiate_answer(&off_no_d, &our_d).unwrap().d, None);
    }

    #[test]
    fn negotiate_answer_rfc2032_fallback_for_offerless_picture_size() {
        // §6.2.1: an offer with no picture-size parameters is assumed
        // to support QCIF at MPI=1 (RFC 2032 fallback).
        let offer = H261FmtpParams::default(); // no CIF, no QCIF, no D.
        let ours = H261FmtpParams {
            cif: Some(2),
            qcif: Some(2),
            d: Some(true),
        };
        let ans = negotiate_answer(&offer, &ours).unwrap();
        // Offer is treated as QCIF=1 ⇒ intersect with our QCIF=2 ⇒
        // answer QCIF = max(1, 2) = 2.
        assert_eq!(ans.cif, None);
        assert_eq!(ans.qcif, Some(2));
        // Offer doesn't carry D ⇒ answer must omit it.
        assert_eq!(ans.d, None);
    }

    #[test]
    fn negotiate_answer_round_trips_through_format_value() {
        // The negotiated answer formats with CIF before QCIF — matching
        // the §6.2.1 worked example.
        let offer = H261FmtpParams {
            cif: Some(2),
            qcif: Some(1),
            d: Some(true),
        };
        let ours = H261FmtpParams {
            cif: Some(2),
            qcif: Some(1),
            d: Some(true),
        };
        let ans = negotiate_answer(&offer, &ours).unwrap();
        assert_eq!(ans.format_value(), "CIF=2;QCIF=1;D=1");
    }

    #[test]
    fn negotiate_answer_validate_passes() {
        // A successful negotiation always satisfies validate().
        let offer = H261FmtpParams {
            cif: Some(2),
            qcif: Some(1),
            d: None,
        };
        let ours = H261FmtpParams {
            cif: Some(2),
            qcif: None,
            d: None,
        };
        let ans = negotiate_answer(&offer, &ours).unwrap();
        assert!(ans.validate().is_ok());
    }

    #[test]
    fn negotiate_answer_full_offer_answer_max_frame_rate_satisfies_both_peers() {
        // The answer's max_frame_rate for each size must be no greater
        // than either peer's advertised max_frame_rate for that size.
        let offer = H261FmtpParams {
            cif: Some(1),
            qcif: Some(4),
            d: None,
        };
        let ours = H261FmtpParams {
            cif: Some(2),
            qcif: Some(1),
            d: None,
        };
        let ans = negotiate_answer(&offer, &ours).unwrap();
        // CIF: offer 29.97 fps, ours 14.985 fps ⇒ answer must be the lower
        // bound = 14.985 fps ⇒ MPI=2.
        assert_eq!(ans.cif, Some(2));
        // QCIF: offer 7.49 fps, ours 29.97 fps ⇒ answer must be the lower
        // bound = 7.49 fps ⇒ MPI=4.
        assert_eq!(ans.qcif, Some(4));
        // Sanity: the answer's max rate per size is the min of the two
        // peers' max rates.
        // For CIF: min(29.97/1, 29.97/2) = 29.97/2 ⇒ MPI=2.
        // For QCIF: min(29.97/4, 29.97/1) = 29.97/4 ⇒ MPI=4.
        let (num_cif, den_cif) = ans.max_frame_rate(SourceFormat::Cif).unwrap();
        assert_eq!((num_cif, den_cif), (2997, 200));
        let (num_q, den_q) = ans.max_frame_rate(SourceFormat::Qcif).unwrap();
        assert_eq!((num_q, den_q), (2997, 400));
    }

    #[test]
    fn parse_preference_order_spec_worked_example() {
        // §6.2.1 worked example "CIF=2;QCIF=1;D=1" advertises CIF first ⇒
        // CIF is the most preferred picture mode to be received.
        assert_eq!(
            H261FmtpParams::parse_preference_order("CIF=2;QCIF=1;D=1"),
            vec![SourceFormat::Cif, SourceFormat::Qcif],
        );
    }

    #[test]
    fn parse_preference_order_qcif_first() {
        // A peer that emits QCIF before CIF prefers QCIF — the structural
        // CIF-over-QCIF rule of `preferred_picture_size` would mis-report.
        assert_eq!(
            H261FmtpParams::parse_preference_order("QCIF=1;CIF=2"),
            vec![SourceFormat::Qcif, SourceFormat::Cif],
        );
        assert_eq!(
            H261FmtpParams::parse_preference_order("QCIF=1;CIF=2;D=1"),
            vec![SourceFormat::Qcif, SourceFormat::Cif],
        );
    }

    #[test]
    fn parse_preference_order_single_size() {
        assert_eq!(
            H261FmtpParams::parse_preference_order("CIF=2"),
            vec![SourceFormat::Cif],
        );
        assert_eq!(
            H261FmtpParams::parse_preference_order("QCIF=1"),
            vec![SourceFormat::Qcif],
        );
        // Annex D and unknown parameters interleaved with the picture size
        // don't affect the picture-size order.
        assert_eq!(
            H261FmtpParams::parse_preference_order("D=1;CIF=2;MAXBR=256"),
            vec![SourceFormat::Cif],
        );
    }

    #[test]
    fn parse_preference_order_no_picture_size_is_empty() {
        // §6.2.1 RFC 2032 fallback case ⇒ no picture-size token ⇒ empty
        // list; the caller is responsible for substituting the QCIF=1
        // default if it wants the §6.2.1 fallback applied.
        assert!(H261FmtpParams::parse_preference_order("").is_empty());
        assert!(H261FmtpParams::parse_preference_order("D=1").is_empty());
        assert!(H261FmtpParams::parse_preference_order("D=0;MAXBR=256").is_empty());
    }

    #[test]
    fn parse_preference_order_is_case_insensitive() {
        // SDP attribute token names are case-insensitive.
        assert_eq!(
            H261FmtpParams::parse_preference_order("qcif=1;cif=2"),
            vec![SourceFormat::Qcif, SourceFormat::Cif],
        );
        assert_eq!(
            H261FmtpParams::parse_preference_order("Cif=2;Qcif=1"),
            vec![SourceFormat::Cif, SourceFormat::Qcif],
        );
    }

    #[test]
    fn parse_preference_order_tolerates_whitespace_and_empty_tokens() {
        assert_eq!(
            H261FmtpParams::parse_preference_order(" CIF = 2 ; QCIF=1 "),
            vec![SourceFormat::Cif, SourceFormat::Qcif],
        );
        // Leading / trailing / double semicolons are ignored — same as
        // `parse_value`.
        assert_eq!(
            H261FmtpParams::parse_preference_order(";CIF=2;;QCIF=1;"),
            vec![SourceFormat::Cif, SourceFormat::Qcif],
        );
    }

    #[test]
    fn parse_preference_order_dedupes_repeated_tokens() {
        // A duplicate CIF / QCIF only contributes once — the first
        // occurrence sets the preferred-order position. (parse_value
        // rejects the duplicate as malformed; this helper is lenient
        // because its purpose is to mine wire-order from a possibly-
        // malformed offer.)
        assert_eq!(
            H261FmtpParams::parse_preference_order("CIF=2;CIF=3;QCIF=1"),
            vec![SourceFormat::Cif, SourceFormat::Qcif],
        );
        assert_eq!(
            H261FmtpParams::parse_preference_order("QCIF=1;CIF=2;QCIF=3"),
            vec![SourceFormat::Qcif, SourceFormat::Cif],
        );
    }

    #[test]
    fn parse_preference_order_skips_malformed_tokens() {
        // Missing '=' / unknown keys / out-of-range MPIs are all ignored
        // here — by design — so the caller can inspect a malformed offer's
        // preference order without parse_value rejecting it.
        //
        // `"CIF;QCIF=9;CIF=2"` ⇒ the bare `CIF` token is skipped (no `=`),
        // `QCIF=9` is kept first (range is ignored here), then `CIF=2`.
        assert_eq!(
            H261FmtpParams::parse_preference_order("CIF;QCIF=9;CIF=2"),
            vec![SourceFormat::Qcif, SourceFormat::Cif],
        );
        // Bare `QCIF` and out-of-range `CIF=9` survive: CIF wins because
        // it has an `=` and is encountered first.
        assert_eq!(
            H261FmtpParams::parse_preference_order("QCIF;CIF=9;QCIF=1"),
            vec![SourceFormat::Cif, SourceFormat::Qcif],
        );
    }

    #[test]
    fn parse_preference_order_agrees_with_parse_value_on_spec_example() {
        // For the §6.2.1 worked example, the first entry of the preference
        // order matches `preferred_picture_size` (both say CIF).
        let p = H261FmtpParams::parse_value("CIF=2;QCIF=1;D=1").unwrap();
        let order = H261FmtpParams::parse_preference_order("CIF=2;QCIF=1;D=1");
        assert_eq!(p.preferred_picture_size(), order.first().copied());
    }

    #[test]
    fn parse_preference_order_diverges_from_preferred_picture_size_on_qcif_first() {
        // The structural accessor returns CIF when both are present; the
        // wire-order helper honours the §6.2.1 "offered first" rule.
        let value = "QCIF=1;CIF=2;D=1";
        let p = H261FmtpParams::parse_value(value).unwrap();
        let order = H261FmtpParams::parse_preference_order(value);
        // Structural: CIF wins because it's advertised at all.
        assert_eq!(p.preferred_picture_size(), Some(SourceFormat::Cif));
        // Wire-order: QCIF wins because it's first in the offer.
        assert_eq!(order.first().copied(), Some(SourceFormat::Qcif));
    }

    #[test]
    fn display_covers_all_error_variants() {
        let cases: [SdpError; 6] = [
            SdpError::MpiOutOfRange {
                param: "CIF",
                value: 5,
            },
            SdpError::BadAnnexD {
                value: "2".to_string(),
            },
            SdpError::NotAnInteger {
                param: "QCIF".to_string(),
                value: "x".to_string(),
            },
            SdpError::MalformedToken {
                token: "CIF".to_string(),
            },
            SdpError::DuplicateParam {
                param: "CIF".to_string(),
            },
            SdpError::NoPictureSize,
        ];
        for e in &cases {
            let s = e.to_string();
            assert!(s.starts_with("h261 sdp:"), "unexpected Display: {s}");
        }
    }
}
