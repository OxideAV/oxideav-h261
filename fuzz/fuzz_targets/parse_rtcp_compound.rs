#![no_main]

//! Drive arbitrary fuzz-supplied bytes through the H.261 crate's RTCP
//! control-channel parser surface (RFC 3550 §6).
//!
//! Unlike the H.261 elementary-stream decoder, the RTCP packet parser
//! consumes bytes that arrive from a *peer on the network*. The bytes are
//! attacker-controlled, the packet types are extensible (PT 200..=204 are
//! modelled here; 205..=255 surface as `Other`), and each sub-packet's
//! reach into the buffer is governed by its own 16-bit `length` field —
//! exactly the shape a "walk the self-delimited TLV stream" parser is
//! vulnerable to mis-implementing.
//!
//! The wire format under test:
//!
//! * **Compound walk** (§6.1) — `parse_compound` walks consecutive
//!   sub-packets via each header's `length = words - 1` advance.
//!   A malformed advance (truncated header, stated length past the
//!   buffer end, length field claiming 0 words → would loop forever
//!   if the +1 wasn't honoured) must surface as `Err`, not a panic
//!   or hang.
//! * **SR / RR** (§6.4.1 / §6.4.2) — `parse_report` decodes V/P/RC,
//!   the 8-byte fixed header, the 20-byte sender-info section (SR only),
//!   and up to 31 24-byte reception report blocks. RC count vs declared
//!   length must reconcile or the parser rejects.
//! * **SDES** (§6.5) — `parse_sdes` walks one or more chunks, each
//!   binding an SSRC to a NUL-terminated list of (type, 8-bit length,
//!   text) items. The PRIV (item type 8) sub-format carries an inner
//!   8-bit-length prefix and value, so the parser does two nested
//!   length walks against a single buffer. SDES item text is decoded
//!   UTF-8-lossily so a malformed packet from the network never panics.
//! * **BYE** (§6.6) — `parse_bye` reads up to 31 SSRCs from the
//!   header SC field and an optional 8-bit-length-prefixed reason
//!   string. The reason length field is similarly attacker-controlled.
//! * **APP** (§6.7) — `parse_app` enforces V=2, PT=204, length>=12,
//!   and a 4-octet ASCII name; remaining data is opaque but must be a
//!   multiple of 32 bits per §6.7.
//!
//! The contract under test is purely that every parser call *returns*:
//! a malformed datagram yields `Err(RtcpError::…)`; a well-formed one
//! yields `Ok(...)`. No path may panic, abort, integer-overflow (in a
//! debug / ASAN build), index out of bounds, or OOM.
//!
//! The harness also feeds the bytes through the individual parser
//! entry points (`parse_report`, `parse_sdes`, `parse_bye`, `parse_app`)
//! so a fuzz-discovered input that confuses one parser specifically (not
//! framed as a complete compound packet) still gets coverage.

use libfuzzer_sys::fuzz_target;
use oxideav_h261::rtcp::{parse_app, parse_bye, parse_compound, parse_report, parse_sdes};

fuzz_target!(|data: &[u8]| {
    // §6.1 compound walk — the canonical receiver entry point.
    let _ = parse_compound(data);

    // Each sub-packet parser exercised in isolation as well, so a
    // single bad sub-packet that the compound walk rejects before
    // reaching it is still driven through its own parser. None of the
    // parsers allocate beyond `Vec::new()` on the error path.
    let _ = parse_report(data);
    let _ = parse_sdes(data);
    let _ = parse_bye(data);
    let _ = parse_app(data);
});
