#![no_main]

//! Drive arbitrary fuzz-supplied bytes through the H.261 crate's
//! Session Description Protocol (SDP) parser surface — the
//! attribute-line parsers an endpoint runs on every received SDP
//! offer or answer before any RTP / RTCP / H.261 layer sees a byte.
//!
//! The four prior fuzz targets cover the data-path (`decode_h261`),
//! the RTCP control channel (`parse_rtcp_compound`), the BCH §5.4
//! FEC framing (`decode_bch_multiframe`), and the RTP payload
//! wire format (`parse_rtp_payload`). This target rounds the
//! attacker-reachable parser surface out by driving the **SDP
//! signalling** layer — the `a=rtpmap` and `a=fmtp` attribute
//! lines defined for `video/H261` by RFC 4587 §6.1.1 / §6.2 /
//! §6.2.1.
//!
//! The SDP wire format is text, not bytes — the parsers consume
//! `&str`. We therefore decode the fuzz input as UTF-8 lossily
//! once per iteration and feed the resulting string through each
//! parser. (Real-world SDP arrives over SIP / WebRTC / RTSP as
//! UTF-8 text, so the lossy decode preserves the attacker's reach
//! into every code path the production parsers can hit.)
//!
//! ## Four-mode driver
//!
//! Mode A — **`parse_rtpmap`** — drive the bare line parser, which
//! accepts an `a=rtpmap:<pt> <enc>/<clock>[/<params>]` line and
//! returns `Some(RtpMap)` only if it parses and the encoding name is
//! `H261` (case-insensitive). The parser bounds-checks the integer
//! payload type and the clock rate against `u8` / `u32` overflow,
//! tolerates leading whitespace and an optional `a=` prefix, and
//! must never panic on a malformed line.
//!
//! Mode B — **`parse_fmtp`** + **`parse_fmtp_strict`** — drive the
//! SDP `a=fmtp` line parser against arbitrary text plus a
//! fuzzer-chosen payload type. The lenient parser returns `None` if
//! the line is not an `fmtp` attribute or its embedded payload type
//! does not match; otherwise it delegates to
//! `H261FmtpParams::parse_value` which walks the semicolon-separated
//! `key=value` list, enforcing the §6.1.1 ranges (CIF/QCIF MPI
//! ∈ 1..=4, `D` ∈ {0, 1}), the §6.1.1 duplicate-parameter rule, and
//! the §6.1.1 forward-compatible "ignore unknown parameter"
//! behaviour. The §6.2.1 strict variant additionally enforces
//! "Implementations following this specification SHALL specify at
//! least one supported picture size"; we check both parsers and
//! assert the strict ⇔ lenient + §6.2.1 invariant on every input so
//! a future divergence between the two paths trips the fuzz loop.
//!
//! Mode C — **`H261FmtpParams::parse_value`** — feed the fuzz
//! string straight at the bare parameter-list parser so the
//! `split(';')` walker, the `trim()` / `split_once('=')` / case-
//! folding key lookup, and the per-parameter integer / Annex-D
//! parsers are reachable even when the `a=fmtp:` wrapper would
//! reject the input upstream.
//!
//! Mode D — **`negotiate_answer`** — split the fuzz string into
//! two halves on a deterministic boundary, parse each half as a
//! parameter list (silently substituting an empty list for a
//! malformed half), and feed the resulting `(offer, our)` pair to
//! the §6.2.1 offer/answer helper. This drives the RFC-2032
//! fallback path (offer with no picture-size parameters), the
//! `max(offer.MPI, our.MPI)` per-shared-size rule, the
//! `(Some(true), Some(true)) ⇒ Some(true)` Annex-D rule, and the
//! `NoPictureSize` rejection on a disjoint advertisement.
//!
//! The contract under test is uniform across all modes:
//! every parser call must *return* — a malformed input yields
//! `None` / `Err(SdpError::…)`; a well-formed input yields
//! `Some(…)` / `Ok(…)`. No path may panic, abort, integer-overflow
//! (in a debug / ASAN build), index out of bounds, or OOM,
//! regardless of how hostile the attacker's bytes are.

use libfuzzer_sys::fuzz_target;
use oxideav_h261::sdp::{
    negotiate_answer, parse_fmtp, parse_fmtp_strict, parse_rtpmap, parse_rtpmap_strict,
    H261FmtpParams, SdpError,
};

fuzz_target!(|data: &[u8]| {
    // SDP is line-oriented text. The production parsers consume `&str`;
    // a lossy UTF-8 decode is the cheapest faithful path from
    // attacker-supplied bytes to those parsers. Any non-UTF-8 bytes
    // become the U+FFFD replacement character — same shape the
    // production stack sees from a SIP / WebRTC stack that has already
    // done its own UTF-8 sanitisation.
    let text = String::from_utf8_lossy(data);
    let s: &str = &text;

    // ---- Mode A: `parse_rtpmap` + `parse_rtpmap_strict` oracle. ----
    //
    // The vast majority of random strings fail the `rtpmap:` prefix
    // check and short-circuit out as `None`. The strings that pass
    // exercise the payload-type integer parse (must reject `> 255`),
    // the encoding-name match (must reject non-`H261`
    // case-insensitively), and the clock-rate integer parse (must
    // reject `> u32::MAX`). The §6.2 strict variant additionally
    // enforces `clock_rate == 90000`; we check both parsers and assert
    // the strict ⇔ lenient + §6.2 invariant on every input so a future
    // divergence between the two paths trips the fuzz loop.
    let lenient = parse_rtpmap(s);
    let strict = parse_rtpmap_strict(s);
    match (lenient, strict) {
        (Some(l), Some(s2)) => {
            assert_eq!(l, s2);
            assert!(s2.is_rfc4587_compliant());
        }
        (Some(l), None) => assert!(!l.is_rfc4587_compliant()),
        (None, Some(_)) => panic!("strict parse must not succeed where lenient parse fails"),
        (None, None) => {}
    }

    // ---- Mode B: `parse_fmtp` + `parse_fmtp_strict` oracle. ----
    //
    // The first byte of the fuzz input selects the expected payload
    // type modulo 256 so we hit both the "PT matches" and "PT
    // mismatches" branches. (An empty input falls back to PT=0.)
    // §6.2.1 contract on the strict path: a strict success implies a
    // lenient success with **equal** parsed params AND a passing
    // `validate()`; a lenient success that fails `validate()` (no
    // CIF/QCIF) implies the strict variant returned `None`; a parse
    // error propagates byte-for-byte through both paths. We check all
    // directions on every fuzz input so a future regression in either
    // parser trips the fuzz loop.
    let expected_pt = data.first().copied().unwrap_or(0);
    let lenient_fmtp = parse_fmtp(s, expected_pt);
    let strict_fmtp = parse_fmtp_strict(s, expected_pt);
    match (&lenient_fmtp, &strict_fmtp) {
        (Some(Ok(l)), Some(Ok(s2))) => {
            assert_eq!(l, s2);
            assert!(s2.validate().is_ok());
        }
        (Some(Ok(l)), None) => {
            assert!(matches!(l.validate(), Err(SdpError::NoPictureSize)));
        }
        (Some(Err(le)), Some(Err(se))) => assert_eq!(le, se),
        (None, None) => {}
        (Some(Err(_)), None) => {
            panic!("strict must propagate parse errors as Some(Err(_))");
        }
        (None, Some(_)) => {
            panic!("strict must not succeed where lenient parse fails");
        }
        (Some(Ok(_)), Some(Err(_))) | (Some(Err(_)), Some(Ok(_))) => {
            panic!("strict / lenient parse disagree on Ok/Err");
        }
    }

    // ---- Mode C: `H261FmtpParams::parse_value` against raw text. ----
    //
    // The bare parameter-list parser is the inner walker of mode B;
    // driving it standalone covers inputs that mode B would reject at
    // the `a=fmtp:` wrapper. Each `split(';')` token goes through
    // `trim()`, `split_once('=')`, case-folding for the parameter
    // name, and the per-parameter integer / Annex-D parse.
    let _ = H261FmtpParams::parse_value(s);

    // ---- Mode D: `negotiate_answer(offer, ours)` on a split input. ----
    //
    // Split the fuzz text on the first '|' character (a separator
    // that does not appear in legal SDP `fmtp` parameter syntax) so
    // the fuzzer can drive offer and our-capability halves
    // independently. A malformed half is silently treated as the
    // default (all-`None`) parameter set so the negotiator still
    // sees a structurally well-formed `(offer, ours)` pair — the
    // negotiator's own behaviour, not its inputs, is what we're
    // fuzzing here.
    let (offer_text, our_text) = match s.split_once('|') {
        Some((a, b)) => (a, b),
        None => (s, ""),
    };
    let offer = H261FmtpParams::parse_value(offer_text).unwrap_or_default();
    let ours = H261FmtpParams::parse_value(our_text).unwrap_or_default();
    let _ = negotiate_answer(&offer, &ours);

    // ---- Mode E: format-then-reparse round trip when parse succeeded. ----
    //
    // For inputs that successfully parse as a parameter list, exercise
    // the formatter and re-parse the formatter output. A round-trip
    // mismatch would point at a `format_value` / `parse_value`
    // disagreement; an unstable round trip would point at a bug in
    // either direction. (No assertion: the only contract is "must
    // return"; a true mismatch would be caught by `format_value`'s
    // unit tests on the canonical spec example.)
    if let Ok(params) = H261FmtpParams::parse_value(s) {
        let formatted = params.format_value();
        let _ = H261FmtpParams::parse_value(&formatted);

        // ---- Mode F: §6.2.1 preference-order oracle. ----
        //
        // `parse_preference_order` is lenient by design (it skips
        // malformed tokens so a caller can mine wire-order from a
        // possibly-malformed offer), so we only oracle it when the
        // input was already well-formed enough for `parse_value` to
        // succeed. In that case the **set** of picture sizes reported
        // by the order helper must equal the set of `Some(_)` picture
        // size fields on `params` — both walkers see the same tokens.
        let order = H261FmtpParams::parse_preference_order(s);
        let order_has_cif = order
            .iter()
            .any(|f| matches!(f, oxideav_h261::picture::SourceFormat::Cif));
        let order_has_qcif = order
            .iter()
            .any(|f| matches!(f, oxideav_h261::picture::SourceFormat::Qcif));
        assert_eq!(order_has_cif, params.cif.is_some());
        assert_eq!(order_has_qcif, params.qcif.is_some());
        // No duplicates: at most one CIF and at most one QCIF entry.
        assert!(order.len() <= 2);
    }
});
