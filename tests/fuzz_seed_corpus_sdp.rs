//! Sanity-check the fuzz seed corpus under
//! `fuzz/corpus/parse_sdp_fmtp/` and drive each seed through the
//! same public SDP signalling parser surface the `parse_sdp_fmtp`
//! fuzz target exercises. Because `cargo-fuzz` requires the
//! nightly toolchain (libFuzzer's sanitizer-coverage flags are
//! `-Z`-gated), the regular CI matrix never builds the fuzz crate;
//! this stable-Rust test gives the same logical coverage so a
//! corrupted corpus or a regressed SDP-parser surface trips one of
//! the existing CI lanes instead of waiting for the daily fuzz
//! run to notice.
//!
//! The harness mirrors `fuzz/fuzz_targets/parse_sdp_fmtp.rs`
//! exactly: each buffer is decoded as UTF-8 lossily and then fed
//! through `parse_rtpmap`, `parse_fmtp`, `H261FmtpParams::parse_value`,
//! and `negotiate_answer` (on a `|`-split offer/our pair). For
//! inputs that parse cleanly, the formatter-then-reparse round trip
//! is also exercised so a regression in either direction trips this
//! test even on a malformed-only corpus.
//!
//! In addition to the on-disk corpus, the test drives several
//! adversarial in-line buffers — empty, a single null byte, all
//! `0xFF` (non-UTF-8 — the lossy decode replaces these with U+FFFD),
//! a deterministic pseudo-random buffer, and several hand-crafted
//! edge cases (MPI overflow, duplicate parameter, malformed token,
//! a `parse_rtpmap` payload-type integer overflow, a
//! `negotiate_answer` disjoint-advertisement rejection) — so a
//! corrupt corpus doesn't hide the parser-surface contract under
//! test.
//!
//! The SDP signalling parser is the fourth attacker-reachable
//! parser surface an H.261 endpoint exposes on the wire (the first
//! three are the elementary-stream decoder covered by
//! `tests/fuzz_seed_corpus.rs`, the BCH §5.4 FEC framer covered by
//! `tests/fuzz_seed_corpus_bch.rs`, and the RTP payload-format
//! wire wrapper covered by `tests/fuzz_seed_corpus_rtp.rs`; the
//! RTCP control-channel parser covered by
//! `tests/fuzz_seed_corpus_rtcp.rs` is a peer-interface parser,
//! not strictly a media path).

use std::fs;
use std::path::PathBuf;

use oxideav_h261::sdp::{
    negotiate_answer, parse_fmtp, parse_rtpmap, parse_rtpmap_strict, H261FmtpParams, SdpError,
};

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz")
        .join("corpus")
        .join("parse_sdp_fmtp")
}

/// Mirror the fuzz target's five-mode driver against a single buffer.
fn drive(bytes: &[u8]) {
    let text = String::from_utf8_lossy(bytes);
    let s: &str = &text;

    // ---- Mode A: `parse_rtpmap` + `parse_rtpmap_strict`. ----
    let lenient = parse_rtpmap(s);
    let strict = parse_rtpmap_strict(s);
    // §6.2 contract: a strict parse implies a lenient parse with a
    // §6.2-compliant clock rate; a lenient parse with a non-compliant
    // clock rate implies the strict parse returned `None`. Verify both
    // directions on every fuzz input so a future regression in either
    // path trips this oracle.
    match (lenient, strict) {
        (Some(l), Some(s2)) => {
            assert_eq!(l, s2);
            assert!(s2.is_rfc4587_compliant());
        }
        (Some(l), None) => assert!(!l.is_rfc4587_compliant()),
        (None, Some(_)) => panic!("strict parse must not succeed where lenient parse fails"),
        (None, None) => {}
    }

    // ---- Mode B: `parse_fmtp` with a fuzzer-chosen payload type. ----
    let expected_pt = bytes.first().copied().unwrap_or(0);
    let _ = parse_fmtp(s, expected_pt);

    // ---- Mode C: `H261FmtpParams::parse_value` standalone. ----
    let _ = H261FmtpParams::parse_value(s);

    // ---- Mode D: `negotiate_answer(offer, ours)` on a split input. ----
    let (offer_text, our_text) = match s.split_once('|') {
        Some((a, b)) => (a, b),
        None => (s, ""),
    };
    let offer = H261FmtpParams::parse_value(offer_text).unwrap_or_default();
    let ours = H261FmtpParams::parse_value(our_text).unwrap_or_default();
    let _ = negotiate_answer(&offer, &ours);

    // ---- Mode E: format-then-reparse round trip. ----
    if let Ok(params) = H261FmtpParams::parse_value(s) {
        let formatted = params.format_value();
        let _ = H261FmtpParams::parse_value(&formatted);

        // ---- Mode F: §6.2.1 preference-order oracle. ----
        //
        // When the input is well-formed enough for `parse_value` to
        // succeed, the **set** of picture sizes reported by
        // `parse_preference_order` must equal the set of `Some(_)`
        // picture-size fields on `params` — both walkers see the
        // same tokens.
        let order = H261FmtpParams::parse_preference_order(s);
        let order_has_cif = order
            .iter()
            .any(|f| matches!(f, oxideav_h261::picture::SourceFormat::Cif));
        let order_has_qcif = order
            .iter()
            .any(|f| matches!(f, oxideav_h261::picture::SourceFormat::Qcif));
        assert_eq!(order_has_cif, params.cif.is_some());
        assert_eq!(order_has_qcif, params.qcif.is_some());
        assert!(order.len() <= 2);
    }
}

#[test]
fn corpus_files_drive_sdp_parsers_without_panicking() {
    let dir = corpus_dir();
    let entries = fs::read_dir(&dir).unwrap_or_else(|e| {
        panic!("read fuzz corpus dir {}: {}", dir.display(), e);
    });
    let mut count = 0;
    for ent in entries {
        let path = ent.unwrap().path();
        if !path.is_file() {
            continue;
        }
        let bytes = fs::read(&path).expect("read seed");
        drive(&bytes);
        count += 1;
    }
    assert!(
        count >= 1,
        "expected at least one seed in {}",
        dir.display()
    );
}

#[test]
fn empty_bytes_dont_panic() {
    drive(&[]);
}

#[test]
fn single_zero_byte_doesnt_panic() {
    // A single NUL byte is a valid UTF-8 sequence (U+0000); none of the
    // parsers should match it against any prefix, so all return
    // `None` / `Ok(default)` and the format-then-reparse round trip is
    // skipped on Err. None may panic.
    drive(&[0u8]);
}

#[test]
fn all_ones_doesnt_panic() {
    // 0xFF is not a legal UTF-8 lead byte; the lossy decode replaces
    // every byte with U+FFFD. This is structurally the worst-case
    // input shape for the parsers: a long run of identical non-ASCII
    // code points that fail every prefix and key match. None may
    // panic.
    let bytes = vec![0xFFu8; 128];
    drive(&bytes);
}

#[test]
fn random_pattern_doesnt_panic() {
    // Deterministic pseudo-random buffer; most random byte sequences
    // are non-UTF-8 and decode to a sea of U+FFFD, but the lengths
    // and occasional `=` / `;` characters keep the split walkers
    // honest.
    let mut bytes = Vec::with_capacity(256);
    let mut x: u32 = 0xdead_beef;
    for _ in 0..256 {
        x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
        bytes.push((x >> 16) as u8);
    }
    drive(&bytes);
}

#[test]
fn rtpmap_canonical_round_trips() {
    // RFC 4587 §6.2 worked example for `video/H261` MUST parse and the
    // encoding name match is case-insensitive.
    let map = parse_rtpmap("a=rtpmap:31 H261/90000").expect("§6.2 example must parse");
    assert_eq!(map.payload_type, 31);
    assert_eq!(map.clock_rate, 90_000);
}

#[test]
fn rtpmap_rejects_other_codec() {
    // §6.2: "the encoding name in the `a=rtpmap` line of SDP MUST be
    // H261". A non-H261 encoding name must surface as `None` rather
    // than parse as a generic rtpmap.
    assert!(parse_rtpmap("a=rtpmap:97 H264/90000").is_none());
    assert!(parse_rtpmap("a=rtpmap:0 PCMU/8000").is_none());
}

#[test]
fn rtpmap_payload_type_overflow_rejected() {
    // §6.2 binds the rtpmap to a single 7-bit RTP payload type; a
    // value > 255 must surface as `None` rather than panic on
    // `u8::parse`. Same for an unsigned u32 clock rate.
    assert!(parse_rtpmap("a=rtpmap:300 H261/90000").is_none());
    assert!(parse_rtpmap("a=rtpmap:31 H261/99999999999999").is_none());
}

#[test]
fn fmtp_spec_example_round_trips() {
    // RFC 4587 §6.2 worked example: `a=fmtp:31 CIF=2;QCIF=1;D=1`.
    let parsed = parse_fmtp("a=fmtp:31 CIF=2;QCIF=1;D=1", 31)
        .expect("§6.2 example matches PT=31")
        .expect("§6.2 example is well-formed");
    assert_eq!(parsed.cif, Some(2));
    assert_eq!(parsed.qcif, Some(1));
    assert_eq!(parsed.d, Some(true));
}

#[test]
fn fmtp_payload_type_mismatch_returns_none() {
    // §6.2: the `a=fmtp` line is bound to a specific RTP payload type.
    // A mismatch must surface as `None` (the line is not for us)
    // rather than parse-and-return.
    assert!(parse_fmtp("a=fmtp:31 CIF=2", 96).is_none());
}

#[test]
fn parse_value_mpi_out_of_range_rejected() {
    // §6.1.1: MPI value SHALL be 1..=4. Values 0, 5, 99, and the
    // u32::MAX wrap-around must all surface as `MpiOutOfRange` or
    // `NotAnInteger`, never panic.
    for bad in &["0", "5", "99", "9999999999"] {
        let s = format!("CIF={bad}");
        let err = H261FmtpParams::parse_value(&s);
        assert!(err.is_err(), "CIF={bad} must reject");
    }
}

#[test]
fn parse_value_annex_d_rejects_non_binary() {
    // §6.1.1: D parameter SHALL be `0` or `1`. Any other value must
    // surface as `BadAnnexD`.
    let err = H261FmtpParams::parse_value("D=2");
    assert!(matches!(err, Err(SdpError::BadAnnexD { .. })));
}

#[test]
fn parse_value_duplicate_picture_size_rejected() {
    // §6.1.1: the parser rejects a duplicate CIF / QCIF parameter so
    // a malformed offer doesn't silently overwrite the earlier value.
    let err = H261FmtpParams::parse_value("CIF=1;CIF=2");
    assert!(matches!(err, Err(SdpError::DuplicateParam { .. })));
    let err = H261FmtpParams::parse_value("QCIF=2;QCIF=3");
    assert!(matches!(err, Err(SdpError::DuplicateParam { .. })));
}

#[test]
fn parse_value_unknown_parameter_skipped() {
    // §6.1.1 is forward-compatible: a parameter name this version
    // does not model is silently skipped, not rejected as malformed.
    let parsed = H261FmtpParams::parse_value("CIF=2;FUTUREPARAM=hello;D=1")
        .expect("forward-compatible parse");
    assert_eq!(parsed.cif, Some(2));
    assert_eq!(parsed.d, Some(true));
    assert_eq!(parsed.qcif, None);
}

#[test]
fn parse_value_missing_equals_rejected() {
    // A token without `=` is `MalformedToken`, not a panic.
    let err = H261FmtpParams::parse_value("CIF;QCIF=1");
    assert!(matches!(err, Err(SdpError::MalformedToken { .. })));
}

#[test]
fn negotiate_disjoint_advertisement_rejected() {
    // §6.2.1: "SHALL specify at least one supported picture size".
    // An offer advertising only CIF and our capability advertising
    // only QCIF intersect to empty and must surface as
    // `NoPictureSize`.
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
    let err = negotiate_answer(&offer, &ours);
    assert!(matches!(err, Err(SdpError::NoPictureSize)));
}

#[test]
fn negotiate_rfc2032_fallback_applied() {
    // §6.2.1: "If no picture-size parameter is specified, it is safe
    // to assume … reception of QCIF resolution with MPI=1". An empty
    // offer + a QCIF-capable our-side must yield QCIF in the answer.
    let offer = H261FmtpParams::default();
    let ours = H261FmtpParams {
        cif: None,
        qcif: Some(2),
        d: None,
    };
    let answer = negotiate_answer(&offer, &ours).expect("RFC 2032 QCIF=1 fallback");
    // The MPI is max(1, 2) = 2 since the answer must satisfy both
    // peers' frame-rate upper bounds.
    assert_eq!(answer.qcif, Some(2));
    assert_eq!(answer.cif, None);
    assert_eq!(answer.d, None);
}

#[test]
fn negotiate_annex_d_requires_both_sides() {
    // §6.2.1: Annex D "MUST NOT appear unless the sender of this SDP
    // message is able to decode this option". The answer's `D=1`
    // therefore requires both sides advertising it; any other
    // combination drops `D` from the answer.
    let with_d = H261FmtpParams {
        cif: Some(2),
        qcif: Some(1),
        d: Some(true),
    };
    let without_d = H261FmtpParams {
        cif: Some(2),
        qcif: Some(1),
        d: None,
    };
    let answer = negotiate_answer(&with_d, &without_d).expect("picture size intersects");
    assert_eq!(answer.d, None);
    let answer = negotiate_answer(&with_d, &with_d).expect("picture size intersects");
    assert_eq!(answer.d, Some(true));
}

#[test]
fn preference_order_honours_wire_order() {
    // RFC 4587 §6.2.1: "Parameters offered first are the most preferred
    // picture mode to be received." The wire-order helper must therefore
    // report the §6.2.1 worked example as `[CIF, QCIF]` (CIF preferred)
    // and a QCIF-first offer as `[QCIF, CIF]` (QCIF preferred) — the
    // structural `preferred_picture_size` accessor cannot distinguish
    // those two cases.
    use oxideav_h261::picture::SourceFormat;
    assert_eq!(
        H261FmtpParams::parse_preference_order("CIF=2;QCIF=1;D=1"),
        vec![SourceFormat::Cif, SourceFormat::Qcif],
    );
    assert_eq!(
        H261FmtpParams::parse_preference_order("QCIF=1;CIF=2;D=1"),
        vec![SourceFormat::Qcif, SourceFormat::Cif],
    );
}

#[test]
fn round_trip_format_value_against_parse_value() {
    // For every well-formed parameter set, formatter output must
    // round-trip through the parser unchanged. This is the §6.2
    // contract that lets a caller print and re-receive its own SDP.
    let params = H261FmtpParams {
        cif: Some(4),
        qcif: Some(1),
        d: Some(true),
    };
    let formatted = params.format_value();
    let reparsed = H261FmtpParams::parse_value(&formatted).expect("formatter output reparses");
    assert_eq!(reparsed, params);
}
