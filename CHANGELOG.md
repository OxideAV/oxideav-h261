# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **§4.2.3.4 MVD predictor reset at MB-row boundaries (MBA 12 and 23).**
  The decoder's motion-vector-data predictor only reset to zero at GOB
  start, on MBA discontinuities, and when the previous MB was not
  motion-compensated. It was missing §4.2.3.4 condition (1): the
  predictor "is regarded as zero" for macroblocks 1, 12 and 23 (the
  first MB of each of the three rows in an 11×3-MB GOB). MBA 1 was
  already covered by the per-GOB context reset, but MBA 12 and 23 were
  not — so a conformant stream carrying a non-zero MV at MB 11 (or 22)
  immediately followed by a motion-compensated MB 12 (or 23) decoded
  the wrong vector. The in-crate encoder had previously worked around
  this by forcing the MV to zero at MBs 11 and 22 to keep the two
  sides in agreement; with the decoder now spec-conformant, that
  constraint is removed and the encoder may use motion compensation at
  every MB. The RFC 4587 §4.2 MB-level fragmentation walker, which
  carries its own §4.2.3.4 predictor tracking, was given the same
  reset so it stays bit-for-bit in lockstep with the decoder. A new
  shared `mb::mvd_predictor` helper is the single source of truth for
  the three reset conditions, unit-tested across all of them.

### Added

- **`filter_mc` criterion benchmark — §3.2.3 loop filter + §3.2.2
  integer-pel motion-comp.** The existing `transform` bench covered
  the inner (I)DCT, and `encode` / `decode` cover end-to-end picture
  cost, but the two *other* per-block P-picture reconstruction
  primitives the decoder runs on every coded P-block had no isolated
  baseline. The new `benches/filter_mc.rs` times `mb::apply_loop_filter`
  (the separable 1/4-1/2-1/4 in-loop filter with 0-1-0 edge taps) and
  `mb::copy_block_integer` (the integer-pel reference fetch) across
  three motion regimes (`center`, `mv_nonzero`, `corner_clamp`). Both
  functions are now `pub` (matching the existing `fdct` / `idct`
  primitive exports) so an optimisation pass — a SIMD loop filter or a
  branchless edge-clamp copy — has an A/B baseline distinct from the
  transform numbers. Round-287 release-build aarch64 baseline: loop
  filter ≈ 25 ns / block (≈ 2.5 Gelem/s); integer-pel copy ≈ 15 ns /
  block (≈ 4.1 Gelem/s) interior, ≈ 14.5 ns fully corner-clamped.

- **RFC 4587 §4.2 MB-level fragmentation.** The RTP module previously
  shipped only the "cheap" GOB-aligned packetizer; a GOB larger than
  the payload budget was split at arbitrary byte boundaries with
  zeroed context fields, so its continuation packets were not
  independently decodable after a loss — the exact problem the §4.1
  H.261 header exists to solve. The new
  `rtp::packetize_mb_fragmented` implements the §4.2 RECOMMENDED
  packetization: a single Huffman-layer walk over the elementary
  stream (`walk_mb_split_points` — MBA/MTYPE/MQUANT/MVD/CBP/TCOEFF
  VLCs parsed, nothing dequantised or transformed, per §4.2 "it is
  not necessary to decompress the stream fully") records every legal
  split point with its §4.1 context, then packets are filled greedily
  (multiple GOBs/MBs per packet when they fit, per §3.2) under the
  §3.2 rules: an MB is never split across packets, the stream is
  never fragmented between a GOB header and MB 1, and no packet
  crosses a PSC. Mid-GOB packets carry non-zero SBIT/EBIT plus the
  GOBN / MBAP (biased -1) / QUANT / HMVD / VMVD context; the walker
  tracks the §4.2.3.4 MV predictor (including the consecutive-MBA
  and last-MB-was-MC rules) so the reference MVD is exact.
  `RtpPacketizer::with_mb_fragmentation(true)` routes `pack_frame`
  through the new path with an automatic fallback to the byte-split
  cheap packetizer when no MB-boundary split exists, and two new
  `RtpError` variants (`MalformedStream`, `FragmentTooLarge`) surface
  walk/budget failures on the direct path. Eleven new tests cover
  it: the Huffman-layer walk is checked bit-for-bit against a
  real-decoder (`decode_macroblock`) oracle on I-pictures across a
  quantiser sweep and on a P-picture with live motion vectors; round
  trips at multiple budgets are byte-exact through `depacketize`
  (which already handled non-zero SBIT/EBIT); fragment chains are
  verified bit-contiguous (shared split byte,
  `next.SBIT == (8 - prev.EBIT) % 8`); continuation headers are
  matched back to walker split points; the PSC-crossing ban, the
  whole-frame-in-one-packet case, the `FragmentTooLarge` path, and
  two end-to-end RTP-session decodes (`tests/rtp_e2e.rs`) round it
  out.

- **RFC 4587 §6.2.1 preference-aware `a=fmtp` formatter.** §6.2.1
  states "Parameters offered first are the most preferred picture
  mode to be received" — an endpoint expresses its receive preference
  purely through token order. The fixed-order `format_value` /
  `format_fmtp` are locked to the §6.2.1 worked-example CIF-first
  order, so an endpoint advertising both picture sizes but preferring
  to receive QCIF could not express that on the wire. The new
  `H261FmtpParams::format_value_preferred(preferred)` method and
  `format_fmtp_preferred(pt, &params, preferred)` free function emit
  the preferred picture-size token first when that size is advertised
  (`QCIF=1;CIF=2;D=1` for a QCIF-preferring endpoint), the other
  advertised size second, and `D` last (`D` is an Annex-D codec
  option, not a picture mode, so the §6.2.1 "offered first" rule does
  not order it). A CIF preference is byte-identical to the canonical
  formatter (`format_value` is now a thin wrapper over
  `format_value_preferred(SourceFormat::Cif)`); an unadvertised
  preference falls back to the canonical order; the §6.2 "if any"
  no-parameters ⇒ no-line rule is preserved. This is the emit-side
  dual of the parse-side `parse_preference_order` accessor: whenever
  the params advertise the preferred size, the leading entry of
  `parse_preference_order(format_value_preferred(fmt))` is `fmt`,
  closing the §6.2.1 wire-order loop in both directions. Five new
  unit tests in `src/sdp.rs` cover CIF-preference identity across
  five parameter shapes, the QCIF-first emission + wire-order
  read-back, the unadvertised-preference fallback (both directions),
  the parse round trip under both preferences, and the full
  `a=fmtp` line builder (QCIF-first line, CIF-preference byte
  equality with `format_fmtp`, empty-params `None`, reparse through
  `parse_fmtp`). The existing fuzz target `parse_sdp_fmtp` and the
  stable-CI `tests/fuzz_seed_corpus_sdp.rs` driver gain a Mode G
  oracle: on every input that parses cleanly, (1)
  `format_value_preferred(Cif) == format_value()`, (2) both
  preference emissions reparse to equal params, and (3) when the
  params advertise the preferred size, `parse_preference_order`
  reads it back as the leading entry. New seed
  `fuzz/corpus/parse_sdp_fmtp/14_fmtp_value_both_sizes_d0.txt`
  carries a both-sizes + `D=0` bare parameter list. README's
  SDP-media-type section is updated with the emit-side accessor.

- **RFC 4587 §6.2.1 strict-conformance `a=fmtp` parser.** §6.2.1
  states "Implementations following this specification SHALL specify
  at least one supported picture size." The lenient `parse_fmtp`
  deliberately preserves a well-formed-but-§6.2.1-violating line
  (e.g. `a=fmtp:31 D=1`) so a caller that wants to recover the
  parsed `D` value from a slightly-non-conformant peer can still see
  what the wire said; the new `parse_fmtp_strict(line, payload_type)`
  free function chains lenient parse + `H261FmtpParams::validate()`
  so an SDP front end that wants to drop a §6.2.1-violating offer
  can do so with one call. Mirrors the existing `parse_rtpmap` /
  `parse_rtpmap_strict` pair on the `a=rtpmap` side of §6.2:
  malformed parameter lists still surface as `Some(Err(SdpError::…))`
  (the §6.2.1 picture-size check runs only after a successful parse)
  so typed error propagation is preserved on the strict path; the
  §6.2.1 RFC-2032 fallback (assume QCIF=1) is **not** silently
  applied during strict validation because that fallback is a
  negotiation rule, not an advertisement-validation rule (callers
  that want it combine `parse_fmtp` with
  `H261FmtpParams::rfc2032_fallback()` themselves). Eight new unit
  tests in `src/sdp.rs` cover the §6.2.1 worked example (strict ==
  lenient), the empty-parameter-list rejection, the `D=1`-only
  rejection, malformed-token error propagation, payload-type
  mismatch, QCIF-only / CIF-only acceptance, "no fallback applied"
  documentation, and a cross-cutting strict-implies-lenient sweep
  on five §6.2.1-compliant lines. The existing fuzz target
  `parse_sdp_fmtp` and the stable-CI `tests/fuzz_seed_corpus_sdp.rs`
  driver gain a Mode B oracle: every fuzz input is run through both
  `parse_fmtp` and `parse_fmtp_strict`, asserting `(Some(Ok(l)),
  Some(Ok(s)))` ⇒ `l == s` and `s.validate().is_ok()`; `(Some(Ok(l)),
  None)` ⇒ `l.validate() == Err(NoPictureSize)`; `(Some(Err(le)),
  Some(Err(se)))` ⇒ `le == se`; and the four impossible
  combinations panic (strict succeeding where lenient fails, strict
  dropping a parse error, or the two paths disagreeing on Ok/Err).
  New seed
  `fuzz/corpus/parse_sdp_fmtp/13_fmtp_no_picture_size_strict_rejects.txt`
  carries the canonical "lenient accepts / strict rejects"
  `a=fmtp:31 D=1` input. README's SDP-media-type section is updated
  with the strict-parser accessor.

- **RFC 4587 §6.2.1 wire-order preference accessor for `a=fmtp`.**
  §6.2.1 says "Parameters offered first are the most preferred picture
  mode to be received", and the spec's worked example
  `a=fmtp:31 CIF=2;QCIF=1;D=1` advertises CIF first ⇒ CIF is the
  preferred mode. The existing structural `preferred_picture_size`
  accessor cannot distinguish that case from a peer that emits
  `QCIF=1;CIF=2` (QCIF preferred) because `H261FmtpParams` stores
  `cif` and `qcif` as independent `Option<u8>` fields and discards
  token order. The new `H261FmtpParams::parse_preference_order(&str)`
  helper walks the same `;`-separated token list `parse_value` does
  and returns `Vec<SourceFormat>` in wire order (skipping unknown /
  malformed / Annex-D tokens, deduplicating, and ignoring MPI range
  so the helper can mine wire-order from a possibly-malformed offer).
  The first entry, if any, is the peer's preferred mode per §6.2.1's
  "offered first" rule. New unit tests in `src/sdp.rs` cover the §6.2.1
  worked example, the QCIF-first variant, single-size advertisements,
  the RFC 2032 fallback (empty list ⇒ caller substitutes QCIF=1),
  case-insensitive parameter names, whitespace tolerance, deduplication,
  malformed-token skipping, and the divergence from
  `preferred_picture_size` on a QCIF-first input. The existing fuzz
  target `parse_sdp_fmtp` and the stable-CI `tests/fuzz_seed_corpus_sdp.rs`
  driver gain a Mode F oracle: whenever `parse_value` succeeds, the
  **set** of picture sizes reported by `parse_preference_order` must
  equal the set of `Some(_)` picture-size fields on the parsed params
  (the two walkers see the same tokens), and the order list never
  exceeds two entries. New seed
  `fuzz/corpus/parse_sdp_fmtp/12_fmtp_qcif_preferred_first.txt`
  carries the canonical QCIF-first wire-order input. README's
  SDP-media-type section is updated with the wire-order accessor.

- **RFC 4587 §6.2 strict-conformance accessor for `RtpMap`.** §6.2
  states "The clock rate in the `a=rtpmap` line MUST be 90000."
  `parse_rtpmap` deliberately preserves the wire `clock_rate`
  verbatim so a misbehaving peer can be diagnosed without losing the
  parsed payload type; the typed accessor `RtpMap::is_rfc4587_compliant`
  is the single-call check for "did the peer follow §6.2?", and the
  new `parse_rtpmap_strict` free function combines parse +
  conformance into one call (returns `None` for any well-formed but
  non-90000 line, plus everything `parse_rtpmap` already rejects).
  Two sweep tests in `src/sdp.rs` cover the §6.2 worked example
  (`8000` and `180000` lenient-but-not-strict variants, an RFC 2032
  rtpmap that happens to share §6.2's clock rate, the optional
  third `<params>` field, payload-type and encoding-name rejections
  inherited from `parse_rtpmap`); the existing fuzz target
  `parse_sdp_fmtp` and the stable-CI `tests/fuzz_seed_corpus_sdp.rs`
  driver now run a strict-vs-lenient oracle on every fuzz input
  (`(Some(l), Some(s))` ⇒ `l == s` and `s.is_rfc4587_compliant()`;
  `(Some(l), None)` ⇒ `!l.is_rfc4587_compliant()`; the
  "strict succeeds where lenient fails" case panics) so a future
  regression in either path trips the daily fuzz run. New seed
  `fuzz/corpus/parse_sdp_fmtp/11_rtpmap_non_90000_clock_rate.txt`
  carries the canonical lenient-but-not-strict input
  (`a=rtpmap:31 H261/8000`). README's "SDP media type and
  rtpmap/fmtp parameters" section is updated with the strict accessor.

- **Criterion bench for the §4.1 / §4.2 start-code scanner.** New
  `benches/start_code.rs` adds a fifth Criterion bench to the
  round-175 `transform` / `encode` / `decode` + round-233 `bch`
  suite, covering the bit-by-bit scanner
  (`find_next_start_code_bits`, `find_next_start_code`,
  `iter_start_codes`) that sits on the inner loop of every
  `H261Decoder::send_packet`, every `rtp::packetize_gob_aligned`,
  and the spec-mandated RFC 4587 §4 `rtp::depacketize` sanity
  check. Two groups: `h261_start_code_iter` walks one full QCIF
  I-frame (1 PSC + 3 GBSCs), one full CIF I-frame (1 PSC + 12
  GBSCs), and three concatenated QCIF I-frames; `h261_start_code_single`
  covers `find_next_start_code` on the first-byte-aligned PSC
  (best case),  `find_next_start_code_bits` starting at bit 3
  (the §4.2 GOB-aligned packetizer's slow path when a GOB does
  not land on a byte boundary), and `find_next_start_code` on a
  pseudo-random 4 KiB buffer with every accidental `0x0001` window
  flipped out (worst case — scan to end, return `None`, the cost
  a misbehaving network endpoint can force on an RTP receiver per
  fuzzed payload). Headline points on the round-238 baseline
  (release, aarch64): the bit-by-bit scanner clocks ≈ 295–300 MiB/s
  across all three full-stream walks (QCIF I-frame 15.1 µs at
  295 MiB/s, CIF I-frame 60.2 µs at 297 MiB/s, three-QCIF 44.7 µs
  at 300 MiB/s); a byte-aligned first-PSC hit costs ≈ 6 ns; a
  3-bit-misaligned hit costs ≈ 18 ns; the worst-case 4 KiB no-hit
  walk takes ≈ 13 µs (298 MiB/s). All inputs are synthesised
  in-bench (encoder output for hit cases, xorshift + sweep-flip
  for the no-hit case); no on-disk fixtures, no third-party tools,
  no `docs/` files at bench time. `Cargo.toml` gains a fifth
  `[[bench]]` entry (`start_code`, `harness = false`); the README's
  `### Benchmarks` section is updated with the new target's
  sub-scenarios and headline numbers. An eventual SIMD pre-scan
  over byte-aligned `0x00 0x01` candidates plus the bit-walk on
  the few near-hit windows is the obvious follow-up; this bench
  gives that change its A/B baseline.

- **Criterion bench for the §5.4 BCH (511, 493) FEC layer.** New
  `benches/bch.rs` rounds out the round-175 `transform` / `encode` /
  `decode` Criterion suite with the outer-coding hot path. Three
  groups: `bch_primitives` covers `parity18` and `syndrome18` (the
  493-bit shift-register long-division primitives, once each per
  emitted / received §5.4.3 frame); `bch_correction` covers the
  §5.4.1 `locate_single_error` walk on a worst-case syndrome
  (constructed locally as `x^510 mod g(x)` so the walk takes the
  full 511 iterations before returning `Some`) and on an
  uncorrectable syndrome (weight ≥ 2, returns `None` after the same
  511 iterations); `bch_multiframe` covers the integrated
  `encode_multiframe` / `decode_multiframe` /
  `decode_multiframe_with_correction` entry points on one full
  8-frame §5.4.4 multiframe (clean + one-bit-corrupted variants).
  Headline points on the round-233 baseline (release, aarch64):
  `parity18` ≈ 350 ns / frame (≈ 0.7 ns / data bit), `syndrome18`
  ≈ 346 ns / frame, `locate_single_error` worst-case ≈ 460 ns and
  uncorrectable ≈ 448 ns, `encode_multiframe` ≈ 12.2 µs / multiframe,
  `decode_multiframe` clean ≈ 24.5 µs, `decode_multiframe`
  one-bit-corrupted ≈ 24.2 µs, and
  `decode_multiframe_with_correction` one-bit-corrupted ≈ 24.0 µs
  (the §5.4.1 correcting decoder adds essentially no overhead over
  the detection-only decoder when at most one frame in the
  multiframe is corrupted — the `locate_single_error` walk is dwarfed
  by the per-frame syndrome work the decoder already does). All
  inputs are synthesised in-bench from deterministic xorshift seeds;
  no on-disk fixtures, no third-party tools, no `docs/` files are
  read at bench time. `Cargo.toml` gains a fourth `[[bench]]` entry
  (`bch`, `harness = false`); README's `### Benchmarks` section is
  updated with the new target's sub-scenarios and headline numbers.

- **BCH (511,493) single-bit error correction (§5.4.1, t = 1).** New
  `bch::locate_single_error` maps a non-zero 18-bit syndrome to the
  corresponding 511-bit codeword position (where `p = 0` is `Fi`,
  `p = 1..493` are the 492 data bits, `p = 493..511` are the 18 parity
  bits) by walking `pow = x^i mod g(x)` for `i = 0..511` and matching
  the running power against the syndrome; a non-zero syndrome that
  doesn't match any single-bit pattern reports `None` (the weight ≥ 2
  case the t = 1 code cannot resolve). New `decode_multiframe_with_correction`
  is the integrated path: same alignment-pattern lock as
  `decode_multiframe`, plus `locate_single_error` is called on every
  non-zero-syndrome frame and the indicated bit is flipped inside the
  511-bit codeword before the per-frame `Fi` / data interpretation
  runs. `DecodedMultiframe` gains two new counters — `corrected_frames`
  (subset of `corrupted_frames` that were successfully corrected) and
  `uncorrectable_frames` (complement: weight ≥ 2 patterns the code
  cannot resolve). The detection-only `decode_multiframe` preserves its
  prior behaviour and reports `corrected_frames = 0,
  uncorrectable_frames = 0`. A sweep test in `src/bch.rs`
  (`decode_with_correction_sweeps_every_protected_bit`) walks every of
  the 511 protected bit positions in a frame, flips one at a time,
  decodes with correction, and verifies the recovered payload matches
  the original bit-exact for every position. Two new integration tests
  in `tests/bch_e2e.rs` round-trip a real H.261 elementary stream
  through `encode_multiframe` → single-bit corruption →
  `decode_multiframe_with_correction` → `H261Decoder` and verify
  PSNR_Y ≥ 32 dB after correction (matches the clean-channel
  baseline), and confirm a two-bit error in the same frame is left in
  `uncorrectable_frames` with the breakdown
  `corrupted_frames == corrected_frames + uncorrectable_frames`
  internally consistent. Brings the BCH module to spec-mandated
  capability: §5.4.1 explicitly labels the outer layer an "error
  correcting code" and `g(x) = (x^9 + x^4 + 1)(x^9 + x^6 + x^4 + x^3 + 1)`
  has minimum distance `d = 3`, supporting `t = (d − 1) / 2 = 1`
  correction. The `lib.rs` doc-comment "Out of scope" entry that
  previously deferred single-bit correction is removed; the new
  out-of-scope entry covers only the multi-bit (weight ≥ 2) case the
  code mathematically cannot resolve.

- **Annex D still-image transmission helpers (§D.2 + §D.3).** New
  `annex_d` module wires Annex D into the codec without disturbing the
  motion-video pipeline. `SubImageIndex` is the 0..=3 sub-image
  identifier; `still_image_tr` packs it into the 5-bit `TR` field with
  the §D.3 invariants (low 2 bits = index, top 3 bits = 0), and
  `parse_still_image_tr` enforces them on the receive path.
  `still_image_dimensions` / `still_image_chroma_dimensions` derive the
  §D.2 4× video-format sizes (QCIF ⇒ 352×288 still, CIF ⇒ 704×576).
  `subsample_still_image` implements the Figure D.1 2:1 × 2:1 transform
  (per-sub-image tile origin `0→(0,0)`, `1→(0,1)`, `2→(1,1)`,
  `3→(1,0)`) over Y/Cb/Cr planes; `reassemble_still_image` is the
  bit-exact inverse. `PictureHeader::still_image_sub_index` returns
  `Ok(None)` for ordinary motion-video pictures and surfaces the §D.3
  high-bits-must-be-zero violation when HI_RES=0 with malformed TR.
  `encoder::write_picture_header_full` is a new explicit-HI_RES variant
  of `write_picture_header`; the original 3-arg entry point is
  preserved as a thin wrapper. Integration test `tests/annex_d.rs`
  rounds-trips the picture header for every sub-image in both QCIF
  and CIF mode and rounds-trips the sub-sample/reassemble transform
  on full-size synthetic still images (luma + both chroma planes).
- **Fifth cargo-fuzz target: `parse_sdp_fmtp`** drives arbitrary
  fuzz-supplied bytes through the H.261 Session Description Protocol
  parser surface — the attribute-line parsers an endpoint runs on every
  received SDP offer or answer at session setup **before** any RTP /
  RTCP / H.261 layer sees a byte. Four entry points are exercised:
  `parse_rtpmap` (RFC 4587 §6.2: `a=rtpmap:<pt> H261/90000` with
  case-insensitive encoding-name match, payload-type and clock-rate
  integer bounds; a non-`H261` encoding name surfaces as `None`, an u8
  payload-type overflow or u32 clock-rate overflow surfaces as `None`),
  `parse_fmtp` (RFC 4587 §6.2: `a=fmtp:<pt> CIF=…;QCIF=…;D=…` with
  payload-type match), `H261FmtpParams::parse_value` (RFC 4587 §6.1.1:
  semicolon-separated key=value list with CIF/QCIF MPI ∈ 1..=4 and
  D ∈ {0,1} validation, duplicate-parameter rejection, forward-
  compatible unknown-parameter skip), and `negotiate_answer` (RFC 4587
  §6.2.1 offer/answer rules: per-shared-size `max(MPI)` upper bound,
  RFC 2032 QCIF=1 fallback for an offer with no picture size, both-
  sides-required Annex-D survival). The harness decodes the fuzz input
  as UTF-8 lossily once per iteration, drives each parser standalone,
  splits the input on `|` and feeds the `(offer, our)` halves to the
  negotiator, then runs a formatter → parser round trip on any input
  that parses cleanly so a `format_value` / `parse_value` disagreement
  trips the daily run. Contract under test: every call must *return*
  — no panic, no abort, no integer overflow (in debug / ASAN builds),
  no out-of-bounds index, no allocator OOM.
- **Seed corpus** at `fuzz/corpus/parse_sdp_fmtp/` (10 text buffers,
  ≈ 270 B total): the §6.2 worked-example `a=rtpmap:31 H261/90000` and
  `a=fmtp:31 CIF=2;QCIF=1;D=1` lines, a dynamic-payload-type rtpmap
  (PT=96), a QCIF-only fmtp, a forward-compatible fmtp with an unknown
  parameter, a lowercase-key fmtp (case-insensitive match per §6.1.1),
  a `|`-split offer/our pair for `negotiate_answer`, an empty-offer
  pair that exercises the §6.2.1 RFC 2032 QCIF=1 fallback, a malformed
  parameter list with every value out of range (`CIF=5;QCIF=0;D=2`),
  and a non-H.261 rtpmap (`H264/90000`) the parser must reject.
- **Stable-CI seed test** `tests/fuzz_seed_corpus_sdp.rs` (19 tests,
  ≈ 1 ms) mirrors the fuzz target on stable Rust so the regular CI
  matrix catches a regression in the SDP signalling parser surface
  without waiting for the daily fuzz run. Also drives empty /
  single-zero / all-ones (non-UTF-8 → U+FFFD lossy decode) /
  pseudo-random adversarial buffers, the §6.2 worked rtpmap + fmtp
  round trips, a non-H.261 rtpmap rejection, an u8 payload-type
  and u32 clock-rate overflow rejection on `parse_rtpmap`, a
  payload-type mismatch on `parse_fmtp`, MPI-out-of-range /
  Annex-D non-binary / duplicate-CIF / duplicate-QCIF / missing-`=`
  rejections on `parse_value`, a forward-compatible unknown-parameter
  skip, the §6.2.1 disjoint-advertisement `NoPictureSize` rejection,
  the §6.2.1 RFC 2032 QCIF=1 fallback, the §6.2.1 both-sides Annex-D
  rule, and a formatter → parser round-trip on the canonical
  `(CIF=4, QCIF=1, D=1)` shape.

- **Annex A IDCT accuracy conformance test** (`tests/idct_annex_a.rs`)
  implements the §A.1..§A.9 measurable conformance procedure that ITU-T
  Recommendation H.261 mandates for every compliant inverse 8×8 DCT.
  The §A.1 deterministic 32-bit LCG (`randx = randx * 1103515245 + 12345`,
  keep 30 bits LSB-cleared, divide by `2^31`, scale by `L+H+1`, truncate to
  int, subtract `L`) generates 10 000 8×8 pel blocks per dataset; the §A.2
  forward DCT and §A.4 reference IDCT run in 64-bit float directly from the
  equations in §3.2.4 (no third-party IDCT source is consulted); §A.3 rounds
  each transform coefficient to the nearest integer and clips it to
  `[-2048, +2047]` to produce the 12-bit IDCT input; our in-crate
  `idct::idct_signed` is run on the same input and clipped to `[-256, +255]`
  per §A.5; the §A.7 statistics (per-pel peak error, per-pel MSE, overall
  MSE, per-pel \|mean error\|, overall \|mean error\|) are asserted against
  the spec thresholds (≤ 1, ≤ 0.06, ≤ 0.02, ≤ 0.015, ≤ 0.0015 respectively)
  across all three §A.1 ranges `(L=256, H=255)`, `(L=H=5)`, and `(L=H=300)`,
  with the §A.9 sign-flipped rerun on each. §A.8 (all-zeros in produces
  all-zeros out) and a smoke test on the §A.1 RNG round out the eight new
  test cases. Measured margins on (L=256, H=255): peak=1 (limit ≤ 1),
  per-pel MSE=1.0e-4 (limit ≤ 0.06), overall MSE=6.0e-6 (limit ≤ 0.02),
  per-pel \|mean\|=1.0e-4 (limit ≤ 0.015), overall \|mean\|=3.0e-6 (limit
  ≤ 0.0015); (L=H=5) is bit-exact against the reference (peak=0). The
  reference f64 DCT/IDCT used as the §A.4 comparison oracle are coded
  directly from the §3.2.4 equation; no external library is consulted.

- **Fourth cargo-fuzz target: `parse_rtp_payload`** drives arbitrary
  fuzz-supplied bytes through the H.261 RTP data-path parser surface —
  the network-receive parsers an endpoint runs on every received UDP
  datagram **before** any H.261 bitstream layer sees a byte. Three
  entry points are exercised: `parse_rtp_fixed_header` (RFC 3550 §5.1
  — V / P / X / CC / M / PT / seq / ts / SSRC + 0..=15 CSRC entries,
  bounds-checked against the attacker-controlled CC field, with V != 2
  surfacing as `FieldOverflow` and an under-sized buffer surfacing as
  `ShortHeader`), `unpack_header` (RFC 4587 §4.1 — the 4-byte H.261
  RTP payload header with SBIT / EBIT / I / V / GOBN / MBAP / QUANT
  / HMVD / VMVD and the §4.1 sign-extension of the two 5-bit MV
  deltas), and `depacketize` (the multi-packet bit-walker that
  honours per-packet SBIT/EBIT alignment and asserts the recovered
  elementary stream contains at least one start code). The harness
  carves the fuzz input into up to four synthetic `H261RtpPayload`s
  with attacker-chosen header fields and attacker-chosen data
  lengths, so the slow-path bit-walker, the `pack_header`/
  `unpack_header` round-trip, and the final `iter_start_codes`
  sanity check all stay covered against attacker-controlled bytes.
  Contract under test: every call must *return* — no panic, no abort,
  no integer overflow (in debug / ASAN builds), no out-of-bounds
  index, no allocator OOM.
- **Seed corpus** at `fuzz/corpus/parse_rtp_payload/` (9 buffers,
  ≈ 257 B total): a 12-byte fixed-header-only packet (CC=0, empty
  payload), a 16-byte fixed-header + empty H.261 payload header
  boundary, a V=2 CC=0 fixed header + GOB-aligned H.261 header +
  faked PSC stream, the same shape with CC=3 and CC=15 CSRC lists, a
  non-GOB-aligned H.261 header carrying SBIT=3 / EBIT=5 / I=V=1 /
  GOBN=7 / MBAP=12 / QUANT=17 / HMVD=-7 / VMVD=11, a marker=0 mid-
  frame packet, an adversarial V=1 datagram, and an adversarial
  CC=15 buffer truncated at 12 bytes that must surface as
  `ShortHeader`.
- **Stable-CI seed test** `tests/fuzz_seed_corpus_rtp.rs` (10 tests,
  ≈ 10 ms) mirrors the fuzz target on stable Rust so the regular CI
  matrix catches a regression in the RTP payload-parser surface
  without waiting for the daily fuzz run. Also drives empty /
  single-zero / all-ones / pseudo-random adversarial buffers, the
  exact 12-byte fixed-header boundary, a CC=15 truncated input (must
  return `ShortHeader`), a V=1 rejection (must return
  `FieldOverflow`), a `pack_header` ↔ `unpack_header` round-trip on
  the typical-fields header, and a `depacketize` SBIT+EBIT-overflow
  input that exercises the `8 * data.len() - sbit - ebit` empty-
  payload branch.

- **Third cargo-fuzz target: `decode_bch_multiframe`** drives arbitrary
  fuzz-supplied bytes through the H.261 §5.4 BCH (511, 493) FEC
  multiframe parser surface (`decode_multiframe` / `parity18` /
  `syndrome18`). Covers the §5.4.4 lock-search candidate sweep (511
  bit offsets × 24 framing-bit reads at a stride of `FRAME_BITS =
  511`), the §5.4.2 GF(2) generator-polynomial long-division
  shift-register, and the §5.4.3 per-frame `Fi` / 492-data-bit /
  18-parity-bit walk against attacker-controlled bytes. The target
  also runs an **error-injection mode**: frames a deterministic
  synthetic payload via the in-crate `encode_multiframe`, then uses
  the fuzz input as a bit-flip vector to corrupt up to 16
  attacker-chosen bit positions in the framed stream before
  re-decoding — driving the non-zero-syndrome `corrupted_frames`
  branch and (when a flip lands inside the 24-bit lock window) the
  lock-loss path. Contract under test: every call must *return* — no
  panic, no abort, no integer overflow (in debug / ASAN builds), no
  out-of-bounds index, no allocator OOM.
- **Seed corpus** at `fuzz/corpus/decode_bch_multiframe/` (9 buffers,
  ≈ 12.5 KB total): one and three multiframes of pure stuffing, three
  multiframes of all-zeros / 0xC3 / pseudo-random payload, a payload-
  then-fill mix that crosses the `Fi=1`-to-`Fi=0` boundary, a single-
  bit-flipped 3-multiframe stream, a 4-bit-prefix-shifted stream that
  forces the lock-search past `bit0 = 0`, a 2-multiframe input that
  falls one frame short of the §5.4.4 lock window, and a 6-multiframe
  `0x5A` payload (twice the lock window).
- **Stable-CI seed test** `tests/fuzz_seed_corpus_bch.rs` (9 tests,
  ≈ 10 ms) mirrors the fuzz target on stable Rust so the regular CI
  matrix catches a regression in the BCH parser surface without
  waiting for the daily fuzz run. Also drives empty / single-zero /
  all-ones / pseudo-random adversarial buffers, a
  one-byte-short-of-lock input, an attacker-chosen 32-bit parity
  value, and a deterministic multi-bit injection that surfaces in
  `corrupted_frames` without breaking lock.

## [0.0.6](https://github.com/OxideAV/oxideav-h261/compare/v0.0.5...v0.0.6) - 2026-05-29

### Other

- RFC 4587 §6.2.1 offer/answer negotiation helper
- fix truncated `1?` TCOEFF prefix panic (daily-fuzz finding)
- criterion suite for transform / encode / decode hot paths
- scrub decorative external-implementation attribution
- second cargo-fuzz target for RTCP compound parser
- cargo-fuzz decoder harness + daily workflow

### Added

- **SDP offer/answer negotiation helper** (RFC 4587 §6.2.1). The
  `sdp` module gains the free function `negotiate_answer(offer,
  our_capability) -> Result<H261FmtpParams, SdpError>` that computes
  the §6.2.1 **answer** parameters from a received offer and our
  local capability:
  * **Picture-size intersection.** Only sizes both peers advertise
    survive into the answer; a disjoint pair (e.g. CIF-only offer vs
    QCIF-only capability) errors with `SdpError::NoPictureSize`,
    matching §6.2.1's "SHALL specify at least one supported picture
    size".
  * **MPI per shared size.** §6.1.1's MPI is the *minimum picture
    interval*, so `29.97 / MPI` is the **upper bound** on frame rate.
    The answer carries `MPI = max(offer.MPI, our.MPI)` per shared
    size, i.e. the more restrictive bound binds.
  * **Annex D (`D`).** §6.2.1: "This option MUST NOT appear unless
    the sender of this SDP message is able to decode this option."
    The answer's `D=1` requires both `offer.d == Some(true)` AND
    `our_capability.d == Some(true)`; otherwise `D` is omitted from
    the answer (matching §6.1.1's "SHOULD NOT be used … if not
    supported").
  * **RFC 2032 fallback.** §6.2.1: "If the receiver does not specify
    the picture size/MPI parameter … assume that such a receiver is
    able to support reception of QCIF resolution with MPI=1." The
    helper applies that fallback automatically (equivalent to
    `H261FmtpParams::rfc2032_fallback()`) when the offer carries no
    picture-size parameter. The fallback is **not** applied to
    `our_capability` — that side is local and should be supplied
    explicitly.

  The companion method `H261FmtpParams::preferred_picture_size()`
  returns the preferred receiver mode per §6.2.1 ("Parameters offered
  first are the most preferred") — `Some(SourceFormat::Cif)` when CIF
  is advertised (matching `format_value`'s CIF-before-QCIF emission
  order from the §6.2.1 worked example), `Some(SourceFormat::Qcif)`
  when only QCIF is, else `None`. Eight new tests cover the
  intersection / MPI-max / disjoint-sizes / Annex-D / RFC-2032-
  fallback / format round-trip / max-frame-rate / validate-passes
  paths; the negotiation example also runs as a doctest.

### Fixed

- **Decoder panic on a truncated `1?` TCOEFF prefix** (round 175,
  surfaced by the scheduled daily `decode_h261` fuzz harness).
  `decode_tcoeff(.., is_first = false)` saw a bit-reader where exactly
  one bit remained and that bit was `1`. The function took the
  `b0 == 1` branch and then peeked two bits from
  `peek >> (avail - 2)`, where `avail = 1` caused an unsigned
  underflow → `attempt to subtract with overflow` panic under debug
  / ASAN builds. The two-bit peek is now gated behind `avail >= 2`
  and the call returns `Error::invalid("h261 tcoeff: truncated `1?`
  prefix")` on the malformed input, restoring the public-surface
  contract from the fuzz harness: every call returns — no panic, no
  abort, no out-of-bounds. New regression test
  `tcoeff_truncated_one_bit_does_not_panic` covers it on stable Rust.

### Added

- **Criterion benchmark suite** (`benches/transform`, `benches/encode`,
  `benches/decode`). Round 175 (depth-mode) wires up `criterion = "0.5"`
  as a dev-dependency and registers three `harness = false` bench
  binaries so future optimisation rounds have a recorded baseline to
  A/B against:
  * `transform` times the 8×8 inverse / forward DCT block hot path —
    `fdct_intra` + `fdct_signed` (encoder forward pass) and
    `idct_intra` + `idct_signed` (decoder inverse pass). One block per
    iteration; throughput reported in samples so per-sample cycle-
    equivalents land naturally.
  * `encode` times whole-picture encode through the production
    `encode_intra_picture` / `H261Encoder::encode_frame` paths in four
    scenarios: QCIF intra-only (no ME), single-P from a pre-built I
    reference, I + 3 P chain (full rate-controller carryover), and CIF
    intra (the 4× area test).
  * `decode` times whole-picture decode through `H261Decoder::send_packet`
    + `receive_frame`, mirroring the encode scenarios. Each decode
    bench runs the in-crate encoder once during setup to produce a
    real elementary stream, so the timed loop measures the decoder
    alone.
  Every benchmark synthesises its YUV source inline from a
  deterministic striped pattern plus low-amplitude xorshift noise —
  no on-disk fixtures, no third-party CLI, no `docs/` files read at
  bench time. `cargo bench -p oxideav-h261 --no-run` doubles as a
  compile-only CI regression guard via the existing matrix.

- **Second `cargo-fuzz` target — RTCP compound parser.** New
  `parse_rtcp_compound` fuzz target drives arbitrary fuzz-supplied bytes
  through the public RTCP parser surface (`parse_compound`,
  `parse_report`, `parse_sdes`, `parse_bye`, `parse_app`) so the §6.1
  compound walk (16-bit-length advance), the SR/RR fixed header + RC
  block walk, SDES chunk + item walk (including the PRIV inner 8-bit
  length), BYE reason-string length-prefix, and APP `name`/`data` 32-bit
  alignment are all exercised against bytes whose shape the fuzzer
  dictates. Same contract as the existing `decode_h261` target: every
  call must return — no panic, no abort, no integer overflow (in debug
  / ASAN builds), no out-of-bounds index, no allocator OOM. The seed
  corpus under `fuzz/corpus/parse_rtcp_compound/` contains nine valid
  datagrams (empty RR, SR with no blocks, SR with one block, RR with
  two blocks, SDES CNAME, BYE with reason, APP with PING payload, and
  two compound packets). `tests/fuzz_seed_corpus_rtcp.rs` drives the
  same logic on stable Rust against the corpus plus several adversarial
  in-line buffers (lying header length, zero-length advance, truncated
  compound, SDES PRIV length overflow, BYE reason overflow, APP at the
  5-bit subtype maximum, unknown PT=205) so a regression in the public
  parser surface trips an existing CI lane rather than waiting for the
  daily fuzz run.

## [0.0.5](https://github.com/OxideAV/oxideav-h261/compare/v0.0.4...v0.0.5) - 2026-05-24

### Other

- RFC 4587 §6.1.1/§6.2 video/H261 rtpmap + fmtp parameter mapping
- RFC 3550 §6.7 Application-Defined (APP, PT=204) packet
- RFC 3550 §6.5 SDES + §6.6 BYE + §6.1 compound packets
- RFC 3550 §6.4 Sender/Receiver Report builders + RtpPacketizer counters
- encoder-side RFC 3550 packetiser stamps RFC 4587 RTP packets
- implement RFC 4587 H.261 RTP payload-format wrap/unwrap
- implement §5.2 + Annex B HRD buffer model and §5.4.2 spec test
- implement BCH (511,493) forward error correction framing (§5.4)

### Added

- **Daily `cargo-fuzz` decoder harness.** New `fuzz/` subcrate with a
  single `decode_h261` target that drives arbitrary fuzz-supplied
  bytes through the decoder's full public surface (`send_packet` →
  drain `receive_frame` → `flush` → drain) so the PSC / GBSC scanners,
  every VLC table (MBA / MTYPE / MVD / CBP / TCOEFF + 20-bit escape),
  the §4.2.3.4 MV predictor, the integer-pel MC indexing, the §3.2.3
  loop filter, and the 8×8 IDCT are all exercised against attacker-
  controlled bytes. The contract under test is purely that every call
  *returns* — no panic, no abort, no integer overflow, no out-of-bounds
  index, no allocator OOM. Seed corpus is encoder-derived: QCIF + CIF
  I-pictures across `q ∈ {8, 12, 31}`, plus QCIF + CIF I+P pairs that
  exercise motion compensation and the loop filter. A new
  `tests/fuzz_seed_corpus.rs` test drives the same logic on stable
  Rust against the same corpus so the regular CI matrix catches a
  regression in the public decoder surface without waiting for the
  daily fuzz run. Workflow `.github/workflows/fuzz.yml` runs the
  target once a day via the org-shared `crate-fuzz.yml` reusable
  workflow (30-minute budget).

- **SDP media-type / `rtpmap` / `fmtp` parameter mapping (`oxideav_h261::sdp`).**
  New module implementing the RFC 4587 §6.1.1 `video/H261` media-type
  registration and its §6.2 SDP mapping. The `a=rtpmap` line is fixed —
  encoding name `H261`, clock rate `90000`, `m=` media name `video` (pinned by
  `ENCODING_NAME` / `CLOCK_RATE` / `MEDIA_NAME`); `format_rtpmap(pt)` emits it
  and `parse_rtpmap` reads it back, confirming the encoding name is H.261
  (case-insensitively), tolerating the optional trailing channel field, and
  rejecting other codecs / non-rtpmap lines. `H261FmtpParams { cif, qcif, d }`
  models the three §6.1.1 optional `a=fmtp` parameters: `CIF` / `QCIF` carry an
  MPI integer 1..=4 ("max rate `29.97 / value` fps"), `D` signals Annex D
  still-image support (`1`/`0`). `format_value` / `format_fmtp` emit
  `CIF=2;QCIF=1;D=1` (CIF before QCIF, the §6.2.1 example order; no line when
  no parameters are set, per §6.2 "if any"); `parse_value` / `parse_fmtp`
  reverse it, enforcing the 1..=4 MPI range (`MpiOutOfRange`) and `D ∈ {0,1}`
  (`BadAnnexD`), rejecting non-integer values / malformed tokens / duplicate
  picture-size params, tolerating whitespace, matching parameter names
  case-insensitively, and skipping unknown parameters forward-compatibly. The
  §6.2.1 offer/answer helpers: `validate` enforces "SHALL specify at least one
  supported picture size" (`NoPictureSize`), `rfc2032_fallback` returns the
  §6.2.1 default (QCIF MPI=1) for a peer that omits picture-size params, and
  `max_frame_rate(fmt)` returns the exact `29.97 / MPI` bound as an integer
  rational (`(2997, 100 * MPI)`) so the §6.2.1 "≤ 15 fps for CIF=2" bound is
  computed without floating-point round-off. The SDP offer/answer state machine
  and the rest of the session description (`v=` / `o=` / `c=` / `t=`) remain
  caller-side — this module owns only the H.261-specific `rtpmap` / `fmtp` wire
  format. The RFC 2032 H.261-specific RTCP control packets (FIR / NACK) are
  deliberately not implemented: RFC 4587 §7.1 mandates new implementations
  SHALL ignore them and SHALL NOT use them. 36 new unit tests cover the
  spec-example round trip, both line builders/parsers (with and without the
  `a=` prefix), the full 1..=4 MPI range, all six error variants, the
  forward-compatible unknown-parameter skip, case-insensitive name matching,
  the RFC-2032 fallback, the frame-rate rational, and a full two-line session-
  description round trip.

- **RTCP APP (Application-Defined) packet (`oxideav_h261::rtcp`).** Builder and
  parser for RFC 3550 §6.7 (PT = 204). `build_app(subtype, ssrc, name, data)`
  emits the standard 4-byte RTCP header (with the 5-bit RC slot reused as the
  §6.7 subtype) + SSRC + 4-octet ASCII name + application-dependent data;
  `parse_app` reverses it. The builder enforces three §6.7 invariants —
  `subtype` ≤ 31 (5-bit field, `AppSubtypeOutOfRange`), `name` exactly 4
  octets (`AppNameWrongLength`), and `data.len() % 4 == 0`
  (`AppDataNotAligned`, "must be a multiple of 32 bits long"). The parser
  rejects truncated buffers, V != 2, PT != 204, and length-field-smaller-than-
  mandatory-header. APP packets now round-trip through `parse_compound` as a
  typed `RtcpPacket::App(AppPacket)` variant rather than falling into the
  catch-all `Other`; unknown PTs (e.g. RFC 4585 RTPFB = 205) still surface as
  `Other`. §6.7 mandates names be case-sensitive ("uppercase and lowercase
  characters treated as distinct"), so the parser surfaces the four bytes
  verbatim without case folding. The §6.2 transmission-interval scheduler and
  the §A.1 / §A.3 / §A.8 loss-fraction / jitter estimators remain caller-side
  (out of scope for the codec). 14 new unit tests cover the header layout for
  empty + data-bearing packets, subtype-0 / subtype-31 boundaries, 1024-byte
  payload round-trip, byte-exact (non-case-folded) name preservation,
  short-header / bad-version / wrong-PT / truncated-by-length / past-stated-
  length rejection paths, all three builder validation errors (subtype-32,
  name-of-every-non-4-length, data-of-every-non-aligned-length 1..=9), and a
  compound RR + SDES + APP round-trip that pulls the App variant back typed.
  The pre-existing "compound_preserves_unknown_app_packet" test was updated to
  use PT = 205 (a stand-in for an RFC 4585 RTPFB packet this module doesn't
  model) since APP is no longer "unknown."

- **RTCP SDES + BYE + compound packets (`oxideav_h261::rtcp`).** Rounds out
  the control channel beyond SR/RR (RFC 3550 §6.5 / §6.6 / §6.1).
  `build_sdes` / `parse_sdes` (PT=202, §6.5) handle Source Description
  packets: 0..=31 chunks, each binding an SSRC/CSRC to a list of `SdesItem`s
  — `Cname` (§6.5.1, mandatory), `Name`, `Email`, `Phone`, `Loc`, `Tool`,
  `Note`, and `Priv` (§6.5.8 prefix/value) — independently 32-bit-aligned
  with a trailing END (item-type-0) byte and null padding. `build_cname_sdes`
  is the one-call helper for the minimal "SSRC → CNAME" chunk §6.1 requires
  in every compound packet. `build_bye` / `parse_bye` (PT=203, §6.6) carry
  0..=31 leaving SSRC/CSRC identifiers plus an optional 8-bit-length-prefixed,
  null-padded free-text reason. `compound` concatenates pre-built sub-packets
  into one datagram body; `parse_compound` walks a received datagram back into
  typed `RtcpPacket`s (`Report` / `Sdes` / `Bye` / `Other` for unmodelled PTs
  such as APP=204), advancing via each sub-packet's self-delimiting `length`
  field. Item text / reason strings are validated against the 255-octet 8-bit
  length limit (`TextTooLong` / `PrivTooLong`); the SC field is capped at 31
  (`TooManySources`); parsers decode text UTF-8-lossily so a malformed
  datagram never panics, skip unknown SDES item types forward-compatibly, and
  reject truncated / wrong-PT / wrong-version input. Scheduling (§6.2) and the
  §A.1/§A.3/§A.8 loss/jitter estimators remain caller-side (out of scope for
  the codec). 19 new unit tests cover header/alignment, all-item-type and
  multi-chunk round-trips, max-length and empty-chunk edges, the 31/255 caps,
  unknown-item skipping, and three compound-packet round-trips (RR+SDES+BYE,
  SR-with-block+SDES, RR+unmodelled-APP) plus truncation rejection.

- **RTCP Sender / Receiver Report builders (`oxideav_h261::rtcp`).** The
  control-channel companions to the RTP data path (RFC 3550 §6.4).
  `build_sender_report` (PT=200, §6.4.1) emits the 8-byte RTCP header +
  20-byte sender-info section (NTP + RTP timestamps, sender's packet &
  octet counts) + 0..=31 reception report blocks; `build_receiver_report`
  (PT=201, §6.4.2) is the same minus the sender-info section, with an
  empty RR (RC=0) as the canonical "nothing to report" packet.
  `ReceptionReportBlock` (24 bytes: SSRC, 8-bit fraction lost, 24-bit
  two's-complement cumulative lost, extended highest sequence number,
  jitter, LSR, DLSR) and `SenderInfo` round-trip through `parse_report`,
  which validates V=2, the SR/RR PT, and the §6.4.1 `length` field
  (32-bit words minus one). `RtpPacketizer` now tracks the session's
  running packet/octet counts and the last frame's RTP timestamp, exposed
  via `packet_count()` / `octet_count()` / `sender_info()` and a
  `sender_report()` convenience that drops a conformant SR straight out of
  the packetiser state. Scheduling (§6.2), SDES/CNAME/BYE, and the
  §A.1/§A.3/§A.8 loss/jitter estimators remain caller-side (out of scope
  for the codec). Wired through end-to-end tests that encode QCIF
  I-pictures, packetize them, build an SR from the packetiser counters,
  and round-trip both SR and RR through the parser.

- **Encoder-side RTP packetiser (`RtpPacketizer`).** Higher-level glue
  between `H261Encoder` and the RTP wire format. Construct with
  `RtpPacketizer::new(payload_type, ssrc, initial_sequence_number,
  max_rtp_packet_size)`; call `pack_frame(frame_bytes,
  rtp_timestamp_90khz)` once per coded picture. Returns a sequence of
  `RtpPacket`s whose `bytes` field is a complete RFC 3550 §5.1 fixed
  header (V=2, P=0, X=0, CC=0, M, PT, seq, ts, SSRC) followed by the
  RFC 4587 §4.1 4-byte H.261 header and the GOB-aligned payload slice.
  The marker bit is set on the LAST packet of each frame per RFC 4587
  §4.1 ("MUST be set to one in the last packet of a video frame;
  otherwise, it MUST be zero"); sequence numbers auto-advance mod
  2^16 across frames; the same RTP timestamp is stamped on every
  packet of one frame (§4.1). The 7-bit payload type is masked
  internally so callers passing a `u8` with the high bit set don't
  corrupt the M bit. `parse_rtp_fixed_header` parses RFC 3550 §5.1
  headers (including any CSRC list) for the receiver side. Wired
  through an end-to-end test that drives `H261Encoder.encode_frame()`
  for an I + P pair, packets them, parses the RTP fixed headers,
  reuses `depacketize` on the inner payloads, and decodes the result
  back into video frames.

- **RTP payload format (RFC 4587).** New `oxideav_h261::rtp` module
  implements the H.261 RTP payload-format §4.1 4-byte header (SBIT,
  EBIT, I, V, GOBN, MBAP, QUANT, HMVD, VMVD) with bit-exact
  `pack_header` / `unpack_header`, plus the GOB-aligned cheap
  packetizer (`packetize_gob_aligned`) and `depacketize` reassembler
  from §4.2. The packetizer splits at byte-aligned PSC / GBSC
  boundaries, fragments oversized GOBs at byte boundaries (SBIT/EBIT
  stay zero), and sets the RTP marker-bit hint on the last payload of
  each frame. Round-trips are byte-exact against `encode_intra_picture`
  output and the recovered stream still decodes through the regular
  `H261Decoder`. RFC 4587 §4.1's explicit "no BCH on the RTP path" rule
  is documented in the module's intro; the `bch` and `rtp` modules are
  mutually exclusive consumers of an elementary stream. `pack_header`
  enforces the `-16` MVD prohibition (5-bit field `'10000'` is
  forbidden by §4.1).

- **Hypothetical Reference Decoder buffer model (§5.2 + Annex B).** New
  `oxideav_h261::hrd` module exposes the §5.2 per-picture cap
  (`64 kbits` QCIF, `256 kbits` CIF, excluding §5.4 FEC framing) and
  the Annex B buffer-occupancy walk. `HrdParams::new(R_max)` derives
  `B = 4 * R_max / 29.97` and the receiver buffer size `B + 256 kbits`
  via integer-rational arithmetic so long sequences don't drift on
  floating-point round-off. `walk_buffer(pictures, N, params)` returns
  the post-removal occupancy after every picture and the first underflow
  index (if any); `check_overflow(pictures, N, params)` flags the dual
  pre-removal-overflow case. The HRD is a coder-side compliance check
  only — no on-wire changes.

- **Spec §5.4.2 worked-example regression test for `bch::parity18`.**
  The ITU-T H.261 (03/93) spec publishes a single validation vector
  for the BCH parity routine — for the 493-bit input `0` followed by
  492 ones, the parity is exactly `011011010100011011`₂ = `0x1B51B`.
  The new `parity_matches_spec_5_4_2_worked_example` test feeds that
  input through `parity18` and asserts equality with the spec value,
  pinning the implementation to the spec's own published test data.

### Tests added

- `hrd::tests::*` (12 unit tests in `src/hrd.rs`):
  - Per-picture cap returns 64 / 256 kbits for QCIF / CIF.
  - `check_picture_cap` returns `Ok` at-or-below the cap, `Overflow`
    above it with both `actual_bits` and `cap_bits` populated.
  - `HrdParams::new` derives `B` correctly at 64 kbit/s and 2 Mbit/s
    channel rates (integer-rational truncation matches the spec
    fraction `4 * R * 10000 / 299700`).
  - `walk_buffer` at matched rate ⇒ buffer drains to exactly 0;
    smaller pictures accumulate monotonically; oversized first picture
    triggers `first_underflow = Some(0)`; skip factor `N = 2` doubles
    per-interval arrival as expected.
  - `check_overflow` is silent under normal drain, trips at the
    correct frame index when pictures are tiny relative to arrival.
- `tests/hrd_e2e.rs` (4 integration tests):
  - Real QCIF I-pictures at quant=8 / quant=2 fit the §5.2 cap.
  - 10 real I-pictures at quant=12 fail the HRD walk at N=1 / 29.97 fps
    over 64 kbit/s (each picture far larger than per-interval arrival)
    but succeed at N=10 (≈3 fps); confirms the HRD correctly identifies
    both regimes.
  - 30 real I-pictures matched against a 64 kbit/s / N=4 channel pass
    the overflow check (matched-rate drain).
- `bch::tests::parity_matches_spec_5_4_2_worked_example` (1 new unit
  test) — verifies `parity18` against the §5.4.2 worked example.

### Changed

- `lib.rs` module-docstring scope: HRD §5.2 + Annex B added to
  in-scope items.
- README feature matrix: `HRD buffer model (§5.2 + Annex B)` row
  added (`yes / yes`).

- **BCH (511, 493) forward error correction framing (§5.4).** New
  `oxideav_h261::bch` module wraps and unwraps the outer multiframe
  FEC layer H.261 prescribes for noisy `p × 64 kbit/s` channels. The
  module computes the 18-bit BCH parity over the 493-bit `Fi || data`
  field via the spec generator polynomial
  `g(x) = (x^9 + x^4 + 1)(x^9 + x^6 + x^4 + x^3 + 1)
        = x^18 + x^15 + x^12 + x^10 + x^8 + x^7 + x^6 + x^3 + 1`
  (`0x495C9` in 19-bit form), assembles 8-frame multiframes carrying
  the alignment pattern `S1..S8 = 0 0 0 1 1 0 1 1`, and surfaces the
  per-frame BCH syndrome as a corruption diagnostic.

  - `parity18(data: &[u8]) -> u32` — long-division shift-register
    implementation, 19-bit register XORed with `GEN_POLY` whenever
    the bit-18 sentinel is set.
  - `syndrome18(data, parity) -> u32` — zero means the codeword
    matches `g(x)`, non-zero means at least one bit error.
  - `encode_multiframe(coded, bits)` — packs an arbitrary
    inner-bitstream payload into 512-byte multiframes, emitting
    Fi=0 stuffing frames to round up to a multiframe boundary.
  - `decode_multiframe(framed)` — requires 3 consecutive complete
    alignment patterns (24 framing bits ≡ 3 multiframes) for lock
    per §5.4.4; reports `corrupted_frames` (non-zero-syndrome count),
    `fill_frames` (Fi=0 frames skipped), and the recovered inner data.

  The BCH layer is transport-level — neither the public `H261Decoder`
  nor the encoder change shape. Callers that need framed output for
  a raw bit-serial link wrap their bytes; callers receiving a framed
  stream (e.g. RFC 4587 §6.2 historical deployments) recover the
  inner stream.

### Tests added

- `bch::tests::*` (12 unit tests in `src/bch.rs`):
  - Generator polynomial factors `(0x211)*(0x259) == 0x495C9` over
    GF(2).
  - All-zero input ⇒ zero parity.
  - All-ones 493-bit input round-trips through `parity18` /
    `syndrome18` with zero residue.
  - Single-bit flip in either data or parity is detected by the
    syndrome.
  - Round-trip across 1 / 3 / 6 whole multiframes plus a
    < 1-multiframe payload that exercises the fill-frame path.
  - Data corruption surfaces as `corrupted_frames >= 1` without
    breaking lock.
  - Lock acquired when the framed stream is preceded by 4 junk bits.
  - All-ones noise input fails to obtain frame lock (no false
    positive on alignment).
- `tests/bch_e2e.rs` (3 integration tests):
  - End-to-end QCIF I-picture encode → BCH wrap → BCH unwrap → H.261
    decode round-trip, PSNR ≥ 32 dB.
  - Single-bit error in the FEC payload is flagged via syndrome but
    data is still passed through.
  - Two concatenated pictures BCH-wrapped separately survive the
    unwrap intact.

### Changed

- `lib.rs` module-docstring "Out of scope" entry for BCH §5.4 replaced
  with the in-scope description (single-bit *correction* of corrupted
  codewords is the only remaining out-of-scope item).
- README feature matrix: `BCH forward error correction (§5.4)` row
  flipped from `no / no` to `yes / yes`.

## [0.0.4](https://github.com/OxideAV/oxideav-h261/compare/v0.0.3...v0.0.4) - 2026-05-06

### Other

- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- wire Encoder trait + registry, spiral+diamond ME
- registry calls: rename make_decoder/make_encoder → first_decoder/first_encoder
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-h261/pull/502))

### Added

- **Encoder `Encoder` trait implementation and registry wiring.**
  `H261RegistryEncoder` wraps `H261Encoder` and implements the
  `oxideav_core::Encoder` trait (`send_frame` / `receive_packet` / `flush`).
  `make_encoder(params)` is now registered in `register_codecs` via
  `.encoder(encoder::make_encoder)` so callers can use
  `reg.first_encoder(&params)` with H.261.
  The factory accepts `params.width` / `params.height` (QCIF or CIF) and
  derives an initial GQUANT from `params.bit_rate` via an empirical
  bits-per-frame-at-Q1 model. The per-GOB MQUANT rate controller then
  nudges ±1 step dynamically. At 64 kbit/s QCIF the factory selects
  QUANT ≈ 9, delivering ≥ 45 dB PSNR_Y on smooth gradient content.

- **Spiral + diamond motion estimation** replaces the prior flat full-window
  scan. The new search evaluates concentric ring boundaries innermost-first
  with early termination when two consecutive rings show no improvement,
  then refines with an 8-connected neighbourhood around the winner. On
  static/smooth content this saves evaluating the outer rings (~80 % of the
  961-point full scan) while maintaining quality; on complex motion it falls
  back to the full ±15 range. A compact 3-tap diamond fallback catches any
  one-pel miss at ring boundaries.

### Changed

- Updated `lib.rs` module docstring — the crate is no longer decode-only.
- README feature matrix updated: MC encode, loop-filter-with-RDO encode,
  per-GOB rate control, and registry encoder rows added.

### Tests added

- `register_tests::register_via_runtime_context_installs_encoder_factory`
- `register_tests::encoder_factory_qcif_defaults`
- `register_tests::encoder_factory_cif`
- `register_tests::encoder_factory_rejects_bad_dimensions`
- `encoder::tests::make_encoder_derives_quant_from_bit_rate` — encodes 8 QCIF
  frames via the `Encoder` trait at 64 kbit/s target; asserts avg PSNR_Y ≥ 35 dB.
- `ffmpeg_roundtrip::registry_encoder_qcif_roundtrip` — encodes 4 QCIF frames
  via the `Encoder` trait, feeds the stream to ffmpeg, asserts clean decode.
- `ffmpeg_roundtrip::encoder_psnr_vs_source_at_default_quant` — encodes 8
  QCIF frames of a moving gradient and verifies avg PSNR_Y ≥ 32 dB after
  ffmpeg cross-decode.

## [0.0.3](https://github.com/OxideAV/oxideav-h261/compare/v0.0.2...v0.0.3) - 2026-05-03

### Other

- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- oxideav-core ^0.2 -> ^0.1 (0.2.0 was yanked)
- implement receive_arena_frame() for true zero-copy
- wire DoS framework (DecoderLimits + ArenaPool + arena Frame)
- adopt slim VideoFrame/AudioFrame shape
- investigate r5/r6 long-clip drift, prove no IDCT precision loss
- fix chained-P decoder bug at GOB end, add MQUANT-delta rate ctrl
- add FIL (loop filter) MTYPEs to P-picture mode decision
- integer-pel motion estimation + MC for P-pictures
- add P-picture (INTER, no MC) Baseline encoder
- pin release-plz to patch-only bumps

### Added

- **`receive_arena_frame()` — zero-copy decode path.**
  Overrides the new `oxideav_core::Decoder::receive_arena_frame()`
  method (added in oxideav-core 0.2.0) to return an arena-backed
  `oxideav_core::arena::sync::Frame` directly, skipping the per-plane
  memcpy that the legacy `receive_frame() -> Frame::Video(VideoFrame)`
  path requires for `Send`. Internal queueing was reorganised so the
  arena lease happens at drain time rather than decode time:
  decoded `Picture`s are queued raw and converted to either a
  heap-backed `VideoFrame` (legacy path) or an arena `Frame` (new
  path) on demand. This keeps the pool short-lived — pool
  exhaustion in the typical send-many-then-drain pattern is no
  longer possible because the pool only holds slots for the
  duration of frames the caller has explicitly drained via
  `receive_arena_frame`.

### Changed

- **Bumped `oxideav-core` dep from `0.1` to `0.2`** to pick up the
  new `Decoder::receive_arena_frame` trait method (additive; default
  impl preserves backwards compatibility for every other
  `oxideav-h261` consumer).

- **DoS-protection wiring (oxideav-core 0.1.8 framework).**
  `H261Decoder` now honours [`oxideav_core::DecoderLimits`] at two
  layers, sub-task #85's second proof-of-concept after the h263 port.
  * **Construction.** `make_decoder(params)` reads `params.limits()` and
    forwards to the new `H261Decoder::with_limits(codec_id, limits)`
    constructor; `H261Decoder::new(codec_id)` is a thin wrapper that
    uses `DecoderLimits::default()` (32 k × 32 k pixels, 1 GiB / arena,
    8 arenas in flight). `with_limits` builds an
    `Arc<oxideav_core::arena::ArenaPool>` sized at
    `limits.max_arenas_in_flight` slots × `min(limits.max_alloc_bytes_per_frame,
    160 KiB)` per arena (the 160 KiB cap is the H.261 worst-case CIF I420
    frame plus alignment headroom — no real H.261 picture allocates more
    than this regardless of the caller's `max_alloc_bytes_per_frame`).
    Per-arena alloc-count cap is `limits.max_alloc_count_per_frame`
    (default 1M).
  * **Header-parse cap.** `decode_one_picture` checks
    `(width × height) <= limits.max_pixels_per_frame` immediately after
    `parse_picture_header` returns the QCIF / CIF dimensions and surfaces
    a tighter-than-format mismatch as `Error::InvalidData` (NOT
    `ResourceExhausted` — H.261's source format is fixed by a single
    PTYPE bit, so the bitstream cannot declare an arbitrary size; a
    failure here means "this codec's intrinsic frame size doesn't fit in
    the caller's caps").
  * **Arena pool.** Each picture decode leases one arena from the pool,
    builds an `oxideav_core::arena::Frame` (refcounted handle to the
    leased buffer + per-plane offset/length pairs + a `FrameHeader`),
    materialises a `VideoFrame` from it for the public `Frame::Video`
    enum, then drops the arena handle to return its buffer to the pool.
    Pool exhaustion (every slot checked out) surfaces as
    `Error::ResourceExhausted` from the lease call, which propagates up
    through `decode_one_picture` and out of `send_packet` / `flush`.
  * **Send-boundary trade-off.** The `oxideav_core::Decoder` trait
    requires `Send`, but the new `oxideav_core::arena::Frame` is
    `Rc<FrameInner>` and therefore `!Send`. The `Frame::Video(VideoFrame)`
    enum returned by `Decoder::receive_frame` stays heap-backed so
    downstream consumers (oxideav-pipeline, oxideplay, etc.) keep the
    same shape — the arena `Frame` is a transient internal value that
    backs each picture's allocation. When the workspace gains an
    `Arc<FrameInner>` parallel-decoder variant the public API can be
    migrated without disturbing this crate's pool wiring (the wiring
    lives entirely inside `decode_one_picture`).
  * **Public test surface.** `H261Decoder` exposes `limits()` and
    `arena_pool() -> &Arc<ArenaPool>` for diagnostics and pool-aware
    tests. Five new tests in `tests/dos_limits.rs`:
    * `picture_header_too_large_returns_invalid_data` — QCIF header
      against a 99×99 pixel cap → `InvalidData` mentioning
      `max_pixels_per_frame`.
    * `picture_header_within_cap_decodes_normally` — same QCIF header
      against a 1024×1024 cap doesn't trip the dimension check.
    * `pool_exhaustion_returns_resource_exhausted` — pool sized at 2,
      third concurrent lease → `ResourceExhausted`.
    * `default_limits_admit_qcif_and_cif` — sanity that the default
      32 k × 32 k cap admits CIF.
    * `pool_buffer_returns_after_decode` — pool sized at 1, lease/drop/
      re-lease cycle works.
  * Encoder is unchanged (no DoS surface — it consumes caller-owned
    `VideoFrame`s and produces compressed packets).
- Encoder: P-picture (INTER) support with integer-pel motion compensation
  (full-window ±15 SAD search per H.261 §3.2.2 / Annex A) and the
  `Inter+MC+FIL` MTYPEs (loop filter §3.2.3, separable 1/4-1/2-1/4 with
  edge-pel passthrough). Each P-MB picks the cheapest of skip / Inter /
  Inter+MC{-only,+CBP} / Inter+MC+FIL{-only,+CBP} via a bit-cost estimator
  comparing MTYPE + MVD + CBP + a residual proxy. The FIL stream
  passes the third-party-decoder roundtrip test
  (`ffmpeg_decodes_our_fil_p_pictures`). On testsrc QCIF the pipeline
  lifts the third-party-decoder roundtrip PSNR from r12's 39.27 dB /
  8680 B to 39.40 dB / 8546 B (–1.5 % bytes, +0.13 dB).
- Encoder: per-GOB MQUANT-delta rate controller (§4.2.3.3). Tracks
  cumulative bits within each GOB and nudges the quantiser ±1 step (within
  a ±6 window around GQUANT) when an MB lands far over the linear bit
  budget. Honoured only on MQUANT-bearing MTYPEs (Intra+MQUANT,
  Inter+MQUANT, InterMc+CBP+MQUANT, InterMcFil+CBP+MQUANT); other modes
  defer the change. Disabled by `OXIDEAV_H261_NO_MQUANT=1` for A/B
  benchmarks. Trims 0.3 % bytes on the testsrc fixture at –0.03 dB
  (8517 B / 39.37 dB vs r13's 8546 B / 39.40 dB).

### Fixed

- Decoder: chained P-frame mishandling at GOB 5 MBA=33 (last MB of QCIF
  GOB 5) on streams where the picture's last MB used MC-only mode. The
  GOB MB-loop in `decoder::decode_picture_body` and `mb::decode_mba_diff`
  used to break early on `bits_remaining < 16`, but the start-code prefix
  is itself ≥16 zero bits — fewer than that cannot encode a start code,
  so the remaining bits are valid MB data + padding. Now we only invoke
  the start-code peek when ≥16 bits remain and otherwise let the VLC
  decoder consume what's there. Self-decode of a 5-frame P-chain on
  testsrc QCIF jumps from 25–28 dB PSNR (drift from misdecoded final
  MBs) to a clean 36–37 dB; the third-party-decoder roundtrip is now
  byte-tight on the affected GOB.

## [0.0.2](https://github.com/OxideAV/oxideav-h261/compare/v0.0.1...v0.0.2) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- h261 tests: ffmpeg PSNR conformance suite (QCIF/CIF, I+P, qscale sweep)
- h261 encoder: drop unused is_intra_first arg from emit_runlevel
- h261 encoder: add ffmpeg roundtrip integration test
- h261 encoder: QCIF/CIF I-picture foundation (DCT + quant + VLC + layers)
