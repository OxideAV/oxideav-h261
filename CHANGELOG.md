# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
  comparing MTYPE + MVD + CBP + a residual proxy. ffmpeg interop holds on
  the FIL stream (`ffmpeg_decodes_our_fil_p_pictures`). On testsrc QCIF
  the pipeline lifts ffmpeg-roundtrip PSNR from r12's 39.27 dB / 8680 B
  to 39.40 dB / 8546 B (–1.5 % bytes, +0.13 dB).
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
  MBs) to a clean 36–37 dB matching ffmpeg byte-for-byte.

## [0.0.2](https://github.com/OxideAV/oxideav-h261/compare/v0.0.1...v0.0.2) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- h261 tests: ffmpeg PSNR conformance suite (QCIF/CIF, I+P, qscale sweep)
- h261 encoder: drop unused is_intra_first arg from emit_runlevel
- h261 encoder: add ffmpeg roundtrip integration test
- h261 encoder: QCIF/CIF I-picture foundation (DCT + quant + VLC + layers)
