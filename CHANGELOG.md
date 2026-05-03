# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
