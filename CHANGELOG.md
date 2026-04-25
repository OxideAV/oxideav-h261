# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
