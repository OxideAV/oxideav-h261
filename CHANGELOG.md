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

## [0.0.2](https://github.com/OxideAV/oxideav-h261/compare/v0.0.1...v0.0.2) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- h261 tests: ffmpeg PSNR conformance suite (QCIF/CIF, I+P, qscale sweep)
- h261 encoder: drop unused is_intra_first arg from emit_runlevel
- h261 encoder: add ffmpeg roundtrip integration test
- h261 encoder: QCIF/CIF I-picture foundation (DCT + quant + VLC + layers)
