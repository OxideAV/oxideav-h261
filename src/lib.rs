//! Pure-Rust ITU-T H.261 video codec (decoder + encoder).
//!
//! Scope (ITU-T Rec. H.261 03/93):
//! * Picture header — PSC (20 bits: `0000 0000 0000 0001 0000`), TR (5),
//!   PTYPE (6) with `source format` bit (0 = QCIF, 1 = CIF), PEI/PSPARE loop.
//! * GOB header — GBSC (16 bits), GN (4), GQUANT (5), GEI/GSPARE loop.
//! * Macroblock layer — MBA VLC (Table 1), MTYPE VLC (Table 2), MQUANT (5),
//!   MVD VLC (Table 3), CBP VLC (Table 4).
//! * Block layer — TCOEFF VLC (Table 5) + EOB + 20-bit `(escape | run | level)`
//!   escape, zigzag scan per Figure 12, dequantisation per §4.2.4, 8x8 IDCT,
//!   INTRA DC as 8-bit FLC per Table 6.
//! * INTRA / INTER / INTER+MC / INTER+MC+FIL macroblock types.
//! * Integer-pel motion compensation (range +-15 pels). Chroma MVs are the
//!   luma MV halved then truncated toward zero (§3.2.2).
//! * Loop filter (FIL bit of MTYPE) — separable 1/4, 1/2, 1/4 filter, §3.2.3.
//! * Output is YUV 4:2:0 in an `oxideav_core::VideoFrame`. Two picture sizes:
//!   QCIF 176x144 and CIF 352x288.
//! * Encoder: I + P pictures, integer-pel MC (spiral+diamond ME), per-GOB
//!   MQUANT rate control, FIL loop-filter RDO, full §4.2.3.4 MV-pred.
//! * BCH (511,493) error-correction framing (§5.4): the [`bch`] module
//!   wraps / unwraps the outer FEC multiframe layer (alignment pattern,
//!   `Fi` bit, BCH parity computation and per-frame syndrome check), and
//!   implements the spec-mandated `t = 1` single-bit error correction
//!   via [`bch::locate_single_error`] /
//!   [`bch::decode_multiframe_with_correction`]. Sweep-tested across all
//!   511 protected bit positions per frame.
//! * Hypothetical Reference Decoder buffer model (§5.2 + Annex B): the
//!   [`hrd`] module exposes the per-picture bit cap (256 kbits CIF /
//!   64 kbits QCIF) and the HRD buffer-occupancy walk used to verify
//!   a coded sequence won't underflow a conforming decoder's receiving
//!   buffer at a given channel rate.
//! * RTP payload-format wrapping (RFC 4587): the [`rtp`] module packs an
//!   elementary stream into a sequence of GOB-aligned H.261 RTP payloads
//!   (4-byte header + bitstream slice) and unpacks them back, plus the
//!   §4.2 RECOMMENDED MB-level fragmentation (`packetize_mb_fragmented`)
//!   that parses the Huffman layer to split oversize GOBs on macroblock
//!   boundaries with the full §4.1 GOBN/MBAP/QUANT/HMVD/VMVD context.
//!   The `RtpPacketizer` glue wraps each payload in a full RFC 3550 §5.1
//!   RTP fixed header (V/P/X/CC/M/PT/seq/timestamp/SSRC) so callers can
//!   hand `pack_frame` output straight to UDP / DTLS / SRTP.
//! * RTCP report packets (RFC 3550 §6.4): the [`rtcp`] module builds and
//!   parses Sender Report (SR, PT=200) and Receiver Report (RR, PT=201)
//!   packets — the control-channel companions to the [`rtp`] data path.
//!   `RtpPacketizer` tracks the sender's running packet / octet counts so a
//!   conformant SR can be emitted directly from its session state.
//! * RTCP SDES / BYE / APP packets (RFC 3550 §6.5 / §6.6 / §6.7): the
//!   [`rtcp`] module also builds and parses Source Description (SDES,
//!   PT=202), Goodbye (BYE, PT=203), and Application-Defined (APP, PT=204)
//!   packets, plus compound-packet (§6.1) concatenation and walking. APP
//!   carries a 4-byte ASCII name + 32-bit-aligned application-dependent
//!   data for application-layer extensions that do not warrant their own
//!   IANA-registered packet type.
//! * SDP media-type signalling (RFC 4587 §6.1.1 / §6.2): the [`sdp`] module
//!   maps the `video/H261` media type and its three optional parameters
//!   (`CIF`, `QCIF`, `D`) to the SDP `a=rtpmap` / `a=fmtp` attribute lines,
//!   with the §6.2.1 offer/answer helpers (per-format MPI frame-rate bound,
//!   the "at least one picture size" invariant, the RFC-2032 QCIF fallback).
//! * Codec delay measurement (§5.3 + Annex C): the [`annex_c`] module
//!   implements the §C measurement procedure — the toggling
//!   identification mark on its 97-input-frame schedule
//!   ([`annex_c::MarkGenerator`]), the mid-level transition detector at
//!   measuring points B / C ([`annex_c::MarkDetector`]), the averaged
//!   encoder / decoder / overall delay meter ([`annex_c::DelayMeter`]),
//!   and the §C.1 test-sequence precondition gate
//!   ([`annex_c::SequenceRequirements`]: > 100 s duration, ≥ 7.5 Hz coded
//!   rate, < 10 % stuffing). Like the [`hrd`] model it is a measurement
//!   helper — nothing it does changes the on-wire bitstream.
//! * Multipoint control signals (§2.8 + §4.3): the [`multipoint`] module
//!   models the three switched-multipoint control-signal state machines — the
//!   decoder-side §4.3.1 freeze-picture hold (released by the §4.3.3 PTYPE
//!   bit-3 signal or the §4.3.1 "at least six seconds" timeout) and the
//!   encoder-side §4.3.2 fast-update request (next picture coded INTRA with
//!   the §4.3.3 release bit set). Two of the three signals travel by external
//!   means (§4.3), so the module is control-plane logic the decoder / encoder
//!   wire into their picture loops rather than a bitstream change.
//! * Annex D still-image transmission (§D.2 + §D.3): the [`annex_d`] module
//!   exposes the [`annex_d::SubImageIndex`] mapping to / from the 5-bit `TR`
//!   field, the still-image dimensions (4× the currently transmitted video
//!   format), and the Figure D.1 2:1 × 2:1 sub-sample / reassemble transform
//!   between a full-resolution still image and its four sub-images.
//!
//! Out of scope:
//! * Multi-bit (weight ≥ 2) BCH (511,493) error correction. The
//!   generator polynomial has minimum distance `d = 3`, so the spec's
//!   `t = (d − 1) / 2 = 1` correction capability is the upper bound;
//!   weight-2 or denser error patterns are flagged
//!   (`uncorrectable_frames` in [`bch::DecodedMultiframe`]) but cannot
//!   be uniquely resolved by this code on its own — the inner H.261
//!   video VLC layer handles those via GOB resync.
//!
//! No runtime dependencies beyond `oxideav-core`, `oxideav-codec`, and
//! `oxideav-pixfmt`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::unusual_byte_groupings)]
#![allow(clippy::unnecessary_cast)]

pub mod annex_c;
pub mod annex_d;
pub mod bch;
pub mod block;
pub mod decoder;
pub mod encoder;
pub mod fdct;
pub mod gob;
pub mod hrd;
pub mod idct;
pub mod mb;
pub mod multipoint;
pub mod picture;
pub mod quant;
pub mod rtcp;
pub mod rtp;
pub mod sdp;
pub mod start_code;
pub mod tables;

use oxideav_core::{CodecCapabilities, CodecId, CodecTag};
use oxideav_core::{CodecInfo, CodecRegistry, RuntimeContext};

/// The canonical oxideav codec id for ITU-T H.261 video.
///
/// AVI FourCC `H261` maps to this id; raw `.h261` elementary-stream files
/// probe to it as well.
pub const CODEC_ID_STR: &str = "h261";

/// Register the H.261 decoder and encoder with a codec registry.
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("h261_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(352, 288);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            .encoder(encoder::make_encoder)
            .tags([CodecTag::fourcc(b"H261"), CodecTag::fourcc(b"h261")]),
    );
}

/// Unified registration entry point: install the H.261 codec factories
/// into the codec sub-registry of a [`RuntimeContext`].
///
/// This is the preferred entry point for new code — it matches the
/// convention every sibling crate now follows. Direct callers that need
/// only the codec sub-registry can keep using [`register_codecs`].
///
/// Also wired into [`oxideav_meta::register_all`] via the
/// [`oxideav_core::register!`] macro below.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
}

oxideav_core::register!("h261", register);

#[cfg(test)]
mod register_tests {
    use super::*;
    use oxideav_core::{CodecId, CodecParameters, RuntimeContext};

    #[test]
    fn register_via_runtime_context_installs_codec_factory() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let dec = ctx
            .codecs
            .first_decoder(&params)
            .expect("h261 decoder factory");
        assert_eq!(dec.codec_id().as_str(), CODEC_ID_STR);
    }

    #[test]
    fn register_via_runtime_context_installs_encoder_factory() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let enc = ctx
            .codecs
            .first_encoder(&params)
            .expect("h261 encoder factory");
        assert_eq!(enc.codec_id().as_str(), CODEC_ID_STR);
    }

    #[test]
    fn encoder_factory_qcif_defaults() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(176);
        params.height = Some(144);
        let enc = ctx
            .codecs
            .first_encoder(&params)
            .expect("h261 encoder factory qcif");
        assert_eq!(enc.codec_id().as_str(), CODEC_ID_STR);
        let out = enc.output_params();
        assert_eq!(out.width, Some(176));
        assert_eq!(out.height, Some(144));
    }

    #[test]
    fn encoder_factory_cif() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(352);
        params.height = Some(288);
        let enc = ctx
            .codecs
            .first_encoder(&params)
            .expect("h261 encoder factory cif");
        assert_eq!(enc.codec_id().as_str(), CODEC_ID_STR);
        let out = enc.output_params();
        assert_eq!(out.width, Some(352));
        assert_eq!(out.height, Some(288));
    }

    #[test]
    fn encoder_factory_rejects_bad_dimensions() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(320);
        params.height = Some(240);
        assert!(
            ctx.codecs.first_encoder(&params).is_err(),
            "should reject 320x240"
        );
    }
}
