//! Pure-Rust ITU-T H.261 video decoder.
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
//!
//! Out of scope:
//! * BCH (511,493) forward error correction framing (§5.4) — decoder is fed
//!   already-extracted video bitstream.
//! * Encoder — decode-only for now.
//!
//! No runtime dependencies beyond `oxideav-core`, `oxideav-codec`, and
//! `oxideav-pixfmt`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::unusual_byte_groupings)]
#![allow(clippy::unnecessary_cast)]

pub mod block;
pub mod decoder;
pub mod gob;
pub mod idct;
pub mod mb;
pub mod picture;
pub mod start_code;
pub mod tables;

use oxideav_codec::{CodecInfo, CodecRegistry};
use oxideav_core::{CodecCapabilities, CodecId, CodecTag};

/// The canonical oxideav codec id for ITU-T H.261 video.
///
/// AVI FourCC `H261` maps to this id; raw `.h261` elementary-stream files
/// probe to it as well.
pub const CODEC_ID_STR: &str = "h261";

/// Register the H.261 decoder with a codec registry.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("h261_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(352, 288);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            .tags([CodecTag::fourcc(b"H261"), CodecTag::fourcc(b"h261")]),
    );
}
