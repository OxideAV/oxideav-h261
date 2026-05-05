# oxideav-h261

Pure-Rust **ITU-T H.261** video codec — the original 1990/1993 videoconferencing
codec. Decodes and encodes both I-pictures (INTRA macroblocks) and P-pictures
(INTER with integer-pel motion compensation + loop filter). QCIF (176×144) and
CIF (352×288) source formats. Output is YUV 4:2:0. Zero C dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-h261 = "0.0"
```

## Feature matrix

| Feature                                          | Decode | Encode |
|--------------------------------------------------|:------:|:------:|
| Picture header (PSC / TR / PTYPE / PEI / PSPARE) | yes    | yes    |
| GOB layer (GBSC / GN / GQUANT / GEI / GSPARE)    | yes    | yes    |
| Source formats QCIF (176×144), CIF (352×288)     | yes    | yes    |
| Macroblock layer (MBA / MTYPE / CBP / MVD)       | yes    | yes    |
| TCOEFF VLC + escape                              | yes    | yes    |
| 8×8 (I)DCT, (de)quantisation (Table 5/H.261)     | yes    | yes    |
| INTRA prediction (I-pictures)                    | yes    | yes    |
| INTER prediction (P-pictures, no MC)             | yes    | yes    |
| Integer-pel MC (spiral+diamond ME, ±15)          | yes    | yes    |
| Loop filter (FIL, §3.2.3) with per-MB RDO        | yes    | yes    |
| Per-GOB MQUANT rate control (§4.2.3.3)           | n/a    | yes    |
| Encoder registry (`first_encoder` / `bit_rate`)  | n/a    | yes    |
| BCH forward error correction (§5.4)              | no     | no     |

H.261 only permits integer-pel motion vectors (range ±15); there are no
half-pel refinements, no B-pictures, and no start-code emulation prevention.
The spec tables (MBA / MTYPE / MVD / CBP / TCOEFF) are all implemented
directly from the PDF.

### Encoder quality

At the canonical H.261 target rate (64 kbit/s QCIF / 30 fps), the encoder
achieves ≥ 45 dB PSNR_Y on smooth content and ≥ 39 dB on the standard
`testsrc` test pattern (see `bench_testsrc_psnr`). ffmpeg cross-validates
all P-picture, MC, and FIL streams cleanly.

## Quick use

```rust
use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Packet, TimeBase};
use oxideav_h261::decoder::H261Decoder;

let mut dec = H261Decoder::new(CodecId::new(oxideav_h261::CODEC_ID_STR));
let pkt = Packet::new(0, TimeBase::new(1, 30), bitstream_bytes);
dec.send_packet(&pkt)?;
match dec.receive_frame() {
    Ok(oxideav_core::Frame::Video(vf)) => {
        // vf.format == PixelFormat::Yuv420P
    }
    Err(oxideav_core::Error::NeedMore) => { /* feed more packets */ }
    Err(e) => return Err(e.into()),
    _ => {}
}
# Ok::<(), oxideav_core::Error>(())
```

### Codec ID

- Codec: `"h261"`; accepted pixel format `Yuv420P`; dimensions QCIF or CIF.
- AVI FourCC `H261` maps to this id.

## License

MIT — see [LICENSE](LICENSE).
