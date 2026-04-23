# oxideav-h261

Pure-Rust **ITU-T H.261** video decoder — the original 1990/1993 videoconferencing
codec. Decodes both I-pictures (INTRA macroblocks) and P-pictures (INTER with
integer-pel motion compensation + optional loop filter). QCIF (176×144) and CIF
(352×288) source formats. Output is YUV 4:2:0. Zero C dependencies.

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

| Feature                                          | Decode |
|--------------------------------------------------|:------:|
| Picture header (PSC / TR / PTYPE / PEI / PSPARE) | yes    |
| GOB layer (GBSC / GN / GQUANT / GEI / GSPARE)    | yes    |
| Source formats QCIF (176×144), CIF (352×288)     | yes    |
| Macroblock layer (MBA / MTYPE / CBP / MVD)       | yes    |
| TCOEFF VLC + escape                              | yes    |
| 8×8 IDCT, dequantisation (Table 5/H.261)         | yes    |
| INTRA and INTER prediction                       | yes    |
| Integer-pel motion compensation (MC)             | yes    |
| Loop filter (FIL, §3.2.3)                        | yes    |
| BCH forward error correction (§5.4)              | no     |

H.261 only permits integer-pel motion vectors (range ±15); there are no
half-pel refinements, no B-pictures, and no start-code emulation prevention.
The spec tables (MBA / MTYPE / MVD / CBP / TCOEFF) are all implemented
directly from the PDF.

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
