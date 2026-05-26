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
| BCH (511,493) FEC framing (§5.4)                 | yes    | yes    |
| HRD buffer model (§5.2 + Annex B)                | yes    | yes    |
| RTP payload format (RFC 4587 §4.1)               | yes    | yes    |
| RTCP SR / RR reports (RFC 3550 §6.4)             | yes    | yes    |
| RTCP SDES + BYE + compound (RFC 3550 §6.5/§6.6)  | yes    | yes    |
| RTCP APP application-defined (RFC 3550 §6.7)     | yes    | yes    |
| SDP rtpmap/fmtp media type (RFC 4587 §6.1.1/6.2) | yes    | yes    |

H.261 only permits integer-pel motion vectors (range ±15); there are no
half-pel refinements, no B-pictures, and no start-code emulation prevention.
The spec tables (MBA / MTYPE / MVD / CBP / TCOEFF) are all implemented
directly from the PDF.

### Encoder quality

At the canonical H.261 target rate (64 kbit/s QCIF / 30 fps), the encoder
achieves ≥ 45 dB PSNR_Y on smooth content and ≥ 39 dB on the standard
`testsrc` test pattern (see `bench_testsrc_psnr`). ffmpeg cross-validates
all P-picture, MC, and FIL streams cleanly.

### HRD buffer model (§5.2 + Annex B)

The `hrd` module exposes the Hypothetical Reference Decoder buffer
walk callers use to verify a coded sequence won't underflow a
conforming H.261 receiver at a given channel rate. Two compliance
checks are surfaced as pure functions:

- `check_picture_cap(coded_bits, fmt)` — §5.2 per-picture cap
  (`64 kbits` QCIF, `256 kbits` CIF), excluding §5.4 FEC framing.
- `walk_buffer(pictures, N, params)` — Annex B buffer trajectory
  starting from `b_0 = 0`, accumulating `R * N / 29.97` bits per
  inter-picture interval and removing each picture instantaneously
  at its arrival boundary. Reports the first underflow index (if
  any) and the post-removal occupancy at every step.

`HrdParams::new(R_max)` derives `B = 4 * R_max / 29.97` and the
receiver buffer size `B + 256 kbits` (Annex B.2) via integer-rational
arithmetic so long sequences don't drift on floating-point round-off.

```rust
use oxideav_h261::hrd::{HrdParams, walk_buffer, check_picture_cap, PictureCapStatus};
use oxideav_h261::picture::SourceFormat;

let sizes_bits: Vec<u32> = vec![/* coded picture sizes */];
assert_eq!(
    check_picture_cap(sizes_bits[0], SourceFormat::Qcif),
    PictureCapStatus::Ok,
);
let params = HrdParams::new(64_000); // canonical p × 64 kbit/s ISDN
let trace = walk_buffer(&sizes_bits, /* skip N = */ 1, params);
assert!(trace.first_underflow.is_none(), "HRD violation at picture {:?}", trace.first_underflow);
```

The HRD is purely a coder-side compliance check — nothing in the
on-wire bitstream changes. The encoder doesn't run it by default
(see the module docstring for why); callers that need the assertion
drive it explicitly with their picture-size sequence.

### BCH (511,493) FEC framing (§5.4)

The `bch` module wraps / unwraps the outer forward-error-correction layer
that H.261 prescribes for noisy p × 64 kbit/s channels. The 8-frame
multiframe carries a 1-bit framing bit `Si`, a 1-bit fill indicator `Fi`,
492 bits of coded data, and 18 bits of BCH parity per frame (512 bits
total). The parity is computed via the generator polynomial
`g(x) = (x^9 + x^4 + 1)(x^9 + x^6 + x^4 + x^3 + 1)`. Frame lock requires
three complete multiframes of `(00011011)` alignment-pattern bits per
§5.4.4. Single-bit-error detection is wired through the per-frame
syndrome and reported back to the caller; single-bit correction is left
to the inner H.261 GOB-resync path, which is typically cheaper than
acting on the corrected bit at the FEC layer.

```rust
use oxideav_h261::bch::{encode_multiframe, decode_multiframe};

let coded_video: Vec<u8> = /* your H.261 elementary stream */ vec![];
let framed = encode_multiframe(&coded_video, coded_video.len() * 8);
let unwrapped = decode_multiframe(&framed).expect("frame lock");
assert_eq!(&unwrapped.data[..coded_video.len()], &coded_video[..]);
```

### RTP payload format (RFC 4587)

The `rtp` module packs an H.261 elementary stream into a sequence of
RTP-shaped payloads (the 4-byte H.261 payload header from §4.1 followed
by the bitstream slice) and unpacks them back. The GOB-aligned cheap
packetizer (§4.2) splits at start codes; SBIT/EBIT stay zero so the
round trip is byte-exact. RFC 4587 explicitly forbids the BCH framing
on the RTP path — the `bch` and `rtp` modules are mutually exclusive
consumers of an elementary stream.

```rust
use oxideav_h261::rtp::{packetize_gob_aligned, depacketize};

let elementary_stream: Vec<u8> = /* H.261 bytes from encode_intra_picture */ vec![];
let mtu_payload_budget = 1400; // typical IPv4 MTU minus the RTP fixed header
let packets = packetize_gob_aligned(&elementary_stream, mtu_payload_budget, true, false);
for p in &packets {
    // send `p.bytes` as the RTP payload; the marker bit goes in the RTP fixed header
    let _ = (&p.bytes, p.marker);
}
// At the receiver, depacketize the sequence to recover the elementary stream.
let recovered = depacketize(&packets).expect("depacketize");
assert_eq!(recovered, elementary_stream);
```

The packetizer handles GOBs that exceed the MTU by splitting at byte
boundaries (SBIT/EBIT stay zero in that case too). The 4-byte header
covers SBIT/EBIT, the I/V hint bits, and the GOBN/MBAP/QUANT/HMVD/VMVD
context fields; the GOB-aligned packetizer sets the context fields to
zero per §4.1. MB-level fragmentation with non-zero context is left to
the caller via [`pack_header`] / [`unpack_header`].

#### Encoder-side full RTP packetiser (`RtpPacketizer`)

`RtpPacketizer` is the higher-level glue between the encoder and the
RTP wire format. Construct one per RTP session (one SSRC) and call
`pack_frame(frame_bytes, rtp_timestamp_90khz)` once per coded picture;
it returns a sequence of `RtpPacket`s whose `bytes` field is a
complete RFC 3550 §5.1 fixed header (V=2, P=0, X=0, CC=0, M, PT,
sequence number, timestamp, SSRC) followed by the RFC 4587 §4.1
4-byte H.261 header and the payload slice. The marker bit is set on
the LAST packet of each frame per §4.1; sequence numbers auto-advance
mod 2^16 across frames; the same RTP timestamp is stamped on every
packet of a single frame (§4.1: "If a video image occupies more than
one packet, the timestamp SHALL be the same on all of those packets").

```rust
use oxideav_h261::encoder::H261Encoder;
use oxideav_h261::picture::SourceFormat;
use oxideav_h261::rtp::RtpPacketizer;

let mut enc = H261Encoder::new(SourceFormat::Qcif, 12);
// PT=96 (dynamic-range), SSRC=0xFEEDFACE, initial seq=0, MTU 1400 B.
let mut pk = RtpPacketizer::new(96, 0xFEEDFACE, 0, 1400).with_intra_only(false);

let y = vec![0u8; 176 * 144];
let cb = vec![128u8; 88 * 72];
let cr = vec![128u8; 88 * 72];
let bits = enc.encode_frame(&y, 176, &cb, 88, &cr, 88).unwrap();
let packets = pk.pack_frame(&bits, 0); // RTP 90-kHz timestamp = 0 for first frame
for p in &packets {
    // p.bytes is already framed; ship it as a UDP datagram payload.
    assert!(p.bytes.len() <= 1400);
}
assert!(packets.last().unwrap().marker, "M=1 on the last packet of a frame");
```

Receivers can use `parse_rtp_fixed_header` to strip the 12-byte RFC 3550
fixed header (including any CSRC list) and then feed the remaining inner
payload into the existing `unpack_header` + `depacketize` path.

### RTCP Sender / Receiver Reports (RFC 3550 §6.4)

The `rtcp` module builds and parses the two RTCP report packets an H.261
endpoint emits on its control channel: the **Sender Report** (SR, PT=200,
§6.4.1) with its 20-byte sender-info section (NTP + RTP timestamps,
sender's packet & octet counts) and the **Receiver Report** (RR, PT=201,
§6.4.2). Both carry up to 31 24-byte `ReceptionReportBlock`s (SSRC,
fraction lost, 24-bit two's-complement cumulative lost, extended highest
sequence number, jitter, LSR, DLSR). `RtpPacketizer` tracks the running
packet/octet counts so a conformant SR drops straight out of the session:

```rust
use oxideav_h261::rtcp::{ReceptionReportBlock, parse_report};
use oxideav_h261::rtp::RtpPacketizer;

let mut pk = RtpPacketizer::new(96, 0x1357_9BDF, 0, 1400);
let _ = pk.pack_frame(&frame_bytes, 0); // tracks packet & octet counts
let block = ReceptionReportBlock { ssrc: 0x2468_ACE0, fraction_lost: 26, ..Default::default() };
let sr = pk.sender_report(/* ntp */ 0xB44D_B705_2000_0000, &[block]).unwrap();
let report = parse_report(&sr).unwrap(); // round-trips the SR fields
# let frame_bytes: Vec<u8> = vec![];
```

The builders only (de)serialise the report wire format; the §6.2
transmission-interval scheduler and the §A.1/§A.3/§A.8 loss/jitter
estimators are session-management concerns above the codec and remain
caller-side. Empty RR (RC=0) is supported as the canonical "nothing to
report" packet that heads a compound RTCP packet.

### RTCP SDES / BYE + compound packets (RFC 3550 §6.5 / §6.6 / §6.1)

The `rtcp` module also builds and parses the two RTCP packet types that
round out a conformant control channel: **SDES** (Source Description,
PT=202, §6.5) and **BYE** (Goodbye, PT=203, §6.6). RFC 3550 §6.1 requires
every transmitted RTCP packet to be a *compound* packet — a report (SR/RR)
**first**, followed by an SDES packet carrying at least the mandatory
CNAME item, then optionally BYE. `compound` concatenates pre-built
sub-packets into one datagram body; `parse_compound` walks a received
datagram back into typed `RtcpPacket`s (`Report` / `Sdes` / `Bye` /
`Other` for unmodelled PTs such as APP=204), advancing via each
sub-packet's self-delimiting `length` field.

SDES chunks bind an SSRC/CSRC to a list of `SdesItem`s — `Cname` (§6.5.1,
mandatory), `Name`, `Email`, `Phone`, `Loc`, `Tool`, `Note`, and `Priv`
(the §6.5.8 prefix/value extension) — each independently 32-bit-aligned
with a trailing END item-type-0 byte and null padding. `build_cname_sdes`
is the one-call helper for the minimal "SSRC → CNAME" chunk every compound
packet must carry. BYE lists the leaving SSRC/CSRC identifiers plus an
optional free-text reason (8-bit-length-prefixed, null-padded to a 32-bit
boundary). Item text and reason strings are validated against the 255-octet
8-bit length limit; the parsers decode text UTF-8-lossily so a malformed
datagram never panics.

```rust
use oxideav_h261::rtcp::{build_cname_sdes, build_bye, build_receiver_report, compound, parse_compound};

// The canonical minimal compound: empty RR + SDES CNAME, then a BYE.
let rr = build_receiver_report(0xAAAA_AAAA, &[]).unwrap();
let sdes = build_cname_sdes(0xAAAA_AAAA, "me@host").unwrap();
let bye = build_bye(&[0xAAAA_AAAA], Some("leaving")).unwrap();
let datagram = compound(&[&rr, &sdes, &bye]);
let parsed = parse_compound(&datagram).unwrap();
assert_eq!(parsed.len(), 3);
```

### RTCP APP application-defined (RFC 3550 §6.7)

`build_app` / `parse_app` round-trip the **Application-Defined** RTCP packet
type (PT = 204). The wire format is the standard 4-byte RTCP header (with the
5-bit RC slot reused as a 5-bit `subtype`) followed by the originating SSRC,
a 4-octet ASCII `name`, and optional application-dependent data — §6.7 requires
the data section to be a multiple of 32 bits long. Names are byte-exact:
§6.7 mandates that uppercase and lowercase characters be treated as distinct,
so the parser surfaces the four bytes without any case folding. The builder
rejects `subtype > 31`, `name.len() != 4`, and `data.len() % 4 != 0`; the
parser rejects V != 2, PT != 204, a length field smaller than the mandatory
12-byte header, and a length field that runs past the buffer end. APP packets
that flow through a compound RTCP datagram now come back as a typed
`RtcpPacket::App(AppPacket)` variant from `parse_compound`; unknown packet
types (e.g. RFC 4585 RTPFB = 205) still surface as `RtcpPacket::Other`.

```rust
use oxideav_h261::rtcp::{build_app, parse_app, build_receiver_report, build_cname_sdes,
                          compound, parse_compound, RtcpPacket};

// Application-defined ping-style packet: name "PING", subtype = sequence,
// data = 4-byte echo cookie.
let cookie: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];
let app = build_app(/* subtype */ 7, /* ssrc */ 0xCAFE_F00D, b"PING", &cookie).unwrap();
let parsed = parse_app(&app).unwrap();
assert_eq!(parsed.subtype, 7);
assert_eq!(&parsed.name, b"PING");
assert_eq!(parsed.data, cookie);

// Drops cleanly into a compound datagram alongside the mandatory RR + SDES.
let rr = build_receiver_report(0xCAFE_F00D, &[]).unwrap();
let sdes = build_cname_sdes(0xCAFE_F00D, "me@host").unwrap();
let datagram = compound(&[&rr, &sdes, &app]);
let packets = parse_compound(&datagram).unwrap();
assert!(matches!(packets[2], RtcpPacket::App(_)));
```

The §6.2 transmission-interval scheduler and the §A.1 / §A.3 / §A.8
loss-fraction / jitter estimators remain caller-side — they belong above the
codec in session-management code, not in the wire-format module.

### SDP media type and rtpmap/fmtp parameters (RFC 4587 §6.1.1 / §6.2)

The `sdp` module maps the `video/H261` media-type registration to the SDP
`a=rtpmap` and `a=fmtp` attribute lines an H.261 endpoint exchanges at session
setup. The `a=rtpmap` line is fixed: encoding name `H261`, clock rate `90000`
(`ENCODING_NAME` / `CLOCK_RATE`), media name `video` for the `m=` line.
`format_rtpmap(pt)` emits it; `parse_rtpmap` reads one back and confirms it is
H.261 (case-insensitively on the encoding name, rejecting other codecs).

The three optional `a=fmtp` parameters from §6.1.1 — `CIF`, `QCIF`, and `D` —
are modelled by `H261FmtpParams`. `CIF` / `QCIF` carry an integer 1..=4 (the
minimum picture interval, MPI) meaning "max rate `29.97 / value` fps"; `D`
signals H.261 Annex D still-image support (`1` = yes, `0`/absent = no).
`format_value` emits the bare `CIF=2;QCIF=1;D=1` list (CIF before QCIF, per the
§6.2.1 worked example) and `format_fmtp(pt, &params)` wraps it in the full
`a=fmtp:<pt> …` line (returning `None` when no parameters are set, since §6.2
includes the line only "if any"). `parse_value` / `parse_fmtp` reverse it,
enforcing the 1..=4 MPI range and the `D ∈ {0,1}` rule, tolerating whitespace,
matching parameter names case-insensitively, and skipping unknown parameters
forward-compatibly.

```rust
use oxideav_h261::picture::SourceFormat;
use oxideav_h261::sdp::{format_rtpmap, format_fmtp, parse_rtpmap, parse_fmtp, H261FmtpParams};

// Build the two attribute lines for payload type 31 (the §6.2.1 example).
let params = H261FmtpParams { cif: Some(2), qcif: Some(1), d: Some(true) };
assert_eq!(format_rtpmap(31), "a=rtpmap:31 H261/90000");
assert_eq!(format_fmtp(31, &params).unwrap(), "a=fmtp:31 CIF=2;QCIF=1;D=1");

// Parse a peer's offer back into typed parameters.
let map = parse_rtpmap("a=rtpmap:31 H261/90000").unwrap();
let offered = parse_fmtp("a=fmtp:31 CIF=2;QCIF=1;D=1", map.payload_type).unwrap().unwrap();
assert_eq!(offered, params);
// CIF=2 ⇒ ≤ 15 fps (29.97/2 = 2997/200) per §6.2.1.
assert_eq!(offered.max_frame_rate(SourceFormat::Cif), Some((2997, 200)));
```

The §6.2.1 offer/answer helpers round out the module: `validate` enforces
"SHALL specify at least one supported picture size", `rfc2032_fallback` returns
the §6.2.1 default (QCIF at MPI=1) assumed for a peer that sends no picture-size
parameter, and `max_frame_rate` / `mpi` / `supports` read out the advertised
capability per `SourceFormat`. The SDP offer/answer state machine itself and
the rest of the session description (`v=` / `o=` / `c=` / `t=`) stay caller-side
— this module owns only the H.261-specific `rtpmap` / `fmtp` attribute wire
format. The RFC 2032 H.261-specific RTCP control packets (FIR / NACK) are
deliberately not implemented: RFC 4587 §7.1 mandates that "new implementations
SHALL ignore them, and they SHALL NOT be used by new implementations."

### Daily fuzzing

A `cargo-fuzz` harness lives under `fuzz/`. The single target
(`decode_h261`) drives arbitrary fuzz-supplied bytes through the
decoder's full public surface (`send_packet` → drain `receive_frame`
→ `flush` → drain again), so the PSC / GBSC start-code scanners,
every VLC table (MBA / MTYPE / MVD / CBP / TCOEFF + 20-bit escape),
the §4.2.3.4 MV predictor, the integer-pel MC indexing, the §3.2.3
loop filter, and the 8×8 IDCT are all exercised against bytes whose
shape the fuzzer dictates. The contract under test is purely that
every call *returns*: no panic, no abort, no integer overflow (in
debug / ASAN builds), no out-of-bounds index, no allocator OOM.

The seed corpus (`fuzz/corpus/decode_h261/`) is generated from the
in-tree encoder and covers QCIF + CIF I-pictures across a quantiser
range (8 / 12 / 31), plus QCIF + CIF I+P pairs that exercise motion
compensation and the loop filter. `tests/fuzz_seed_corpus.rs` drives
the same logic on stable Rust against the same corpus so the regular
CI matrix catches a regression in the public decoder surface without
waiting for the daily fuzz run.

The workflow at `.github/workflows/fuzz.yml` runs `cargo fuzz run
decode_h261 -- -max_total_time=1800` (30-minute budget) once a day
via the org-shared `crate-fuzz.yml` reusable workflow.

```sh
cargo install cargo-fuzz
cd crates/oxideav-h261
cargo +nightly fuzz run decode_h261 -- -max_total_time=60
```

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
