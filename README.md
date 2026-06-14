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
| Annex A IDCT accuracy conformance test            | yes    | yes    |
| INTRA prediction (I-pictures)                    | yes    | yes    |
| INTER prediction (P-pictures, no MC)             | yes    | yes    |
| Integer-pel MC (spiral+diamond ME, ±15)          | yes    | yes    |
| Loop filter (FIL, §3.2.3) with per-MB RDO        | yes    | yes    |
| Forced updating (§3.4 per-MB cyclic INTRA refresh) | n/a  | yes    |
| Per-GOB MQUANT rate control (§4.2.3.3)           | n/a    | yes    |
| Encoder registry (`first_encoder` / `bit_rate`)  | n/a    | yes    |
| BCH (511,493) FEC framing (§5.4) + t = 1 correction | yes | yes    |
| HRD buffer model (§5.2 + Annex B)                | yes    | yes    |
| RTP payload format (RFC 4587 §4.1)               | yes    | yes    |
| RTP MB-level fragmentation (RFC 4587 §4.2)       | yes    | yes    |
| RTCP SR / RR reports (RFC 3550 §6.4)             | yes    | yes    |
| RTCP SDES + BYE + compound (RFC 3550 §6.5/§6.6)  | yes    | yes    |
| RTCP APP application-defined (RFC 3550 §6.7)     | yes    | yes    |
| SDP rtpmap/fmtp media type (RFC 4587 §6.1.1/6.2) | yes    | yes    |
| Annex D still-image sub-image transform (§D.2/§D.3) | yes | yes    |

H.261 only permits integer-pel motion vectors (range ±15); there are no
half-pel refinements, no B-pictures, and no start-code emulation prevention.
The spec tables (MBA / MTYPE / MVD / CBP / TCOEFF) are all implemented
directly from the PDF.

The §4.2.3.4 MVD predictor honours all three "vector regarded as zero"
conditions: the start of each MB row in a GOB (MBA 1, 12, 23 — a GOB is
11 MBs wide × 3 rows), an MBA difference other than 1 (intervening skipped
MBs), and a previous macroblock whose MTYPE was not motion-compensated.
The row-boundary reset (MBA 12 / 23) is applied on the absolute MBA, not
just the MBA difference, so a conformant stream that carries a non-zero MV
at MB 11 (or 22) and a motion-compensated MB 12 (or 23) decodes the right
predictor. The decoder, the encoder's MVD derivation, and the RFC 4587
§4.2 MB-level fragmentation walker all share one `mb::mvd_predictor`
implementation of these rules, so they stay bit-for-bit in agreement.

### Encoder quality

At the canonical H.261 target rate (64 kbit/s QCIF / 30 fps), the encoder
achieves ≥ 45 dB PSNR_Y on smooth content and ≥ 39 dB on the standard
`testsrc` test pattern (see `bench_testsrc_psnr`). ffmpeg cross-validates
all P-picture, MC, and FIL streams cleanly.

### Forced updating (§3.4)

H.261 §3.4 mandates that, "for control of accumulation of inverse
transform mismatch error, a macroblock should be forcibly updated at
least once per every 132 times it is transmitted." The encoder honours
this with a per-macroblock cyclic INTRA refresh that runs independently
of the whole-frame `intra_period` I-refresh. `H261Encoder` keeps a
transmission counter for every macroblock (global raster order across
all GOBs — GOB 0 holds MBs 0..33, GOB 1 holds 33..66, …) and forces the
due macroblocks into INTRA mode inside the next P-picture before any
counter can reach the period. To avoid a bandwidth spike when every
counter would otherwise hit the cap on the same frame, the scheduler
also forces a rotating slice of `ceil(total_mbs / period)` MBs per
P-frame, so a complete refresh sweep is spread evenly across the
refresh window. An INTRA macroblock in a P-picture is never
motion-compensated, so emitting one resets the §4.2.3.4 MVD predictor.

`H261Encoder::with_forced_update_period(p)` overrides the period — the
default is `132` (the spec maximum); `0` disables per-MB forced updating
(not recommended for long P-only sequences). Callers that drive the
stateless P-picture path directly can pass their own forced-update set
to `encode_inter_picture_forced_update` — for example, the RFC 4587
§C.3 loss-driven refresh that re-INTRA-codes only the macroblocks a
receiver reports as damaged.

```rust
use oxideav_h261::encoder::H261Encoder;
use oxideav_h261::picture::SourceFormat;

// P-only sequence (no whole-frame I-refresh) that still bounds
// inverse-transform mismatch via §3.4 per-MB forced updating.
let mut enc = H261Encoder::new(SourceFormat::Qcif, 12)
    .with_intra_period(0)        // no whole-frame I after the first
    .with_forced_update_period(132); // §3.4 upper bound
let y = vec![0u8; 176 * 144];
let cb = vec![128u8; 88 * 72];
let cr = vec![128u8; 88 * 72];
let _first = enc.encode_frame(&y, 176, &cb, 88, &cr, 88).unwrap(); // I
let _p = enc.encode_frame(&y, 176, &cb, 88, &cr, 88).unwrap();     // P + forced INTRA MBs
```

### IDCT accuracy (Annex A)

Annex A of Recommendation H.261 is an integral part of the spec and defines a
measurable conformance procedure for the inverse 8×8 DCT. The
`tests/idct_annex_a.rs` integration test implements §A.1..§A.9 end-to-end:
the §A.1 deterministic 32-bit LCG (`randx = randx * 1103515245 + 12345`, keep
30 bits LSB-cleared, divide by `2^31`, scale by `L+H+1`, truncate, subtract
`L`) generates 10 000 8×8 blocks per dataset; the §A.2 forward DCT and §A.4
reference IDCT run in 64-bit float per the equations in §3.2.4; §A.3 rounds
to nearest and clips to `[-2048, +2047]` for the 12-bit IDCT input; our
`idct_signed` produces the test output, clipped to `[-256, +255]`. The §A.7
thresholds are then asserted across all three §A.1 ranges and both polarities
(§A.9 sign-flip), plus the §A.8 all-zero-in-all-zero-out invariant.

Measured headroom against the §A.7 limits on the (L=256, H=255) dataset:

| §A.7 statistic                | Limit  | Measured |
|-------------------------------|-------:|---------:|
| per-pel peak error            | ≤ 1    | 1        |
| per-pel mean square error     | ≤ 0.06 | 1.0e-4   |
| overall mean square error     | ≤ 0.02 | 6.0e-6   |
| per-pel \|mean error\|        | ≤ 0.015| 1.0e-4   |
| overall \|mean error\|        | ≤ 0.0015 | 3.0e-6 |

The (L=H=5) and (L=H=300) datasets are equally well inside the spec
margins. No external IDCT source is read: the f64 reference DCT and IDCT
live in the test file and are written directly from §3.2.4 / Annex A.

### Annex D still-image transmission (§D.2 + §D.3)

H.261 Annex D describes the procedure for transmitting still images at
four times the normal video resolution by temporarily stopping the
motion video and sending four 2:1 × 2:1 sub-sampled sub-images in
sequential order. The `annex_d` module is the helper surface that
wires this into the rest of the codec without disturbing the standard
motion-video pipeline:

- `SubImageIndex` — one of the four sub-images (0..3).
- `still_image_tr(idx)` / `parse_still_image_tr(tr)` — pack / parse
  the §D.3 5-bit `TR` field (top 3 bits = 0, low 2 bits = `idx`).
- `still_image_dimensions(fmt)` — §D.2 4× video-format size (QCIF
  video ⇒ 352×288 still, CIF video ⇒ 704×576 still); chroma sizes
  via `still_image_chroma_dimensions`.
- `subsample_still_image(fmt, y, cb, cr)` — Figure D.1 2:1 × 2:1
  sub-sample (origin map `0→(0,0)`, `1→(0,1)`, `2→(1,1)`, `3→(1,0)`
  per 2×2 tile) producing four `SubImagePlanes` at the video-format
  dimensions for `fmt`.
- `reassemble_still_image(fmt, subs)` — inverse transform; bit-exact
  round-trips `subsample_still_image` on both 4:2:0 chroma planes.
- `PictureHeader::still_image_sub_index()` — convenience accessor
  that returns `Ok(None)` for ordinary motion-video pictures and
  `Ok(Some(idx))` for an Annex-D-signalled sub-image, surfacing the
  §D.3 high-bits-must-be-zero violation as `Err`.
- `encoder::write_picture_header_full` — header writer that lets the
  caller drive the HI_RES bit (clear for an Annex-D sub-image,
  set for normal motion video).

`tests/annex_d.rs` round-trips the encoder + parser through every
sub-image in both QCIF and CIF mode and round-trips the
sub-sample/reassemble transform on full-size synthetic still images
(luma + both chroma planes). H.261's per-frame bit cap (§5.2) still
applies to each sub-image, the multiplex below the picture layer is
unchanged (§D.5), and the §5.4 FEC layer is not affected.

```rust
use oxideav_h261::annex_d::{
    subsample_still_image, reassemble_still_image, SubImageIndex,
    still_image_tr, still_image_dimensions,
};
use oxideav_h261::picture::SourceFormat;

let fmt = SourceFormat::Qcif;
let (sw, sh) = still_image_dimensions(fmt); // (352, 288)
let still_y: Vec<u8> = vec![0; (sw * sh) as usize];
let still_cb: Vec<u8> = vec![128; ((sw / 2) * (sh / 2)) as usize];
let still_cr: Vec<u8> = vec![128; ((sw / 2) * (sh / 2)) as usize];

let subs = subsample_still_image(fmt, &still_y, &still_cb, &still_cr);
for (i, sub) in subs.iter().enumerate() {
    let idx = SubImageIndex::from_u8(i as u8);
    let tr_for_picture = still_image_tr(idx); // top 3 bits = 0
    // ... encode sub-image with `H261Encoder` and stamp `tr_for_picture`
    // into the picture header via `write_picture_header_full(..., false)`.
    let _ = (sub, tr_for_picture);
}
let (y, cb, cr) = reassemble_still_image(fmt, &subs);
assert_eq!(y, still_y);
assert_eq!(cb, still_cb);
assert_eq!(cr, still_cr);
```

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
§5.4.4. The module exposes both a detection-only path
(`decode_multiframe`, raw syndrome surfaced as `corrupted_frames`) and
the spec-mandated `t = 1` single-bit correction path
(`decode_multiframe_with_correction`, which additionally maps each
non-zero syndrome to the corresponding 511-bit codeword position via
`locate_single_error` and flips that bit before emitting the data
field). The correction sweep test in `src/bch.rs` walks every of the
511 protected bit positions in a frame (Fi + 492 data + 18 parity),
flipping one at a time, and verifies the recovered payload matches the
original bit-exact.

```rust
use oxideav_h261::bch::{
    encode_multiframe, decode_multiframe, decode_multiframe_with_correction,
};

let coded_video: Vec<u8> = /* your H.261 elementary stream */ vec![];
let framed = encode_multiframe(&coded_video, coded_video.len() * 8);

// Detection-only path.
let unwrapped = decode_multiframe(&framed).expect("frame lock");
assert_eq!(&unwrapped.data[..coded_video.len()], &coded_video[..]);

// `t = 1` correction path: corrects a single bit flip per frame, surfaces
// uncorrectable (weight ≥ 2) frames via `uncorrectable_frames`.
let corrected = decode_multiframe_with_correction(&framed).expect("lock");
assert_eq!(corrected.corrupted_frames, corrected.corrected_frames + corrected.uncorrectable_frames);
```

For a `t = 1` BCH code the syndrome of a single-bit error at
codeword position `p` (where `p = 0` is `Fi`, `p = 1..493` are the
492 data bits, `p = 493..511` are the 18 parity bits) equals
`x^(510 − p) mod g(x)`. `locate_single_error` walks `x^i mod g(x)`
for `i = 0..511` and stops when the running power matches the
syndrome; a non-zero syndrome that doesn't match any single-bit
pattern reports `None` (weight ≥ 2 error the code cannot resolve).
Per H.261 §5.4.1 the spec explicitly labels the outer layer an
"error correcting code" and the BCH (511, 493) code with this
generator polynomial supports `t = 1` correction; the inner H.261
GOB-resync remains the recovery path for weight-2-or-denser noise.

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

The cheap packetizer handles GOBs that exceed the MTU by splitting at
byte boundaries (SBIT/EBIT stay zero in that case too). The 4-byte
header covers SBIT/EBIT, the I/V hint bits, and the
GOBN/MBAP/QUANT/HMVD/VMVD context fields; the GOB-aligned packetizer
sets the context fields to zero per §4.1.

#### MB-level fragmentation (`packetize_mb_fragmented`, §4.2)

`packetize_mb_fragmented` is the §4.2 RECOMMENDED packetization: the
elementary stream is parsed once at the Huffman layer (§4.2: "only the
Huffman encoding must be parsed and ... it is not necessary to
decompress the stream fully") to collect every legal split point —
each PSC/GBSC and the first bit after every macroblock — then packets
are filled greedily (multiple GOBs/MBs per packet when they fit, per
the §3.2 efficiency recommendation) under the §3.2 rules: an MB is
never split across packets, the stream is never fragmented between a
GOB header and MB 1, and no packet crosses a PSC (different frames
need different RTP timestamps). A packet starting mid-GOB carries the
full §4.1 context — non-zero SBIT/EBIT (the encoder bit-packs GOB
headers, so split bits land mid-byte and consecutive packets share
the split byte), GOBN, the MBAP predictor (last MBA of the previous
packet, biased -1), the QUANT in effect, and the reference HMVD/VMVD
(zero when the previous MB was not MC-coded) — so a receiver can
resume decoding mid-GOB after losing the preceding packet. The
internal Huffman-layer walk is verified bit-for-bit against the real
decoder (a `decode_macroblock` oracle) across I- and P-pictures in
`src/rtp.rs`; `depacketize` reassembles the output byte-exactly. A
stream whose smallest legal fragment exceeds the budget surfaces
`RtpError::FragmentTooLarge` instead of emitting an undecodable
packet. `RtpPacketizer::with_mb_fragmentation(true)` switches the
session glue onto this path (with an automatic fallback to the
byte-split cheap path so a frame is never dropped).

```rust
use oxideav_h261::rtp::{packetize_mb_fragmented, depacketize};

let elementary_stream: Vec<u8> = /* one coded picture */ vec![];
let packets = packetize_mb_fragmented(&elementary_stream, 200, true, false)?;
for p in &packets {
    // Continuation packets carry GOBN/MBAP/QUANT/HMVD/VMVD per §4.1.
    let _ = (p.header.gobn, p.header.mbap, p.header.quant);
}
let recovered = depacketize(&packets).expect("byte-exact");
# Ok::<(), oxideav_h261::rtp::RtpError>(())
```

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
`parse_rtpmap` preserves the wire `clock_rate` verbatim so a misbehaving peer
can be diagnosed; the typed `RtpMap::is_rfc4587_compliant` accessor reports
whether the §6.2 MUST (clock rate `= 90000`) holds, and
`parse_rtpmap_strict` is the single-call wrapper that returns `None` for any
non-conformant line.

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
forward-compatibly. `parse_fmtp_strict` is the §6.2.1 single-call wrapper:
it returns `None` for any well-formed line that violates the §6.2.1 "SHALL
specify at least one supported picture size" MUST (so a `D`-only or empty
parameter list is dropped cleanly), while preserving the typed
`Some(Err(SdpError::…))` propagation on a malformed list so the caller still
sees why the parse failed. Mirrors the `parse_rtpmap` / `parse_rtpmap_strict`
pair on the `a=rtpmap` side of §6.2.

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
parameter, `preferred_picture_size` returns the structural "CIF when advertised,
else QCIF" mode (the order `format_value` emits), and
`H261FmtpParams::parse_preference_order` returns the literal §6.2.1
"Parameters offered first are the most preferred picture mode to be received"
order observed in a raw `a=fmtp` value — so a peer that emits `QCIF=1;CIF=2`
is recognised as QCIF-preferring even though the typed `H261FmtpParams` view
loses token order. The emit-side dual,
`H261FmtpParams::format_value_preferred(preferred)` /
`format_fmtp_preferred(pt, &params, preferred)`, lets an endpoint express its
own §6.2.1 preference on the wire by leading with the preferred picture-size
token (`QCIF=1;CIF=2;D=1` for a QCIF-preferring endpoint); a CIF preference is
byte-identical to the canonical `format_value` / `format_fmtp` order, an
unadvertised preference falls back to that canonical order, and `D` stays
last since it is an Annex-D codec option, not a picture mode. `max_frame_rate`
/ `mpi` / `supports` read out the advertised capability per `SourceFormat`. The
free function `negotiate_answer(offer, our_capability)` computes the §6.2.1
**answer** by intersecting both peers' picture sizes, taking `max(offer.MPI,
our.MPI)` per shared size (the more restrictive frame-rate bound binds), gating
`D=1` on both sides advertising Annex D, and applying the RFC 2032
fallback (`QCIF=1`) when the offer omits picture sizes. The SDP offer/answer
state machine itself and the rest of the session description (`v=` / `o=` /
`c=` / `t=`) stay caller-side — this module owns only the H.261-specific
`rtpmap` / `fmtp` attribute wire format. The RFC 2032 H.261-specific RTCP
control packets (FIR / NACK) are deliberately not implemented: RFC 4587 §7.1
mandates that "new implementations SHALL ignore them, and they SHALL NOT be
used by new implementations."

```rust
use oxideav_h261::sdp::{H261FmtpParams, negotiate_answer};
// Peer A offers CIF=2 (≤ 14.985 fps), QCIF=1 (≤ 29.97 fps), Annex D supported.
let offer = H261FmtpParams { cif: Some(2), qcif: Some(1), d: Some(true) };
// We can decode CIF at MPI=2 (≤ 14.985 fps), QCIF at MPI=2 (≤ 14.985 fps);
// we don't support Annex D.
let ours = H261FmtpParams { cif: Some(2), qcif: Some(2), d: None };
let answer = negotiate_answer(&offer, &ours).unwrap();
// CIF survives at MPI=2 (both sides agree); QCIF tightens to MPI=2 (the
// answer must satisfy both peers' upper rate bounds); D drops because we
// didn't advertise it.
assert_eq!(answer.cif, Some(2));
assert_eq!(answer.qcif, Some(2));
assert_eq!(answer.d, None);
```

### Daily fuzzing

A `cargo-fuzz` harness lives under `fuzz/`. Five targets cover the
five distinct attack surfaces an H.261 endpoint exposes:

* **`decode_h261`** drives arbitrary fuzz-supplied bytes through the
  decoder's full public surface (`send_packet` → drain `receive_frame`
  → `flush` → drain again), so the PSC / GBSC start-code scanners,
  every VLC table (MBA / MTYPE / MVD / CBP / TCOEFF + 20-bit escape),
  the §4.2.3.4 MV predictor, the integer-pel MC indexing, the §3.2.3
  loop filter, and the 8×8 IDCT are all exercised against bytes whose
  shape the fuzzer dictates.
* **`parse_rtcp_compound`** drives arbitrary fuzz-supplied bytes
  through the RTCP control-channel parser surface
  (`parse_compound` / `parse_report` / `parse_sdes` / `parse_bye` /
  `parse_app`), so the RFC 3550 §6.1 compound walk (advance via each
  sub-packet's 16-bit `length` field), the SR / RR fixed header + RC
  block walk (§6.4.1 / §6.4.2), the SDES chunk + item walk including
  the §6.5.8 PRIV inner 8-bit length prefix, the BYE reason-string
  length prefix (§6.6), and the APP `name`/`data` 32-bit alignment
  (§6.7) are all exercised against attacker-controlled bytes.
* **`decode_bch_multiframe`** drives arbitrary fuzz-supplied bytes
  through the H.261 §5.4 BCH (511, 493) FEC multiframe parser
  (`decode_multiframe` / `parity18` / `syndrome18`), so the §5.4.4
  lock-search candidate sweep (511 bit offsets × 24 framing-bit reads
  at a stride of `FRAME_BITS = 511`), the §5.4.2 GF(2) long-division
  shift-register, and the §5.4.3 per-frame `Fi` / 492-data-bit /
  18-parity-bit walk are all exercised against attacker-controlled
  bytes. The target also runs an **error-injection mode**: it frames a
  deterministic synthetic payload through the in-crate
  `encode_multiframe`, then uses the fuzz input as a bit-flip vector
  to corrupt up to 16 attacker-chosen bit positions in the framed
  stream before re-decoding — driving the non-zero-syndrome
  `corrupted_frames` branch and (when a flip lands on a framing bit
  inside the 24-bit lock window) the lock-loss path.
* **`parse_rtp_payload`** drives arbitrary fuzz-supplied bytes
  through the RTP data-path parser surface — the network-receive
  parsers an endpoint runs on every received UDP datagram **before**
  any H.261 bitstream layer sees a byte. Three entry points are
  exercised: `parse_rtp_fixed_header` (RFC 3550 §5.1 — V / P / X / CC
  / M / PT / seq / ts / SSRC + 0..=15 CSRC entries, bounds-checked
  against the attacker-controlled CC field), `unpack_header` (RFC
  4587 §4.1 — the 4-byte H.261 RTP payload header with SBIT / EBIT /
  I / V / GOBN / MBAP / QUANT / HMVD / VMVD and the §4.1 sign-
  extension of the two 5-bit MV deltas), and `depacketize` (the
  multi-packet bit-walker that honours per-packet SBIT/EBIT alignment
  and asserts the recovered elementary stream contains at least one
  start code). The harness carves the fuzz input into up to four
  synthetic `H261RtpPayload`s with attacker-chosen header fields and
  attacker-chosen data lengths, so the slow-path bit-walker and the
  final `iter_start_codes` sanity check both stay covered.
* **`parse_sdp_fmtp`** drives arbitrary fuzz-supplied bytes through
  the SDP signalling parser surface — the attribute-line parsers an
  endpoint runs on every received Session Description Protocol offer
  or answer at session setup **before** any RTP / RTCP / H.261 layer
  sees a byte. Four entry points are exercised: `parse_rtpmap`
  (RFC 4587 §6.2 — `a=rtpmap:<pt> H261/90000` with case-insensitive
  encoding-name match, payload-type and clock-rate integer bounds),
  `parse_fmtp` (RFC 4587 §6.2 — `a=fmtp:<pt> CIF=…;QCIF=…;D=…` with
  payload-type match), `H261FmtpParams::parse_value` (RFC 4587 §6.1.1
  — semicolon-separated key=value list with CIF/QCIF MPI ∈ 1..=4
  and D ∈ {0,1} validation, duplicate-parameter rejection, forward-
  compatible unknown-parameter skip), and `negotiate_answer` (RFC
  4587 §6.2.1 offer/answer rules — per-shared-size `max(MPI)` upper
  bound, RFC 2032 QCIF=1 fallback for an offer with no picture size,
  both-sides-required Annex-D survival). The harness decodes the fuzz
  input as UTF-8 lossily, drives each parser standalone, then splits
  the input on `|` and feeds the `(offer, our)` halves to the
  negotiator; for any input that parses cleanly the formatter output
  is reparsed back through `parse_value` so a round-trip mismatch
  trips the daily run — including both §6.2.1 preference orders via
  `format_value_preferred`, whose leading token is read back through
  `parse_preference_order`.

The contract under test is the same for all five targets: every
call must *return* — no panic, no abort, no integer overflow (in
debug / ASAN builds), no out-of-bounds index, no allocator OOM.

The seed corpus for `decode_h261` (`fuzz/corpus/decode_h261/`) is
generated from the in-tree encoder and covers QCIF + CIF I-pictures
across a quantiser range (8 / 12 / 31), plus QCIF + CIF I+P pairs
that exercise motion compensation and the loop filter. The seed
corpus for `parse_rtcp_compound`
(`fuzz/corpus/parse_rtcp_compound/`) contains nine valid datagrams:
empty RR, SR with no blocks, SR with one block, RR with two blocks,
SDES CNAME, BYE with reason, APP with PING payload, and two
compound packets (RR + SDES + BYE; SR + SDES + APP). The seed
corpus for `parse_rtp_payload`
(`fuzz/corpus/parse_rtp_payload/`) contains nine RTP datagrams: a
12-byte fixed-header-only packet (CC=0, empty payload), a 16-byte
fixed-header + empty H.261 payload header boundary, a typical
V=2 CC=0 fixed header + GOB-aligned H.261 header + faked PSC
stream, the same shape with CC=3 and CC=15 CSRC lists, a non-GOB-
aligned H.261 header carrying SBIT=3 / EBIT=5 / I=V=1 / GOBN=7
/ MBAP=12 / QUANT=17 / HMVD=-7 / VMVD=11, a marker=0 mid-frame
packet, an adversarial V=1 datagram, and an adversarial CC=15
buffer truncated at 12 bytes that must surface as `ShortHeader`
rather than read out of bounds. The seed corpus for
`decode_bch_multiframe`
(`fuzz/corpus/decode_bch_multiframe/`) contains nine framed
buffers generated by the in-crate `encode_multiframe`: one and
three multiframes of pure stuffing, three multiframes of an
all-zeros / 0xC3 / pseudo-random payload, a payload-then-fill mix
that exercises the `Fi=1`-to-`Fi=0` boundary, one multiframe with
a single-bit flip inside a data field (drives the non-zero-syndrome
branch), a 4-bit-prefix-shifted stream that forces the lock-search
past `bit0 = 0`, a 2-multiframe input that falls one frame short of
the §5.4.4 lock window, and a 6-multiframe stream of `0x5A` payload
(twice the lock window). The seed corpus for `parse_sdp_fmtp`
(`fuzz/corpus/parse_sdp_fmtp/`) contains ten text buffers: the §6.2
worked-example `a=rtpmap:31 H261/90000` and `a=fmtp:31 CIF=2;QCIF=1;D=1`
lines, a dynamic-payload-type rtpmap (PT=96), a QCIF-only fmtp, a
forward-compatible fmtp with an unknown parameter, a lowercase-key
fmtp (case-insensitive match per §6.1.1), a `|`-split offer/our pair
for `negotiate_answer`, an empty-offer pair that exercises the §6.2.1
RFC 2032 QCIF=1 fallback, a malformed parameter list (`CIF=5;QCIF=0;D=2`
— every value out of range), and a non-H.261 rtpmap (`H264/90000`) the
parser must reject.

`tests/fuzz_seed_corpus.rs`, `tests/fuzz_seed_corpus_rtcp.rs`,
`tests/fuzz_seed_corpus_bch.rs`, `tests/fuzz_seed_corpus_rtp.rs`,
and `tests/fuzz_seed_corpus_sdp.rs` drive the same logic on stable
Rust against each corpus so the regular CI matrix catches a regression
in any public surface without waiting for the daily fuzz run. The
RTCP stable-CI test also drives a handful of hand-crafted adversarial
buffers — lying header length, zero-length advance, truncated
compound, SDES PRIV length overflow, BYE reason overflow, APP at the
5-bit subtype maximum, unknown PT=205. The BCH stable-CI test drives
empty / single-zero / all-ones / pseudo-random buffers, a
one-byte-short-of-lock input, an attacker-chosen 32-bit parity, and a
multi-bit deterministic injection that surfaces in `corrupted_frames`
without breaking lock. The RTP stable-CI test drives empty /
single-zero / all-ones / pseudo-random buffers, the 12-byte
fixed-header boundary, a CC=15 truncated input (must reject as
`ShortHeader` rather than read past the buffer), a V=1 rejection (must
return `FieldOverflow`), a `pack_header` ↔ `unpack_header` round-trip
on the typical-fields header, and a `depacketize` SBIT+EBIT-overflow
input (must return `EmptyPayload` rather than underflow on
`8 * data.len() - sbit - ebit`). The SDP stable-CI test drives empty /
single-zero / all-ones / pseudo-random buffers, the §6.2 worked
rtpmap + fmtp round trips, a non-H.261 rtpmap rejection, an u8
payload-type and u32 clock-rate overflow rejection on `parse_rtpmap`,
a payload-type mismatch on `parse_fmtp`, MPI-out-of-range / Annex-D
non-binary / duplicate-CIF / duplicate-QCIF / missing-`=` rejections
on `parse_value`, a forward-compatible unknown-parameter skip, the
§6.2.1 disjoint-advertisement `NoPictureSize` rejection, the §6.2.1
RFC 2032 QCIF=1 fallback, the §6.2.1 both-sides Annex-D rule, and a
formatter → parser round-trip on the canonical `(CIF=4, QCIF=1, D=1)`
shape. The parser-surface contracts stay covered even if the on-disk
corpus is corrupted.

The workflow at `.github/workflows/fuzz.yml` runs `cargo fuzz run
decode_h261 -- -max_total_time=1800` (30-minute budget) once a day
via the org-shared `crate-fuzz.yml` reusable workflow. The
`parse_rtcp_compound`, `decode_bch_multiframe`, `parse_rtp_payload`,
and `parse_sdp_fmtp` targets share the same workflow file once the
reusable workflow gains per-target sequencing; until then their
stable-CI seed tests are the primary regression guard.

```sh
cargo install cargo-fuzz
cd crates/oxideav-h261
cargo +nightly fuzz run decode_h261 -- -max_total_time=60
cargo +nightly fuzz run parse_rtcp_compound -- -max_total_time=60
cargo +nightly fuzz run decode_bch_multiframe -- -max_total_time=60
cargo +nightly fuzz run parse_rtp_payload -- -max_total_time=60
cargo +nightly fuzz run parse_sdp_fmtp -- -max_total_time=60
```

### Benchmarks

A `criterion` benchmark suite lives under `benches/`. Six targets
cover the codec pipeline plus the §5.4 outer FEC layer, the
§4.1 / §4.2 start-code scanner, and the §3.2.3 loop filter /
§3.2.2 integer-pel motion-comp primitives so future optimisation
rounds (e.g. a SIMD IDCT, a fixed-point FDCT, a precomputed
CBP-prefix table, a faster spiral+diamond ME, a table-driven BCH
parity, a SIMD start-code pre-scan, a SIMD loop filter, a
branchless edge-clamp block copy) have a baseline to A/B against:

* **`transform`** — single 8×8 block forward / inverse DCT. Four
  sub-scenarios (`fdct_intra` / `fdct_signed` / `idct_intra` /
  `idct_signed`) — the encoder and decoder hot path at the leaf.
* **`encode`** — full picture encode through the production
  `encode_intra_picture` / `H261Encoder::encode_frame` paths.
  Sub-scenarios: `encode_qcif_intra_q8` (single 176×144 I-picture,
  no ME), `encode_qcif_inter_one_q8` (a single P-picture from a
  pre-computed I recon, isolates the per-frame P cost),
  `encode_qcif_inter_chain_4_q8` (I + 3 P at quant=8 — adds the
  full rate-controller carryover), and `encode_cif_intra_q8`
  (352×288 I-only, amortises the per-MB constant).
* **`decode`** — full picture decode through `H261Decoder::send_packet`
  + `receive_frame`. Sub-scenarios mirror the encode bench
  (`decode_qcif_intra_q8`, `decode_qcif_inter_chain_4_q8`,
  `decode_cif_intra_q8`).
* **`bch`** — §5.4 BCH (511, 493) FEC layer. Sub-scenarios:
  `parity18` / `syndrome18` (one 493-bit message through the
  shift-register long-division primitives — the encoder / decoder
  inner loop), `locate_single_error` worst-case / uncorrectable
  (the §5.4.1 `t = 1` correction walk, fixed-cost 511-step), and
  `encode_multiframe` / `decode_multiframe` clean +
  one-bit-corrupted + `decode_multiframe_with_correction` on one
  full 8-frame multiframe. Headline points on the round-233
  baseline (release build, aarch64): `parity18` ≈ 350 ns / frame
  (≈ 0.7 ns / data bit), `locate_single_error` ≈ 460 ns,
  `encode_multiframe` ≈ 12 µs / multiframe, `decode_multiframe`
  ≈ 24 µs, and the §5.4.1 correcting decode adds essentially no
  overhead over the detection-only decode (within run-to-run
  noise) when at most one frame in the multiframe is corrupted.
* **`start_code`** — §4.1 / §4.2 start-code scanner
  (`find_next_start_code_bits`, `find_next_start_code`,
  `iter_start_codes`). Inner loop of every
  `H261Decoder::send_packet`, every `rtp::packetize_gob_aligned`,
  and the spec-mandated RFC 4587 §4 `rtp::depacketize` sanity
  check. Six sub-scenarios:
  `iter_start_codes::{qcif_intra_one_frame,
  cif_intra_one_frame, qcif_intra_three_frames}` (full
  elementary-stream walks; expected 1 PSC + 3 / 12 / 9 GBSCs
  respectively), `find_next_start_code::qcif_intra_first` (best
  case — PSC at bit 0), `find_next_start_code_bits::
  qcif_intra_misaligned_start` (the §4.2 packetizer's slow path
  when a GOB does not land on a byte boundary), and
  `find_next_start_code::no_start_code_in_buffer` (worst case —
  4 KiB scanned end-to-end, no hit). Headline points on the
  round-238 baseline (release build, aarch64): the bit-by-bit
  scanner clocks ≈ 295–300 MiB/s across all three full-stream
  walks; a byte-aligned hit costs ≈ 6 ns; a 3-bit misaligned hit
  costs ≈ 18 ns; the worst-case 4 KiB no-hit walk takes ≈ 13 µs.
  An eventual SIMD pre-scan over byte-aligned `0x00 0x01`
  candidates plus the bit-walk on the few near-hit windows is the
  obvious follow-up; this bench gives that change its A/B.
* **`filter_mc`** — the two per-block P-picture reconstruction
  primitives that sit *outside* the 8×8 transform: the §3.2.3
  in-loop filter (`mb::apply_loop_filter`, separable
  1/4-1/2-1/4 with 0-1-0 edge taps) and the §3.2.2 integer-pel
  motion-comp reference fetch (`mb::copy_block_integer`). The
  decoder runs both on every coded P-block, and the fuzzer's
  `decode_h261` target reaches both; `transform` only covered the
  (I)DCT, so these had no isolated baseline. Sub-scenarios:
  `loop_filter_8x8/apply_loop_filter` (one 8×8 block), and three
  motion-comp regimes — `motion_comp_8x8/copy_block_integer::
  {center, mv_nonzero, corner_clamp}` — exercising the
  no-clamp common case, a small interior MV, and the worst case
  where every sample's edge-clamp branch fires. Headline points
  on the round-287 baseline (release build, aarch64): the loop
  filter clocks ≈ 25 ns / block (≈ 2.5 Gelem/s); the integer-pel
  copy ≈ 15 ns / block (≈ 4.1 Gelem/s) interior, ≈ 14.5 ns when
  fully corner-clamped. A SIMD loop filter and a branchless
  clamp copy are the obvious follow-ups; this bench gives them
  their A/B.

Every benchmark synthesises its YUV source inline from a
deterministic striped pattern plus low-amplitude xorshift noise
(no on-disk fixtures, no third-party CLI, no `docs/` files at
bench time); the decode benches drive their input through the
in-crate encoder first so they always test against a
well-formed elementary stream.

```sh
# Compile-only smoke test (CI uses this).
cargo bench -p oxideav-h261 --no-run

# Quick fast run (~10 s total — short measurement window).
cargo bench -p oxideav-h261 -- --quick

# Full criterion run (default 5 s warm-up + 10 s measure per bench).
cargo bench -p oxideav-h261 --bench transform
cargo bench -p oxideav-h261 --bench encode
cargo bench -p oxideav-h261 --bench decode
cargo bench -p oxideav-h261 --bench bch
cargo bench -p oxideav-h261 --bench start_code

# Filter to one sub-scenario.
cargo bench -p oxideav-h261 --bench encode -- qcif_intra
cargo bench -p oxideav-h261 --bench bch -- parity18
cargo bench -p oxideav-h261 --bench start_code -- iter_start_codes
```

The bench suite is `harness = false` (criterion replaces the
default `libtest` harness), so `cargo bench --no-run` doubles as
a compile-only regression guard the CI matrix already exercises
via the `cargo build --benches`-equivalent path of the reusable
crate workflow.

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
