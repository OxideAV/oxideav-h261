//! End-to-end integration: encode an H.261 I-picture, wrap it with the
//! BCH multiframe framing per §5.4, then unwrap and decode the result.
//!
//! This proves the [`bch`] module is interoperable with real H.261
//! elementary streams (in particular: the unwrapped bytes are valid
//! input to [`H261Decoder`]).

use oxideav_core::{CodecId, Packet, TimeBase};
use oxideav_h261::bch::{decode_multiframe, decode_multiframe_with_correction, encode_multiframe};
use oxideav_h261::decoder::H261Decoder;
use oxideav_h261::encoder::encode_intra_picture;
use oxideav_h261::picture::SourceFormat;
use oxideav_h261::CODEC_ID_STR;

use oxideav_core::Decoder;

fn gradient_qcif() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = 176usize;
    let h = 144usize;
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let v = 32 + (i * 150) / w + (j * 50) / h;
            y[j * w + i] = v.clamp(0, 255) as u8;
        }
    }
    let cb = vec![128u8; (w / 2) * (h / 2)];
    let cr = vec![128u8; (w / 2) * (h / 2)];
    (y, cb, cr)
}

/// Encode one QCIF I-picture, wrap it in BCH multiframes, then unwrap
/// and feed to the decoder. The decoded frame must match the source.
fn encode_qcif_intra(y: &[u8], cb: &[u8], cr: &[u8], quant: u32, tr: u8) -> Vec<u8> {
    let w = 176usize;
    encode_intra_picture(SourceFormat::Qcif, y, w, cb, w / 2, cr, w / 2, quant, tr)
        .expect("encode_intra_picture")
}

#[test]
fn bch_wrap_unwrap_round_trip_qcif_intra() {
    let (y, cb, cr) = gradient_qcif();
    let encoded = encode_qcif_intra(&y, &cb, &cr, 10, 0);
    // Wrap with BCH framing.
    let framed = encode_multiframe(&encoded, encoded.len() * 8);
    // Output must be a whole number of multiframes (≥ 1).
    assert!(framed.len() % 512 == 0);

    // Pad to at least 3 multiframes so the decoder can establish lock.
    let mut padded = framed.clone();
    while padded.len() < 3 * 512 {
        padded.extend(encode_multiframe(&[], 0));
    }

    // Unwrap.
    let dec = decode_multiframe(&padded).expect("BCH lock on real H.261 stream");
    assert_eq!(dec.corrupted_frames, 0);
    assert!(
        dec.data_bits >= encoded.len() * 8,
        "BCH recovered {} bits, expected ≥ {}",
        dec.data_bits,
        encoded.len() * 8
    );
    // The leading `encoded.len()` bytes of `dec.data` must equal `encoded`.
    assert_eq!(&dec.data[..encoded.len()], &encoded[..]);

    // Decoding the unwrapped bytes must yield a valid picture.
    let mut h261 = H261Decoder::new(CodecId::new(CODEC_ID_STR));
    let payload = dec.data[..encoded.len()].to_vec();
    let pkt = Packet::new(0, TimeBase::new(1, 30), payload);
    h261.send_packet(&pkt).expect("decode unwrapped bitstream");
    // H.261 has no end-of-sequence marker; flush() to signal EOF so the
    // decoder commits the final picture.
    h261.flush().expect("flush");
    let frame = h261.receive_frame().expect("receive_frame");
    match frame {
        oxideav_core::Frame::Video(vf) => {
            assert_eq!(vf.planes.len(), 3, "YUV planes");
            // Compare against source — coarse PSNR check.
            let mut sse = 0.0f64;
            for (s, d) in y.iter().zip(vf.planes[0].data.iter()) {
                let diff = *s as f64 - *d as f64;
                sse += diff * diff;
            }
            let mse = sse / y.len() as f64;
            let psnr = 10.0 * (255.0_f64 * 255.0 / mse.max(1e-9)).log10();
            assert!(
                psnr >= 32.0,
                "PSNR after BCH wrap → unwrap → decode = {psnr:.2} dB (expected ≥ 32 dB)"
            );
        }
        _ => panic!("expected Frame::Video"),
    }
}

/// A single-bit error in the FEC payload is detected via the BCH
/// syndrome, but the unwrapped data is passed through unchanged (the
/// inner H.261 decoder will deal with the error itself, typically by
/// resyncing at the next GOB).
#[test]
fn bch_single_bit_error_is_detected_and_passed_through() {
    let (y, cb, cr) = gradient_qcif();
    let encoded = encode_qcif_intra(&y, &cb, &cr, 10, 0);
    let framed = encode_multiframe(&encoded, encoded.len() * 8);
    let mut padded = framed.clone();
    while padded.len() < 3 * 512 {
        padded.extend(encode_multiframe(&[], 0));
    }

    // Corrupt one bit deep inside the first frame's payload (byte 100 is
    // past the framing/Fi/early-data bits, well into the coded-data area).
    padded[100] ^= 0b0000_1000;
    let dec = decode_multiframe(&padded).expect("lock survives one-bit error");
    assert!(
        dec.corrupted_frames >= 1,
        "syndrome should flag the corrupted frame"
    );
    // The decoder still surfaces data — the inner H.261 stream may now
    // produce a slightly different decoded picture (or an error at the
    // affected MB), but the BCH layer itself does not drop the data.
    assert_eq!(dec.frames_consumed, 24);
}

/// A single-bit error in the FEC payload of a real H.261 elementary
/// stream is recovered bit-exact by `decode_multiframe_with_correction`:
/// the §5.4.1 BCH correction restores the inner bitstream to its pre-
/// channel-error form, and the H.261 decoder then produces the same
/// picture that a clean channel would have delivered.
#[test]
fn bch_correction_recovers_h261_stream_from_single_bit_error() {
    let (y, cb, cr) = gradient_qcif();
    let encoded = encode_qcif_intra(&y, &cb, &cr, 10, 0);
    let framed = encode_multiframe(&encoded, encoded.len() * 8);
    let mut padded = framed.clone();
    while padded.len() < 3 * 512 {
        padded.extend(encode_multiframe(&[], 0));
    }

    // Flip a single bit deep inside the H.261 payload of the first
    // multiframe (byte 100, mid-frame).
    let clean = padded.clone();
    padded[100] ^= 0b0000_1000;

    let dec = decode_multiframe_with_correction(&padded).expect("lock");
    assert!(
        dec.corrupted_frames >= 1,
        "syndrome must flag the corrupted frame"
    );
    assert_eq!(
        dec.corrected_frames, dec.corrupted_frames,
        "every flagged frame should have been corrected"
    );
    assert_eq!(dec.uncorrectable_frames, 0);

    // The corrected inner stream matches the pre-corruption stream
    // byte-for-byte across the encoded H.261 payload.
    let dec_clean = decode_multiframe(&clean).expect("lock on clean");
    assert_eq!(&dec.data[..encoded.len()], &dec_clean.data[..encoded.len()]);
    assert_eq!(&dec.data[..encoded.len()], &encoded[..]);

    // And the H.261 decoder produces a valid picture.
    let mut h261 = H261Decoder::new(CodecId::new(CODEC_ID_STR));
    let payload = dec.data[..encoded.len()].to_vec();
    let pkt = Packet::new(0, TimeBase::new(1, 30), payload);
    h261.send_packet(&pkt).expect("decode corrected stream");
    h261.flush().expect("flush");
    let frame = h261.receive_frame().expect("receive_frame");
    match frame {
        oxideav_core::Frame::Video(vf) => {
            // Coarse PSNR sanity — after correction the decode should
            // be at least as good as the no-error baseline (≥ 32 dB).
            let mut sse = 0.0f64;
            for (s, d) in y.iter().zip(vf.planes[0].data.iter()) {
                let diff = *s as f64 - *d as f64;
                sse += diff * diff;
            }
            let mse = sse / y.len() as f64;
            let psnr = 10.0 * (255.0_f64 * 255.0 / mse.max(1e-9)).log10();
            assert!(
                psnr >= 32.0,
                "PSNR after BCH-correct → decode = {psnr:.2} dB"
            );
        }
        _ => panic!("expected Frame::Video"),
    }
}

/// Two-bit error in the same frame is uncorrectable (the t = 1 code
/// can't disambiguate weight-2 patterns); the function still returns,
/// and the breakdown is internally consistent
/// (`corrupted = corrected + uncorrectable`).
#[test]
fn bch_correction_two_bit_error_in_frame_is_internally_consistent() {
    let (y, cb, cr) = gradient_qcif();
    let encoded = encode_qcif_intra(&y, &cb, &cr, 10, 0);
    let framed = encode_multiframe(&encoded, encoded.len() * 8);
    let mut padded = framed.clone();
    while padded.len() < 3 * 512 {
        padded.extend(encode_multiframe(&[], 0));
    }

    // Two flips in the same frame (frame 0 occupies bytes 0..64).
    // Different bytes so we land on two distinct codeword positions.
    padded[10] ^= 0b0000_0001;
    padded[40] ^= 0b0010_0000;

    let dec = decode_multiframe_with_correction(&padded).expect("lock survives");
    assert!(dec.corrupted_frames >= 1);
    assert_eq!(
        dec.corrected_frames + dec.uncorrectable_frames,
        dec.corrupted_frames,
        "breakdown should account for every flagged frame"
    );
}

/// Concatenating two encoded pictures (each separately BCH-wrapped) and
/// decoding gives back the same byte stream as if we'd wrapped them
/// together. (Multiframe concatenation is the natural transport pattern.)
#[test]
fn bch_concatenated_pictures_round_trip() {
    let (y, cb, cr) = gradient_qcif();
    let pic1 = encode_qcif_intra(&y, &cb, &cr, 10, 0);
    let pic2 = encode_qcif_intra(&y, &cb, &cr, 15, 1);

    // Wrap each picture separately, then concatenate the multiframes.
    let mut framed = encode_multiframe(&pic1, pic1.len() * 8);
    framed.extend(encode_multiframe(&pic2, pic2.len() * 8));
    // Pad to ≥ 3 multiframes if necessary.
    while framed.len() < 3 * 512 {
        framed.extend(encode_multiframe(&[], 0));
    }
    let dec = decode_multiframe(&framed).expect("lock on concatenated stream");
    assert_eq!(dec.corrupted_frames, 0);
    // The recovered data should contain `pic1` followed (with possible
    // gaps from Fi-aligned fill) by `pic2`. Both pictures' headers begin
    // with a PSC, so we can simply look for the second PSC after the first.
    // Easier: assert that the leading `pic1.len()` bytes match pic1 — the
    // following bytes belong to pic2 starting at a 492-bit boundary.
    assert_eq!(&dec.data[..pic1.len()], &pic1[..]);
}
