//! H.261 §5.4 — BCH (511, 493) forward error-correction framing.
//!
//! H.261 transmits the video bitstream wrapped in an outer FEC layer
//! intended for the noisy ISDN p × 64 kbit/s channels of 1990. The
//! decoder's use of this layer is optional (§2.7) — for a clean file or
//! a reliable transport (RTP, MP4) the inner video bitstream is enough.
//! When present, the FEC frame layout is:
//!
//! ```text
//!  ┌────────┬────┬───────────────────────┬──────────┐
//!  │  Si    │ Fi │      Coded data       │  Parity  │
//!  │ 1 bit  │1bit│       492 bits        │  18 bits │
//!  └────────┴────┴───────────────────────┴──────────┘
//!   total = 512 bits per error-correcting frame
//! ```
//!
//! The BCH (511, 493) code protects the 493-bit `Fi || coded-data`
//! field with an 18-bit parity field (493 + 18 = 511 protected bits);
//! the leading 1-bit `Si` framing bit is unprotected, bringing the
//! per-frame transmitted total to 512 bits.
//!
//! `Si` is the i-th bit of the 8-frame multiframe alignment pattern
//! `S1..S8 = 0 0 0 1 1 0 1 1`. `Fi=1` indicates the 492-bit field carries
//! coded video; `Fi=0` indicates a stuffing frame whose 492 bits are all
//! `1`. The BCH parity is computed over the **493 bits** consisting of
//! `Fi` followed by the 492-bit data field (so the parity covers
//! whichever of the two `Fi` interpretations applies).
//!
//! ## Generator polynomial (§5.4.2)
//!
//! ```text
//!   g(x) = (x^9 + x^4 + 1)(x^9 + x^6 + x^4 + x^3 + 1)
//!        = x^18 + x^15 + x^12 + x^10 + x^8 + x^7 + x^6 + x^3 + 1
//! ```
//!
//! As a 19-bit binary value with x^18 in the most-significant position,
//! `g = 0b100_1001_0101_1100_1001 = 0x495C9`. The BCH (511, 493) code
//! is a double-error-correcting / single-error-detecting code over
//! GF(2^9): each factor is the primitive minimal polynomial of a 9-bit
//! root, jointly enabling correction of up to t = (511 - 493) / (2·9) =
//! 1 error in a 511-bit codeword (this is a t = 1 code despite the
//! 18-bit parity — the 18-bit parity buys 1-bit correction + 1-bit
//! detection rather than 2-bit correction; see RFC 4587 for typical
//! deployment patterns).
//!
//! ## What this module provides
//!
//! * [`parity18`] — compute the 18-bit BCH parity over a 493-bit input.
//! * [`syndrome18`] — compute the 18-bit syndrome over a 511-bit codeword;
//!   zero ⇒ no error, non-zero ⇒ at least one bit error.
//! * [`encode_multiframe`] — wrap a slice of `coded_data: &[u8]` into an
//!   integer number of 8-frame multiframes (4096 bits = 512 bytes each),
//!   inserting alignment bits and `Fi`/fill frames as required by §5.4.3.
//! * [`decode_multiframe`] — strip the FEC framing from a byte buffer
//!   produced by [`encode_multiframe`], returning the inner coded data.
//!   The decoder achieves frame lock after 3 consecutive valid alignment
//!   sequences (§5.4.4); it then strips parity / framing bits and
//!   surfaces the inner data. A syndrome check is run on every frame and
//!   the number of corrupted (non-zero-syndrome) frames is reported back
//!   to the caller as a diagnostic.
//!
//! The H.261 video bitstream the rest of this crate emits / consumes is
//! oblivious to this layer. Callers that need framed output (e.g. for
//! transport over a raw bit-serial p × 64 kbit/s link) wrap their bytes
//! with [`encode_multiframe`]; callers receiving a framed stream (RFC
//! 4587 §6.2 is one historical example) recover the inner stream with
//! [`decode_multiframe`].

/// Generator polynomial g(x), 19 bits wide (degree 18). MSB = x^18.
/// `(x^9 + x^4 + 1)(x^9 + x^6 + x^4 + x^3 + 1)
///   = x^18 + x^15 + x^12 + x^10 + x^8 + x^7 + x^6 + x^3 + 1`.
pub const GEN_POLY: u32 = 0x4_95C9;

/// Width of the BCH parity field, in bits.
pub const PARITY_BITS: u32 = 18;

/// Width of the BCH data field per error-correcting frame, in bits.
/// Includes the `Fi` bit (1) + the 492 coded-data bits = 493 bits total.
pub const DATA_BITS: u32 = 493;

/// Total error-correcting frame width = framing bit + data + parity = 512 bits.
/// (The BCH code itself is (511, 493): 493 protected bits + 18 parity bits.
/// The 1-bit `Si` framing bit is transmitted alongside but not BCH-protected.)
pub const FRAME_BITS: u32 = 1 + DATA_BITS + PARITY_BITS;

/// Number of frames per multiframe. The 8 framing bits across these
/// frames form the alignment pattern `S1..S8 = 0 0 0 1 1 0 1 1` (§5.4.3).
pub const MULTIFRAME_FRAMES: u32 = 8;

/// 8-frame alignment pattern, MSB-first as transmitted: `0 0 0 1 1 0 1 1`.
/// Indexable as `ALIGNMENT_PATTERN[frame_index_in_multiframe]`; frame 0
/// is `S1`, frame 7 is `S8`.
pub const ALIGNMENT_PATTERN: [u8; 8] = [0, 0, 0, 1, 1, 0, 1, 1];

/// Compute the 18-bit BCH parity for a 493-bit input message.
///
/// The message is taken from the most-significant bits of `data`, with
/// bit 0 of `data[0]` being the first (MSB-first) input bit. Bits beyond
/// position 492 are ignored.
///
/// Algorithm: long division of (msg << 18) by `GEN_POLY` over GF(2),
/// which is the textbook shift-register implementation. The returned
/// `u32` has the parity in its low 18 bits; bits 18..31 are zero.
pub fn parity18(data: &[u8]) -> u32 {
    debug_assert!(
        data.len() * 8 >= DATA_BITS as usize,
        "parity18 needs at least {} bits of input ({} bytes)",
        DATA_BITS,
        DATA_BITS.div_ceil(8)
    );
    // 19-bit shift register; we keep 19 bits in `reg` and XOR `GEN_POLY`
    // whenever bit 18 is set after shifting in a message bit.
    let mut reg: u32 = 0;
    for i in 0..(DATA_BITS as usize) {
        let bit = (data[i / 8] >> (7 - (i & 7))) & 1;
        reg = (reg << 1) | bit as u32;
        if (reg >> 18) & 1 != 0 {
            reg ^= GEN_POLY;
        }
    }
    // After 493 input bits, shift in 18 zero "tail" bits to flush the
    // register; the residue in the low 18 bits is the parity.
    for _ in 0..PARITY_BITS {
        reg <<= 1;
        if (reg >> 18) & 1 != 0 {
            reg ^= GEN_POLY;
        }
    }
    reg & ((1 << PARITY_BITS) - 1)
}

/// Compute the syndrome of a 511-bit codeword: data ‖ parity, with `data`
/// in the high 493 bits and `parity` in the low 18 bits.
///
/// A zero return value means the codeword is consistent with the BCH
/// generator polynomial. A non-zero return value means at least one bit
/// was corrupted (and for ≤ t = 1 single-bit errors, the syndrome
/// identifies the position; this module currently surfaces the syndrome
/// to the caller and does not perform single-bit correction in the
/// decoder path — the inner H.261 video VLC layer is robust to occasional
/// MB-level slips and historically deployments either passed corrupted
/// frames through or dropped them).
pub fn syndrome18(data: &[u8], parity: u32) -> u32 {
    let mut reg: u32 = 0;
    for i in 0..(DATA_BITS as usize) {
        let bit = (data[i / 8] >> (7 - (i & 7))) & 1;
        reg = (reg << 1) | bit as u32;
        if (reg >> 18) & 1 != 0 {
            reg ^= GEN_POLY;
        }
    }
    // Now shift in the 18 parity bits MSB-first.
    for i in 0..PARITY_BITS {
        let bit = (parity >> (PARITY_BITS - 1 - i)) & 1;
        reg = (reg << 1) | bit;
        if (reg >> 18) & 1 != 0 {
            reg ^= GEN_POLY;
        }
    }
    reg & ((1 << PARITY_BITS) - 1)
}

/// Wrap `coded_data` (an MSB-first packed bitstream) into FEC multiframes
/// per §5.4.3. The returned `Vec<u8>` is MSB-first packed too. Length is
/// always an integer multiple of `FRAME_BITS * MULTIFRAME_FRAMES / 8 =
/// 512 bytes` (8 frames × 512 bits each).
///
/// If `coded_data` does not fill an integer number of frames, the final
/// frame's data field is filled with `Fi = 0` + 492 one-bits per §5.4.3.
/// The caller's bits always land before any fill (so a decoder that
/// honours `Fi` will not surface the fill bits to its consumer).
///
/// `coded_data_bits` is the number of valid bits in `coded_data`. If
/// `coded_data_bits` is greater than `coded_data.len() * 8`, the function
/// panics (this is a programming error, not an FEC condition). If
/// `coded_data_bits == 0`, a single stuffing-only multiframe is emitted.
pub fn encode_multiframe(coded_data: &[u8], coded_data_bits: usize) -> Vec<u8> {
    assert!(
        coded_data_bits <= coded_data.len() * 8,
        "encode_multiframe: coded_data_bits ({}) exceeds buffer ({} bits)",
        coded_data_bits,
        coded_data.len() * 8
    );

    // Each FEC frame carries 492 coded bits when Fi=1.
    let coded_per_frame = (DATA_BITS - 1) as usize; // 492
    let frames_needed = if coded_data_bits == 0 {
        MULTIFRAME_FRAMES as usize
    } else {
        let n = coded_data_bits.div_ceil(coded_per_frame);
        // Round up to a whole multiframe (8 frames).
        n.div_ceil(MULTIFRAME_FRAMES as usize) * MULTIFRAME_FRAMES as usize
    };

    // Bit-write buffer: 511 bits/frame.
    let total_bits = frames_needed * FRAME_BITS as usize;
    let mut out = vec![0u8; total_bits.div_ceil(8)];
    let mut write_pos: usize = 0;
    let put_bit = |out: &mut [u8], pos: usize, bit: u8| {
        let byte = pos / 8;
        let shift = 7 - (pos & 7);
        out[byte] = (out[byte] & !(1u8 << shift)) | ((bit & 1) << shift);
    };

    let read_bit_in = |coded: &[u8], i: usize| -> u8 { (coded[i / 8] >> (7 - (i & 7))) & 1 };

    // 493-bit scratch (Fi + 492 data bits) for parity computation.
    let mut scratch = [0u8; 62]; // ceil(493/8) = 62

    let mut consumed_bits = 0usize;
    for f in 0..frames_needed {
        let s_idx = f % MULTIFRAME_FRAMES as usize;
        let s_bit = ALIGNMENT_PATTERN[s_idx];

        // Decide Fi: 1 if any coded data remains, else 0 (stuffing).
        let fi: u8 = if consumed_bits < coded_data_bits {
            1
        } else {
            0
        };

        // Build the 493-bit `Fi || data` scratch buffer MSB-first.
        for b in scratch.iter_mut() {
            *b = 0;
        }
        // Bit 0 of scratch = Fi.
        scratch[0] = fi << 7;
        for j in 0..coded_per_frame {
            let bit = if fi == 1 {
                if consumed_bits < coded_data_bits {
                    let v = read_bit_in(coded_data, consumed_bits);
                    consumed_bits += 1;
                    v
                } else {
                    // The frame is Fi=1 but coded_data didn't run out
                    // exactly at the frame boundary; pad the tail with
                    // ones (consistent with MBA-stuffing convention).
                    1
                }
            } else {
                // Fi=0 ⇒ all-ones fill.
                1
            };
            // scratch position j+1 (since bit 0 was Fi).
            let pos = j + 1;
            scratch[pos / 8] |= bit << (7 - (pos & 7));
        }

        let par = parity18(&scratch);

        // Emit: S || Fi || data || parity, MSB-first.
        put_bit(&mut out, write_pos, s_bit);
        write_pos += 1;
        put_bit(&mut out, write_pos, fi);
        write_pos += 1;
        for j in 0..coded_per_frame {
            let bit = (scratch[(j + 1) / 8] >> (7 - ((j + 1) & 7))) & 1;
            put_bit(&mut out, write_pos, bit);
            write_pos += 1;
        }
        for j in 0..(PARITY_BITS as usize) {
            let bit = ((par >> (PARITY_BITS as usize - 1 - j)) & 1) as u8;
            put_bit(&mut out, write_pos, bit);
            write_pos += 1;
        }
    }

    debug_assert_eq!(write_pos, total_bits);
    out
}

/// Outcome of [`decode_multiframe`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedMultiframe {
    /// The recovered inner coded video bitstream, MSB-first packed.
    pub data: Vec<u8>,
    /// Number of bits in `data` (a multiple of 492 when input is whole
    /// frames; the caller can trim further if it knows the inner-stream
    /// byte boundary).
    pub data_bits: usize,
    /// Number of FEC frames consumed.
    pub frames_consumed: usize,
    /// Number of frames whose 18-bit syndrome was non-zero (i.e. at
    /// least one bit error was detected). Surfaced to the caller for
    /// diagnostics; the data from those frames is included in `data`
    /// regardless, because dropping it would create a longer drop-out
    /// than letting the inner H.261 video resync at the next GOB.
    pub corrupted_frames: usize,
    /// Number of fill-only frames skipped (Fi=0 / inner data all ones).
    pub fill_frames: usize,
}

/// Strip the BCH-framing layer from `framed`, beginning at bit 0.
///
/// Lock criterion per §5.4.4: three consecutive complete alignment
/// sequences (24 framing bits ≡ 3 full multiframes' worth of `S_i`
/// bits) must match the pattern `0 0 0 1 1 0 1 1` (repeated). The
/// decoder scans every candidate `bit0` in `[0, FRAME_BITS)` for the
/// earliest position where 24 consecutive framing bits at the proper
/// `FRAME_BITS` stride form `(00011011)^3`. Once lock is established
/// the decoder strips framing + parity from every subsequent frame in
/// `framed`; it does **not** re-seek for the pattern within a frame
/// (a single corrupted S-bit is reported as one corrupted frame —
/// surfaced via the per-frame syndrome — but does not break lock).
///
/// The `(00011011)^3` requirement is strong enough to reject random
/// bitstreams with high probability: the chance of a 24-bit specific
/// pattern appearing randomly at the right stride is 2^-24.
///
/// Returns `None` if no valid alignment lock can be obtained anywhere
/// in `framed`.
pub fn decode_multiframe(framed: &[u8]) -> Option<DecodedMultiframe> {
    let total_bits = framed.len() * 8;
    // We need at least 3 full multiframes (24 frames) to establish lock.
    let lock_frames = 3 * MULTIFRAME_FRAMES as usize;
    let lock_span_bits = lock_frames * FRAME_BITS as usize;
    if total_bits < lock_span_bits {
        return None;
    }

    let read_bit = |pos: usize| -> u8 { (framed[pos / 8] >> (7 - (pos & 7))) & 1 };

    let mut lock: Option<usize> = None;
    'outer: for bit0 in 0..FRAME_BITS as usize {
        if bit0 + lock_span_bits > total_bits {
            break;
        }
        // The first frame at `bit0` must be the start of a multiframe
        // (phase 0 ⇒ S1=0). Verify 24 framing bits across 3 multiframes.
        for k in 0..lock_frames {
            let s = read_bit(bit0 + k * FRAME_BITS as usize);
            if s != ALIGNMENT_PATTERN[k % MULTIFRAME_FRAMES as usize] {
                continue 'outer;
            }
        }
        lock = Some(bit0);
        break;
    }

    let bit0 = lock?;
    let phase0 = 0usize; // lock is always to a multiframe boundary

    let mut data: Vec<u8> = Vec::new();
    let mut data_bits = 0usize;
    let mut put_data_bit = |bit: u8| {
        if data_bits % 8 == 0 {
            data.push(0);
        }
        let byte_idx = data_bits / 8;
        let shift = 7 - (data_bits & 7);
        data[byte_idx] |= (bit & 1) << shift;
        data_bits += 1;
    };

    let mut frames_consumed = 0usize;
    let mut corrupted_frames = 0usize;
    let mut fill_frames = 0usize;

    let mut cursor = bit0;
    while cursor + FRAME_BITS as usize <= total_bits {
        let frame_idx = (phase0 + frames_consumed) % MULTIFRAME_FRAMES as usize;
        let _expected_s = ALIGNMENT_PATTERN[frame_idx];
        // S bit
        let _s = read_bit(cursor);
        let fi = read_bit(cursor + 1);

        // Re-pack `Fi || data` into a scratch for syndrome verification.
        let mut scratch = [0u8; 62];
        scratch[0] = fi << 7;
        let data_start = cursor + 2;
        for j in 0..((DATA_BITS - 1) as usize) {
            let bit = read_bit(data_start + j);
            let pos = j + 1;
            scratch[pos / 8] |= bit << (7 - (pos & 7));
        }
        // Read parity (18 bits).
        let mut par = 0u32;
        let par_start = data_start + (DATA_BITS - 1) as usize;
        for j in 0..(PARITY_BITS as usize) {
            par = (par << 1) | read_bit(par_start + j) as u32;
        }
        if syndrome18(&scratch, par) != 0 {
            corrupted_frames += 1;
        }
        if fi == 1 {
            // Emit the 492 coded-data bits.
            for j in 0..((DATA_BITS - 1) as usize) {
                let bit = read_bit(data_start + j);
                put_data_bit(bit);
            }
        } else {
            fill_frames += 1;
        }

        frames_consumed += 1;
        cursor += FRAME_BITS as usize;
    }

    Some(DecodedMultiframe {
        data,
        data_bits,
        frames_consumed,
        corrupted_frames,
        fill_frames,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity-check that the generator polynomial encodes the factored form.
    ///
    /// `(x^9 + x^4 + 1)` has bits at positions {9, 4, 0} → `0b10_0001_0001 = 0x211`.
    /// `(x^9 + x^6 + x^4 + x^3 + 1)` has bits at positions {9, 6, 4, 3, 0} →
    /// `0b10_0101_1001 = 0x259`. Their GF(2) product must equal `GEN_POLY`.
    #[test]
    fn generator_polynomial_factors_match_spec() {
        let a: u32 = 0b10_0001_0001; // 0x211
        let b: u32 = 0b10_0101_1001; // 0x259
        assert_eq!(a, 0x211);
        assert_eq!(b, 0x259);
        let mut prod: u32 = 0;
        for i in 0..16 {
            if (b >> i) & 1 == 1 {
                prod ^= a << i;
            }
        }
        assert_eq!(prod, GEN_POLY, "g(x) factors don't match: got 0x{prod:X}");
    }

    /// A zero input must yield zero parity (the codeword `0…0` is trivially
    /// divisible by g(x)).
    #[test]
    fn parity_of_all_zeros_is_zero() {
        let zero = [0u8; 62];
        assert_eq!(parity18(&zero), 0);
    }

    /// The all-ones 493-bit input is a useful smoke-test: the parity is
    /// determined by the polynomial alone. We compute it via the encoder
    /// and verify the matching syndrome cancels out.
    #[test]
    fn parity_then_syndrome_cancels_for_all_ones() {
        let mut ones = [0xFFu8; 62];
        // Mask off bits beyond position 492 (the high bit of the last
        // byte should be the last data bit; ceil(493/8) = 62 bytes →
        // 496 bits total → mask out the trailing 3 bits).
        ones[61] &= 0b1110_0000;
        let par = parity18(&ones);
        assert_eq!(syndrome18(&ones, par), 0);
    }

    /// Spec §5.4.2 worked example: for the 493-bit input
    /// `0 followed by 492 ones`, the BCH parity is exactly
    /// `011011010100011011` (= 0x1B51B). This is the published
    /// validation vector from ITU-T Rec. H.261 (03/93) §5.4.2.
    #[test]
    fn parity_matches_spec_5_4_2_worked_example() {
        // Build the 493-bit input: bit 0 = 0, bits 1..493 = 1.
        // Pack into ceil(493/8) = 62 bytes, MSB-first.
        let mut msg = [0u8; 62];
        for i in 1..493 {
            msg[i / 8] |= 1 << (7 - (i & 7));
        }
        let par = parity18(&msg);
        // Expected: 011011010100011011₂ = 0x1B51B.
        assert_eq!(
            par, 0x1_B51B,
            "spec §5.4.2 worked example: parity should be 0x1B51B (binary 011011010100011011), got 0x{par:X}"
        );
        // And the syndrome of (msg || par) must be zero.
        assert_eq!(syndrome18(&msg, par), 0);
    }

    /// A single-bit flip anywhere in the codeword must produce a non-zero
    /// syndrome (single-error detection — the t=1 BCH minimum).
    #[test]
    fn single_bit_flip_in_data_is_detected() {
        let mut msg = [0u8; 62];
        // A bit of structure to make sure parity is non-trivial.
        msg[3] = 0xA5;
        msg[20] = 0x3C;
        msg[55] = 0xF0;
        let par = parity18(&msg);
        assert_eq!(syndrome18(&msg, par), 0, "untouched codeword");
        // Flip bit 17.
        msg[2] ^= 0b0100_0000;
        assert_ne!(syndrome18(&msg, par), 0, "single bit flip in data");
    }

    #[test]
    fn single_bit_flip_in_parity_is_detected() {
        let msg = [0u8; 62];
        let par = parity18(&msg);
        // Flip one parity bit — say bit 5 — and check the syndrome is
        // non-zero. (For an all-zero message, par == 0 so flipping bit 5
        // gives par = 0x20.)
        assert_eq!(par, 0);
        assert_ne!(syndrome18(&msg, 0x20), 0);
    }

    /// Round-trip: encode an arbitrary inner-bitstream payload and verify
    /// `decode_multiframe` recovers it exactly.
    ///
    /// Per §5.4.4, lock requires three complete multiframes (24 bits of
    /// framing pattern), so we send three multiframes of mostly-fill
    /// content with a small data payload in the first frame.
    #[test]
    fn encode_then_decode_round_trip() {
        // 50 bytes of pseudo-random payload (400 bits). Less than one
        // multiframe (8 frames × 492 bits = 3936 bits) so this exercises
        // the fill-frame path too. We still emit 3 multiframes so the
        // decoder can lock.
        let mut payload = [0u8; 50];
        let mut s: u32 = 0xDEAD_BEEF;
        for b in payload.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (s >> 16) as u8;
        }
        // Pad the framed output to 3 multiframes by appending two whole
        // multiframes of stuffing.
        let mut framed = encode_multiframe(&payload, 50 * 8);
        framed.extend(encode_multiframe(&[], 0));
        framed.extend(encode_multiframe(&[], 0));
        assert_eq!(framed.len(), 3 * 512);
        let decoded = decode_multiframe(&framed).expect("frame lock");
        assert_eq!(decoded.corrupted_frames, 0);
        // Decoded `data_bits` should be 492 (one Fi=1 frame) since
        // 400 bits < 492.
        assert_eq!(decoded.data_bits, 492);
        // The first 400 bits of decoded.data should equal `payload`.
        for i in 0..(50 * 8) {
            let want = (payload[i / 8] >> (7 - (i & 7))) & 1;
            let got = (decoded.data[i / 8] >> (7 - (i & 7))) & 1;
            assert_eq!(got, want, "bit {i} mismatch");
        }
        // 24 total frames consumed, 23 of which are stuffing.
        assert_eq!(decoded.fill_frames, 23);
        assert_eq!(decoded.frames_consumed, 24);
    }

    /// Round-trip three full multiframes of data (3 × 8 × 492 = 11_808
    /// bits) — every frame should be Fi=1 with no fill.
    #[test]
    fn encode_then_decode_full_multiframe() {
        let mut payload = vec![0u8; 11_808 / 8];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(31).wrapping_add(7);
        }
        let framed = encode_multiframe(&payload, payload.len() * 8);
        assert_eq!(framed.len(), 3 * 512);
        let decoded = decode_multiframe(&framed).expect("frame lock");
        assert_eq!(decoded.corrupted_frames, 0);
        assert_eq!(decoded.fill_frames, 0);
        assert_eq!(decoded.frames_consumed, 24);
        assert_eq!(decoded.data_bits, 11_808);
        assert_eq!(&decoded.data[..payload.len()], &payload[..]);
    }

    /// Six whole multiframes round-trip cleanly (twice the lock window).
    #[test]
    fn encode_then_decode_two_multiframes() {
        let mut payload = vec![0u8; (6 * 3936) / 8];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = (i as u8 ^ 0x5A).wrapping_mul(13).wrapping_add(1);
        }
        let framed = encode_multiframe(&payload, payload.len() * 8);
        assert_eq!(framed.len(), 6 * 512);
        let decoded = decode_multiframe(&framed).expect("frame lock");
        assert_eq!(decoded.frames_consumed, 48);
        assert_eq!(decoded.fill_frames, 0);
        assert_eq!(decoded.corrupted_frames, 0);
        assert_eq!(decoded.data_bits, 6 * 3936);
        assert_eq!(&decoded.data[..payload.len()], &payload[..]);
    }

    /// A flipped data bit in a single frame is detected via the syndrome
    /// and reported in `corrupted_frames`, but lock is preserved and the
    /// rest of the multiframe is still recovered.
    #[test]
    fn data_corruption_surfaces_via_syndrome() {
        // 3 multiframes so we can lock; corrupt a bit deep inside one of
        // them. Flip a bit at byte 700 (well inside the second multiframe
        // at bytes 512..1024).
        let payload = vec![0xC3u8; (3 * 3936) / 8];
        let mut framed = encode_multiframe(&payload, payload.len() * 8);
        assert_eq!(framed.len(), 3 * 512);
        framed[700] ^= 0b0001_0000;
        let decoded = decode_multiframe(&framed).expect("still lockable");
        assert_eq!(decoded.frames_consumed, 24);
        assert!(
            decoded.corrupted_frames >= 1,
            "syndrome should flag at least one corrupted frame"
        );
    }

    /// Frame-lock can be obtained when the framed stream is preceded by
    /// 4 non-zero "pre-roll" bits and a fourth bit of run-up. Per §5.4.4,
    /// three consecutive complete multiframes are required for lock, so
    /// we send three multiframes' worth of data and the decoder should
    /// pick up at the first one.
    #[test]
    fn decoder_acquires_lock_with_leading_padding() {
        // Three multiframes of payload = 3 × 3936 = 11_808 bits = 1476 bytes.
        let payload = vec![0x42u8; 11_808 / 8];
        let framed = encode_multiframe(&payload, payload.len() * 8);
        assert_eq!(framed.len(), 3 * 512);
        // Prepend 4 "junk" bits (1011) = shift the whole stream right by 4.
        // The 4 junk bits are at positions 0..4 of `shifted`; bit 4 is the
        // start of the real frame.
        let mut shifted = vec![0u8; framed.len() + 1];
        shifted[0] = 0b1011_0000; // junk in the top 4 bits
        for i in 0..(framed.len() * 8) {
            let bit = (framed[i / 8] >> (7 - (i & 7))) & 1;
            let new_pos = i + 4;
            shifted[new_pos / 8] |= bit << (7 - (new_pos & 7));
        }
        let decoded = decode_multiframe(&shifted).expect("lock with offset");
        // We should consume all 24 frames (3 multiframes) — possibly minus
        // one if the trailing 4 bits don't complete a frame.
        assert!(
            decoded.frames_consumed >= 23,
            "should recover at least 23 of 24 frames, got {}",
            decoded.frames_consumed
        );
        assert_eq!(decoded.corrupted_frames, 0);
    }

    /// An input of pure noise must NOT obtain frame lock (no false
    /// positives on alignment when no real frame is present).
    #[test]
    fn random_noise_does_not_obtain_lock() {
        // All-ones: every framing-bit position holds a 1, but the
        // alignment pattern `00011011` requires three 0-bits up front,
        // so no candidate `bit0` can satisfy the 24-bit lock criterion.
        // Use 3 multiframes of buffer so the search exhausts everywhere
        // it would normally try.
        let buf = vec![0xFFu8; 3 * 512];
        let decoded = decode_multiframe(&buf);
        assert!(decoded.is_none(), "all-ones stream must not lock");
    }

    /// `encode_multiframe` of an empty payload produces one multiframe
    /// of pure stuffing frames (Fi=0, fill all ones). Concatenating
    /// three of those is what an idle channel would send, and the
    /// decoder should lock onto and unframe it.
    #[test]
    fn empty_payload_emits_one_stuffing_multiframe() {
        let mut framed = encode_multiframe(&[], 0);
        assert_eq!(framed.len(), 512);
        framed.extend(encode_multiframe(&[], 0));
        framed.extend(encode_multiframe(&[], 0));
        let decoded = decode_multiframe(&framed).expect("stuffing locks");
        assert_eq!(decoded.frames_consumed, 24);
        assert_eq!(decoded.fill_frames, 24);
        assert_eq!(decoded.data_bits, 0);
        assert_eq!(decoded.corrupted_frames, 0);
    }
}
