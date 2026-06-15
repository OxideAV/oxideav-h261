//! End-to-end §5.3 + Annex C codec-delay measurement, driven through the
//! real H.261 encoder and decoder.
//!
//! This exercises the [`annex_c`] measurement helper against actual coded
//! bytes: the §C identification mark is stamped into the first GOB's first
//! block of each input frame (point A), the frames are encoded and decoded,
//! and the mark transition is re-detected at the decoder output (point C)
//! at the §C mid-level threshold. The measured delay is then averaged by
//! the [`DelayMeter`]. Because this codec's `encode_frame`/`receive_frame`
//! path is synchronous (one input frame in → one decoded frame out, no
//! reordering and no pipeline latency), the A→C delay measured here is
//! zero frame intervals — but the *mechanism* (stamp → encode → decode →
//! detect → pair → average) is the Annex C procedure end-to-end.

use oxideav_core::{CodecId, Decoder, Packet, TimeBase};
use oxideav_h261::annex_c::{
    DelayMeter, MarkDetector, MarkEvent, MarkGenerator, MarkLevels, MarkState, SequenceRequirements,
};
use oxideav_h261::decoder::H261Decoder;
use oxideav_h261::encoder::H261Encoder;
use oxideav_h261::picture::SourceFormat;

/// A modest QCIF sequence spanning four §C mark intervals (so we observe
/// three mark *changes*). 97 frames per interval × 4 = 388 frames would be
/// a real run; to keep the test fast we use a shorter mark interval offset
/// by driving the generator on a compressed frame schedule but mapping the
/// *logical* frame index through it. We feed enough frames to cross at
/// least two real 97-frame boundaries by indexing the generator at the
/// true input-frame number.
fn striped_qcif(seed: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = SourceFormat::Qcif.dimensions();
    let (w, h) = (w as usize, h as usize);
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            // A moving stripe pattern (a "typical moving scene") so the
            // coder produces real INTER content frame to frame.
            let v = 40 + ((i + j + seed as usize * 3) % 160);
            y[j * w + i] = v as u8;
        }
    }
    let cb = vec![128u8; (w / 2) * (h / 2)];
    let cr = vec![128u8; (w / 2) * (h / 2)];
    (y, cb, cr)
}

/// Tracks the point-C (decoder output) mark state across the decoded
/// stream and records each transition into a [`DelayMeter`].
///
/// Decoded frames come out in input order (this codec does no
/// reordering), so the running output-frame counter is the §C frame-
/// interval index at point C.
#[derive(Default)]
struct PointCTracker {
    out_frame: u32,
    prev_state: Option<MarkState>,
}

impl PointCTracker {
    fn drain(&mut self, dec: &mut H261Decoder, det: &MarkDetector, meter: &mut DelayMeter) {
        loop {
            match dec.receive_frame() {
                Ok(oxideav_core::Frame::Video(vf)) => {
                    let y_plane = &vf.planes[0];
                    let c_state = det.sample(&y_plane.data, y_plane.stride);
                    if self.prev_state != Some(c_state) {
                        if self.prev_state.is_some() {
                            meter.record_c(MarkEvent {
                                state: c_state,
                                frame: self.out_frame,
                            });
                        }
                        self.prev_state = Some(c_state);
                    }
                    self.out_frame += 1;
                }
                Ok(_) => continue,
                Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
                Err(e) => panic!("decode error: {e:?}"),
            }
        }
    }
}

#[test]
fn annex_c_delay_measurement_round_trips_through_codec() {
    let levels = MarkLevels::default();
    let gen = MarkGenerator::new(SourceFormat::Qcif, levels);
    let det = MarkDetector::new(levels);

    let mut enc = H261Encoder::new(SourceFormat::Qcif, 10).with_intra_period(0);
    let mut dec = H261Decoder::new(CodecId::new(oxideav_h261::CODEC_ID_STR));

    let (w, _h) = SourceFormat::Qcif.dimensions();
    let stride = w as usize;

    let mut meter = DelayMeter::new();
    let mut prev_b_state: Option<MarkState> = None;
    let mut c_track = PointCTracker::default();

    // Drive enough frames to cross two real 97-frame mark boundaries.
    let n_frames: u32 = 200;

    // Phase 1: build the §C test sequence at point A, recording the mark
    // changes there and the matching point-B (channel-output) instants.
    // H.261 has no end-of-sequence marker, so we collect the whole coded
    // sequence and feed it to the decoder in one stream below, flushing to
    // release the final picture — mirroring a real Annex C run that taps a
    // continuous channel at point B and a continuous display at point C.
    for frame in 0..n_frames {
        let (mut y, cb, cr) = striped_qcif((frame % 7) as u8);

        // Point A: stamp the §C mark for this input frame, and record the
        // change instant when the mark toggles.
        let a_state = gen.state_for_frame(frame);
        gen.stamp(frame, &mut y, stride);
        if gen.is_change_frame(frame) {
            meter.record_a(MarkEvent {
                state: a_state,
                frame,
            });
            // Point B (channel output): in this synchronous coder the mark
            // is observable at B on the same frame it enters A. We tap the
            // pre-encode luma (the encoder reproduces it faithfully at
            // quant=10 on a flat block) to stand in for the channel tap.
            let b_state = det.sample(&y, stride);
            if prev_b_state != Some(b_state) {
                meter.record_b(MarkEvent {
                    state: b_state,
                    frame,
                });
                prev_b_state = Some(b_state);
            }
        } else if prev_b_state.is_none() {
            prev_b_state = Some(det.sample(&y, stride));
        }

        // Encode this picture and feed it straight into the decoder. Each
        // picture's data is only released by the decoder once it sees the
        // *next* picture's start code (or a flush), so we drain after each
        // send and account for the one-picture pipelining at flush time.
        let bytes = enc
            .encode_frame(&y, stride, &cb, 88, &cr, 88)
            .expect("encode_frame");
        let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
        dec.send_packet(&pkt).expect("send_packet");
        c_track.drain(&mut dec, &det, &mut meter);
    }
    // Flush to release the final picture, then drain the tail.
    dec.flush().expect("flush");
    c_track.drain(&mut dec, &det, &mut meter);

    // We crossed boundaries at frame 97 (→High) and 194 (→Low): two A
    // changes recorded.
    assert_eq!(meter.point_a.len(), 2, "expected two §C mark changes");
    assert_eq!(meter.point_a[0].frame, 97);
    assert_eq!(meter.point_a[0].state, MarkState::High);
    assert_eq!(meter.point_a[1].frame, 194);
    assert_eq!(meter.point_a[1].state, MarkState::Low);

    let summary = meter.summary();
    // Encoder delay (A→B) is zero in this synchronous tap.
    assert_eq!(summary.encoder_frames(), Some(0.0));
    // Decoder delay (B→C) is zero frame intervals: the synchronous codec
    // emits the decoded mark on the same frame it was coded.
    assert_eq!(summary.decoder_frames(), Some(0.0));
    assert_eq!(summary.overall_frames(), Some(0.0));
    assert_eq!(summary.overall_seconds(299_700), Some(0.0));
}

#[test]
fn annex_c_sequence_requirements_reject_a_too_short_run() {
    // A 200-frame run at 29.97 Hz is ≈ 6.7 s — well under the §C 100 s
    // minimum, so the precondition gate must reject it even though the
    // delay mechanism above works on it.
    let req = SequenceRequirements {
        frame_count: 200,
        fps_times_10000: 299_700,
        coded_rate_times_10: 300,
        stuffing_bits: 0,
        total_bits: 100_000,
    };
    assert!(req.validate().is_err());

    // A conformant >100 s run at 29.97 Hz with 30 Hz coded rate and 3 %
    // stuffing passes.
    let conformant = SequenceRequirements {
        frame_count: 3001,
        stuffing_bits: 3,
        total_bits: 100,
        ..req
    };
    assert_eq!(conformant.validate(), Ok(()));
}
