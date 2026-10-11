//! Energy-based speech detection against a tracked noise floor.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::error::{Error, Result};
use crate::time::{AudioDuration, AudioInstant, AudioTimestamp};

use super::config::VadConfig;
use super::metrics::{NoopVadMetricsCollector, VadMetricsCollector, VadMetricsSnapshot, VadStats};

/// Number of nanoseconds in one second, used for time conversion.
const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// Audio buffered at the start of a stream to estimate the background level
/// before any frame is classified.
const WARMUP: AudioDuration = AudioDuration::from_millis(300);

/// Per-frame factor for raising the noise floor during speech. With the
/// default 10 ms hop the floor needs a few seconds of sustained level to
/// rise, while words and syllables only nudge it.
const SPEECH_FLOOR_RISE: f32 = 0.998;

/// Streaming voice activity detector. See the [module docs](super) for an
/// example and [`VadConfig`] for tuning.
pub struct VadDetector {
    config: VadConfig,
    window: Vec<f32>,
    metrics: Arc<dyn VadMetricsCollector>,
    /// Frames buffered at the start of a stream to estimate the background.
    warmup_frames: usize,
    /// `speech_padding` in samples.
    padding_samples: usize,
    state: Mutex<DetectorState>,
}

impl fmt::Debug for VadDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let processed_samples = self.lock_state().processed_samples;
        f.debug_struct("VadDetector")
            .field("config", &self.config)
            .field("processed_samples", &processed_samples)
            .finish_non_exhaustive()
    }
}

impl VadDetector {
    /// Create a detector with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] if `config` fails validation.
    pub fn new(config: VadConfig) -> Result<Self> {
        Self::with_metrics(config, Arc::new(NoopVadMetricsCollector))
    }

    /// Create a detector that reports a [`VadMetricsSnapshot`] to `metrics`
    /// after every [`detect`](Self::detect) call.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] if `config` fails validation.
    pub fn with_metrics(config: VadConfig, metrics: Arc<dyn VadMetricsCollector>) -> Result<Self> {
        config.validate()?;

        let window = hann_window(config.frame_length_samples()?);
        let hop_secs = config.hop_length_samples()? as f64 / f64::from(config.sample_rate);
        let warmup_frames = (WARMUP.as_secs_f64() / hop_secs).ceil().max(1.0) as usize;
        let padding_samples =
            (config.speech_padding.as_secs_f64() * f64::from(config.sample_rate)).round() as usize;

        Ok(Self {
            state: Mutex::new(DetectorState::new(&config)),
            config,
            window,
            metrics,
            warmup_frames,
            padding_samples,
        })
    }

    /// The configuration this detector was built with.
    #[must_use]
    pub fn config(&self) -> &VadConfig {
        &self.config
    }

    /// Lock the streaming state. A panic while holding the lock cannot leave
    /// the state half-updated in a way that matters more than restarting
    /// would, so poisoning is ignored.
    fn lock_state(&self) -> MutexGuard<'_, DetectorState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Start sample of the speech segment still open at the end of the last
    /// [`detect`](Self::detect) call, if any.
    #[must_use]
    pub fn active_segment_start_sample(&self) -> Option<usize> {
        let state = self.lock_state();
        state
            .active_segment
            .as_ref()
            .map(|segment| segment.start_sample)
    }

    /// Clear all streaming state, including the noise floor, and set a new
    /// stream start time.
    pub fn reset(&mut self, stream_start_time: AudioTimestamp) {
        self.config.stream_start_time = stream_start_time;
        *self.lock_state() = DetectorState::new(&self.config);
    }

    /// Detect speech in the next block of samples from the stream.
    ///
    /// Successive calls are treated as one continuous stream, so blocks can
    /// be any size: samples that do not fill a frame yet wait for the next
    /// call, and a segment still open at the end of a block is reported once
    /// it closes. Call [`flush`](Self::flush) at the end of the stream to get
    /// the last segment.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Processing`] if a segment time cannot be computed.
    pub fn detect(&self, samples: &[f32]) -> Result<Vec<SpeechChunk>> {
        self.run(samples, false)
    }

    /// Analyze any samples still waiting for a full frame and close the open
    /// speech segment, if there is one. Call this at the end of a stream;
    /// call [`reset`](Self::reset) before starting another.
    ///
    /// # Errors
    ///
    /// Same as [`detect`](Self::detect).
    pub fn flush(&self) -> Result<Vec<SpeechChunk>> {
        self.run(&[], true)
    }

    /// First stream sample whose classification can still change: the start
    /// of the open speech segment, or the first sample not yet covered by a
    /// complete analysis frame.
    pub(crate) fn undecided_from(&self) -> usize {
        let state = self.lock_state();
        if let Some(first) = state.warmup.as_ref().and_then(|w| w.first()) {
            return first.start_sample;
        }
        // A segment is reported with leading padding, so audio up to one
        // padding length before its start (or before the next unanalyzed
        // frame, where a new segment could begin) is still undecided.
        let next_start = state
            .active_segment
            .as_ref()
            .map_or(state.carry_start, |seg| {
                seg.start_sample.min(state.carry_start)
            });
        next_start
            .saturating_sub(self.padding_samples)
            .max(state.last_segment_end.min(next_start))
    }

    /// First stream sample not yet covered by a complete analysis frame.
    pub(crate) fn analyzed_until(&self) -> usize {
        self.lock_state().carry_start
    }

    /// The open speech segment so far, reported as ending at `end_sample`.
    pub(crate) fn open_segment(&self, end_sample: usize) -> Option<SpeechChunk> {
        let state = self.lock_state();
        let seg = state.active_segment.as_ref()?;
        // Same leading padding as a closed segment gets.
        let start = seg
            .start_sample
            .saturating_sub(self.padding_samples)
            .max(state.last_segment_end);
        Some(SpeechChunk {
            start_time: self.absolute_time_for_sample(start).ok()?,
            end_time: self.absolute_time_for_sample(end_sample).ok()?,
            confidence: seg.confidence(),
            avg_energy: seg.avg_energy(),
            frame_count: seg.frame_count,
        })
    }

    fn run(&self, samples: &[f32], finish: bool) -> Result<Vec<SpeechChunk>> {
        let detection_start = AudioInstant::now();
        let mut state = self.lock_state();

        state.processed_samples += samples.len();
        let chunk_end_sample = state.processed_samples;

        let mut frames = self.frame_signal(samples, finish, &mut state)?;

        // Hold the first frames back until the background level can be
        // estimated from them. The quietest one is used rather than the
        // first, so a stream that starts mid-speech still gets an estimate
        // from the dips between syllables.
        if let Some(warmup) = state.warmup.as_mut() {
            warmup.append(&mut frames);
            if warmup.len() >= self.warmup_frames || finish {
                frames = std::mem::take(warmup);
                state.warmup = None;
                if let Some(quietest) = frames.iter().map(|f| f.energy).min_by(f32::total_cmp) {
                    state.noise_floor = quietest.max(self.config.energy_floor);
                }
            }
        }

        let (mut chunks, stats) = self.classify(&frames, chunk_end_sample, &mut state)?;

        if finish {
            if let Some(segment) = state.active_segment.take() {
                let prev_end = state.last_segment_end;
                state.last_segment_end =
                    self.finalize_segment(&segment, chunk_end_sample, prev_end, &mut chunks)?;
            }
        }

        let noise_floor = state.noise_floor;
        drop(state);

        let snapshot = VadMetricsSnapshot {
            stats,
            latency: AudioInstant::now().duration_since(detection_start),
            noise_floor,
        };
        self.metrics.record_vad_metrics(&snapshot);

        Ok(chunks)
    }

    /// Split the stream into analysis frames and measure each one.
    ///
    /// New samples are pre-emphasized and appended to the samples left over
    /// from earlier calls, so frames always sit on one grid regardless of how
    /// the caller sizes its blocks. Only complete frames are analyzed, unless
    /// `finish` is set, in which case the remainder is zero-padded.
    fn frame_signal(
        &self,
        samples: &[f32],
        finish: bool,
        state: &mut DetectorState,
    ) -> Result<Vec<FrameMetrics>> {
        self.pre_emphasize(samples, state);

        let frame_length = self.config.frame_length_samples()?;
        let hop_length = self.config.hop_length_samples()?;

        let mut frames = Vec::new();
        let mut start = 0usize;
        loop {
            let available = state.carry.len().saturating_sub(start);
            if !(available >= frame_length || (finish && available > 0)) {
                break;
            }
            let valid_len = available.min(frame_length);
            let frame = state
                .carry
                .get(start..start + valid_len)
                .unwrap_or_default();
            // Hann-windowed RMS; a short final frame counts as zero-padded.
            let sum_sq: f32 = frame
                .iter()
                .zip(&self.window)
                .map(|(sample, window)| (sample * window).powi(2))
                .sum();
            frames.push(FrameMetrics {
                start_sample: state.carry_start + start,
                valid_len,
                energy: (sum_sq / frame_length as f32).sqrt(),
            });
            start += hop_length;
        }

        let consumed = if finish {
            state.carry.len()
        } else {
            start.min(state.carry.len())
        };
        state.carry.drain(..consumed);
        state.carry_start += consumed;

        Ok(frames)
    }

    /// Append `samples` to the frame buffer, applying the pre-emphasis
    /// filter `y[n] = x[n] - a * x[n-1]` if it is enabled.
    fn pre_emphasize(&self, samples: &[f32], state: &mut DetectorState) {
        let coeff = self.config.pre_emphasis.unwrap_or(0.0);
        let mut previous = state.pre_emphasis_prev;
        state.carry.extend(samples.iter().map(|&sample| {
            let emphasized = coeff.mul_add(-previous, sample);
            previous = sample;
            emphasized
        }));
        state.pre_emphasis_prev = previous;
    }

    /// Classify frames as speech or silence, update the noise floor, and
    /// collect the segments that close.
    fn classify(
        &self,
        frames: &[FrameMetrics],
        chunk_end_sample: usize,
        state: &mut DetectorState,
    ) -> Result<(Vec<SpeechChunk>, VadStats)> {
        let mut stats = VadStats::default();
        let mut segments = Vec::new();
        let smoothing = self.config.noise_smoothing;

        for frame in frames {
            let ratio = frame.energy / state.noise_floor;
            let required = if state.active_segment.is_some() {
                self.config.release_ratio
            } else {
                self.config.activation_ratio
            };
            let loud = frame.energy >= self.config.min_speech_energy && ratio >= required;

            stats.frames_processed += 1;
            if loud {
                stats.speech_frames += 1;
            } else {
                stats.silence_frames += 1;
            }

            // A quiet frame inside a segment is a pause until more than
            // `hangover_frames` of them in a row close it.
            let in_speech = match state.active_segment.as_mut() {
                Some(segment) if loud => {
                    segment.silence_run = 0;
                    true
                }
                Some(segment) => {
                    segment.silence_run += 1;
                    segment.silence_run <= self.config.hangover_frames
                }
                None => loud,
            };

            if loud {
                // Overlapping frames can make a new segment start before the
                // previous one ended; never let segments overlap.
                let start = frame.start_sample.max(state.last_segment_end);
                let segment = state
                    .active_segment
                    .get_or_insert_with(|| ActiveSegmentState::new(start));
                // Only loud frames extend the segment; hangover frames keep
                // it open without moving its end.
                segment.ratio_score_sum += ratio.log10().clamp(0.0, 1.0);
                segment.energy_sum += frame.energy;
                segment.frame_count += 1;
                segment.last_end_sample = frame.start_sample + frame.valid_len.max(1);
            } else if !in_speech {
                if let Some(segment) = state.active_segment.take() {
                    state.last_segment_end = self.finalize_segment(
                        &segment,
                        chunk_end_sample,
                        state.last_segment_end,
                        &mut segments,
                    )?;
                }
            }

            // Track the noise floor through speech as well: follow dips
            // quickly and rise slowly. Without this, a recording that starts
            // with noise above the gate is classified as speech from the
            // first frame and the floor never adapts.
            let floor = state.noise_floor;
            let weight = if in_speech && frame.energy > floor {
                SPEECH_FLOOR_RISE
            } else {
                smoothing
            };
            state.noise_floor = weight
                .mul_add(floor, (1.0 - weight) * frame.energy)
                .max(self.config.energy_floor);
        }

        Ok((segments, stats))
    }

    /// Report a closed segment, padded by `speech_padding` on both sides
    /// without overlapping the previous segment (`prev_end`) or reaching past
    /// the audio received so far. Returns the end to use as the new
    /// `prev_end`.
    fn finalize_segment(
        &self,
        segment: &ActiveSegmentState,
        chunk_end_sample: usize,
        prev_end: usize,
        segments: &mut Vec<SpeechChunk>,
    ) -> Result<usize> {
        let unpadded_end = segment.last_end_sample.min(chunk_end_sample);
        if segment.last_end_sample <= segment.start_sample
            || segment.frame_count < self.config.min_speech_frames
        {
            return Ok(unpadded_end.max(prev_end));
        }

        let start = segment
            .start_sample
            .saturating_sub(self.padding_samples)
            .max(prev_end);
        let end = (segment.last_end_sample + self.padding_samples)
            .min(chunk_end_sample.max(segment.start_sample + 1));
        let start_time = self.absolute_time_for_sample(start)?;
        let end_time = self.absolute_time_for_sample(end)?;

        if end_time <= start_time {
            return Ok(unpadded_end.max(prev_end));
        }

        segments.push(SpeechChunk {
            start_time,
            end_time,
            confidence: segment.confidence(),
            avg_energy: segment.avg_energy(),
            frame_count: segment.frame_count,
        });

        Ok(end)
    }

    fn absolute_time_for_sample(&self, sample_index: usize) -> Result<AudioTimestamp> {
        let offset = samples_to_duration(sample_index, self.config.sample_rate);
        Ok(self.config.stream_start_time.add_duration(offset))
    }
}

fn hann_window(length: usize) -> Vec<f32> {
    if length == 1 {
        return vec![1.0];
    }
    let denom = (length - 1) as f32;
    (0..length)
        .map(|n| {
            let angle = 2.0 * std::f32::consts::PI * n as f32 / denom;
            0.5f32.mul_add(-angle.cos(), 0.5)
        })
        .collect()
}

fn samples_to_duration(samples: usize, sample_rate: u32) -> AudioDuration {
    let sr = u128::from(sample_rate);
    let nanos = ((samples as u128) * NANOS_PER_SECOND + sr / 2) / sr;
    AudioDuration::from_nanos(nanos as u64)
}

/// Energy of one analysis frame.
#[derive(Debug, Clone, Copy)]
struct FrameMetrics {
    start_sample: usize,
    /// Real (not zero-padded) samples in the frame.
    valid_len: usize,
    energy: f32,
}

struct DetectorState {
    /// Samples received so far.
    processed_samples: usize,
    /// Measurements of the first frames of the stream, held back until
    /// there are enough to estimate the background level. `None` once the
    /// noise floor has been seeded.
    warmup: Option<Vec<FrameMetrics>>,
    /// Pre-emphasized samples not yet consumed by a complete frame.
    carry: Vec<f32>,
    /// Stream index of `carry[0]`.
    carry_start: usize,
    /// End of the last closed segment; new segments start at or after it.
    last_segment_end: usize,
    /// Running estimate of the background frame energy.
    noise_floor: f32,
    pre_emphasis_prev: f32,
    active_segment: Option<ActiveSegmentState>,
}

struct ActiveSegmentState {
    start_sample: usize,
    /// End of the last loud frame.
    last_end_sample: usize,
    ratio_score_sum: f32,
    energy_sum: f32,
    /// Loud frames in the segment (hangover frames are not counted).
    frame_count: usize,
    /// Quiet frames since the last loud one.
    silence_run: usize,
}

impl DetectorState {
    fn new(config: &VadConfig) -> Self {
        Self {
            processed_samples: 0,
            warmup: Some(Vec::new()),
            carry: Vec::new(),
            carry_start: 0,
            last_segment_end: 0,
            noise_floor: config.energy_floor,
            pre_emphasis_prev: 0.0,
            active_segment: None,
        }
    }
}

impl ActiveSegmentState {
    fn new(start_sample: usize) -> Self {
        Self {
            start_sample,
            last_end_sample: start_sample,
            ratio_score_sum: 0.0,
            energy_sum: 0.0,
            frame_count: 0,
            silence_run: 0,
        }
    }

    fn confidence(&self) -> f32 {
        self.ratio_score_sum / self.frame_count.max(1) as f32
    }

    fn avg_energy(&self) -> f32 {
        self.energy_sum / self.frame_count.max(1) as f32
    }
}

/// A detected speech segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeechChunk {
    /// Start of the segment, including `speech_padding`.
    pub start_time: AudioTimestamp,
    /// End of the segment, including `speech_padding`.
    pub end_time: AudioTimestamp,
    /// How far the speech frames rise above the noise floor, from 0.0 (at
    /// the floor) to 1.0 (20 dB or more above it), averaged over the
    /// segment.
    pub confidence: f32,
    /// Mean RMS of the speech frames, measured after pre-emphasis and
    /// windowing.
    pub avg_energy: f32,
    /// Number of speech frames in the segment. Hangover frames are not
    /// counted.
    pub frame_count: usize,
}

impl SpeechChunk {
    /// Duration of the speech segment.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Processing`] if `end_time` precedes `start_time`.
    pub fn duration(&self) -> Result<AudioDuration> {
        self.end_time
            .duration_since(self.start_time)
            .ok_or_else(|| {
                Error::Processing(
                    "failed to compute segment duration: end_time precedes start_time".into(),
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use super::*;

    const SR: usize = 16_000;

    /// `silence_s` of silence, `tone_s` of a 1 kHz tone, `silence_s` of
    /// silence.
    fn tone_burst(silence_s: f32, tone_s: f32) -> Vec<f32> {
        let pad = (silence_s * SR as f32) as usize;
        let tone = (tone_s * SR as f32) as usize;
        let mut audio = vec![0.0; pad * 2 + tone];
        for (i, s) in audio[pad..pad + tone].iter_mut().enumerate() {
            *s = 0.5 * (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / SR as f32).sin();
        }
        audio
    }

    #[test]
    fn silence_produces_no_segments() {
        let detector = VadDetector::new(VadConfig::default()).unwrap();
        assert!(detector.detect(&vec![0.0; SR]).unwrap().is_empty());
    }

    #[test]
    fn detects_tone_burst_boundaries() {
        let detector = VadDetector::new(VadConfig::default()).unwrap();
        let segments = detector.detect(&tone_burst(0.5, 1.0)).unwrap();

        // Boundaries include the default 50 ms of padding on each side.
        assert_eq!(segments.len(), 1);
        let seg = segments[0];
        assert!((seg.start_time.as_secs() - 0.45).abs() < 0.03, "{seg:?}");
        assert!((seg.end_time.as_secs() - 1.55).abs() < 0.03, "{seg:?}");
    }

    #[test]
    fn streaming_matches_single_call() {
        let audio = tone_burst(0.5, 1.0);

        let whole = VadDetector::new(VadConfig::default()).unwrap();
        let expected = whole.detect(&audio).unwrap();

        let streamed = VadDetector::new(VadConfig::default()).unwrap();
        let mut actual = Vec::new();
        for block in audio.chunks(1600) {
            actual.extend(streamed.detect(block).unwrap());
        }

        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(&expected) {
            assert!((a.start_time.as_secs() - e.start_time.as_secs()).abs() < 0.03);
            assert!((a.end_time.as_secs() - e.end_time.as_secs()).abs() < 0.03);
        }
    }

    #[test]
    fn tiny_blocks_match_single_call() {
        let audio = tone_burst(0.5, 1.0);
        let expected = VadDetector::new(VadConfig::default())
            .unwrap()
            .detect(&audio)
            .unwrap();

        for block in [1, 7, 64] {
            let detector = VadDetector::new(VadConfig::default()).unwrap();
            let mut actual = Vec::new();
            for b in audio.chunks(block) {
                actual.extend(detector.detect(b).unwrap());
            }
            actual.extend(detector.flush().unwrap());
            assert_eq!(actual.len(), expected.len(), "block {block}");
            let drift = actual[0].start_time.as_secs() - expected[0].start_time.as_secs();
            assert!(drift.abs() < 0.02, "block {block}: start drift {drift}");
        }
    }

    #[test]
    fn flush_closes_trailing_segment() {
        // Audio that ends while the tone is still playing.
        let audio = &tone_burst(0.5, 1.0)[..SR + SR / 2];
        let detector = VadDetector::new(VadConfig::default()).unwrap();
        assert!(detector.detect(audio).unwrap().is_empty());

        let closed = detector.flush().unwrap();
        assert_eq!(closed.len(), 1);
        let end = closed[0].end_time.as_secs();
        assert!((end - 1.5).abs() < 0.03, "segment ends at {end}");
    }

    #[test]
    fn segments_never_overlap() {
        let config = VadConfig {
            frame_overlap: 0.9,
            ..VadConfig::default()
        };
        for gap in (100..1_200).step_by(37) {
            let mut audio = tone_burst(0.25, 0.3);
            audio.truncate(audio.len() - SR / 4 + gap);
            audio.extend(tone_burst(0.0, 0.3));
            audio.extend(vec![0.0; SR / 2]);

            let detector = VadDetector::new(config).unwrap();
            let mut segments = detector.detect(&audio).unwrap();
            segments.extend(detector.flush().unwrap());
            for pair in segments.windows(2) {
                assert!(
                    pair[1].start_time >= pair[0].end_time,
                    "gap {gap}: {pair:?}"
                );
            }
        }
    }

    /// Deterministic noise in [-1, 1), low-pass filtered when `brown` is set.
    fn noise(len: usize, amplitude: f32, brown: bool) -> Vec<f32> {
        let mut state = 0x2545_F491u32;
        let mut low = 0.0f32;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let white = (state as f32 / u32::MAX as f32).mul_add(2.0, -1.0);
                low = 0.98f32.mul_add(low, 0.2 * white);
                amplitude * if brown { low } else { white }
            })
            .collect()
    }

    #[test]
    fn steady_noise_from_the_first_sample_is_not_speech() {
        for (amplitude, brown) in [(0.01, false), (0.05, false), (0.05, true), (0.2, true)] {
            let detector = VadDetector::new(VadConfig::default()).unwrap();
            let audio = noise(SR * 5, amplitude, brown);
            let mut segments = detector.detect(&audio).unwrap();
            segments.extend(detector.flush().unwrap());
            assert!(
                segments.is_empty(),
                "noise {amplitude} brown={brown}: {segments:?}"
            );
        }
    }

    #[test]
    fn speech_in_noise_does_not_lock_on() {
        // Noise loud enough to pass the absolute gate from the first frame,
        // with two 0.6 s tones. Before the baselines tracked the floor during
        // speech, this produced one segment covering the whole input.
        let mut audio = noise(SR * 6, 0.05, true);
        for start in [SR * 2, SR * 4] {
            for (i, s) in audio[start..start + SR * 6 / 10].iter_mut().enumerate() {
                *s += 0.4 * (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / SR as f32).sin();
            }
        }

        let detector = VadDetector::new(VadConfig::default()).unwrap();
        let mut segments = detector.detect(&audio).unwrap();
        segments.extend(detector.flush().unwrap());

        assert_eq!(segments.len(), 2, "{segments:?}");
        for (seg, start) in segments.iter().zip([2.0, 4.0]) {
            // Includes 50 ms of leading padding.
            assert!(
                (seg.start_time.as_secs() - (start - 0.05)).abs() < 0.03,
                "{seg:?}"
            );
            assert!(
                (seg.end_time.as_secs() - (start + 0.6)).abs() < 0.1,
                "{seg:?}"
            );
        }
    }

    fn detect_all(config: VadConfig, audio: &[f32]) -> Vec<SpeechChunk> {
        let detector = VadDetector::new(config).unwrap();
        let mut segments = detector.detect(audio).unwrap();
        segments.extend(detector.flush().unwrap());
        segments
    }

    /// White noise with a loud 1 kHz tone at 1.0-1.2 s followed directly by
    /// a soft one at 1.2-1.6 s, about twice the noise level.
    fn loud_then_soft() -> Vec<f32> {
        let mut audio = noise(SR * 3, 0.02, false);
        for (i, s) in audio[SR..SR * 16 / 10].iter_mut().enumerate() {
            let amplitude = if i < SR / 5 { 0.3 } else { 0.1 };
            *s += amplitude * (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / SR as f32).sin();
        }
        audio
    }

    #[test]
    fn release_ratio_keeps_the_soft_end_of_a_segment() {
        let segments = detect_all(VadConfig::default(), &loud_then_soft());
        assert_eq!(segments.len(), 1, "{segments:?}");
        assert!(
            (segments[0].end_time.as_secs() - 1.65).abs() < 0.05,
            "{segments:?}"
        );

        // Without hysteresis the soft part is too quiet to keep it open.
        let strict = VadConfig {
            release_ratio: 2.5,
            ..VadConfig::default()
        };
        let segments = detect_all(strict, &loud_then_soft());
        assert_eq!(segments.len(), 1, "{segments:?}");
        assert!(
            (segments[0].end_time.as_secs() - 1.25).abs() < 0.05,
            "{segments:?}"
        );
    }

    #[test]
    fn confidence_follows_level_above_the_noise_floor() {
        let clean = detect_all(VadConfig::default(), &tone_burst(1.0, 1.0));
        let soft = detect_all(VadConfig::default(), &loud_then_soft());
        assert!(clean[0].confidence > 0.9, "{clean:?}");
        assert!(
            soft[0].confidence > 0.2 && soft[0].confidence < 0.7,
            "{soft:?}"
        );
    }

    #[test]
    fn stream_starting_mid_speech_is_detected() {
        // Voiced harmonics with 10 dB syllable dips at 4 Hz, starting on a
        // loud peak. Seeding the background from the first frame used to
        // classify all of it as background.
        let mut audio: Vec<f32> = (0..SR * 2)
            .map(|i| {
                let t = i as f32 / SR as f32;
                let phase = 2.0 * std::f32::consts::PI * 4.0 * t;
                let envelope = 0.684f32.mul_add(0.5f32.mul_add(phase.cos(), 0.5), 0.316);
                let voiced: f32 = (1..=15)
                    .map(|h| (2.0 * std::f32::consts::PI * 150.0 * h as f32 * t).sin() / h as f32)
                    .sum();
                0.3 * envelope * voiced
            })
            .collect();
        audio.extend(vec![0.0; SR]);

        let detector = VadDetector::new(VadConfig::default()).unwrap();
        let mut segments = detector.detect(&audio).unwrap();
        segments.extend(detector.flush().unwrap());

        let speech: f64 = segments
            .iter()
            .map(|s| s.end_time.as_secs() - s.start_time.as_secs())
            .sum();
        assert!(segments[0].start_time.as_secs() < 0.05, "{segments:?}");
        assert!(speech > 1.8, "only {speech:.2} s detected: {segments:?}");
    }

    #[test]
    fn stats_count_speech_and_silence_frames() {
        #[derive(Default)]
        struct Capture(StdMutex<Vec<VadStats>>);
        impl VadMetricsCollector for Capture {
            fn record_vad_metrics(&self, snapshot: &VadMetricsSnapshot) {
                self.0.lock().unwrap().push(snapshot.stats);
            }
        }

        let capture = Arc::new(Capture::default());
        let detector = VadDetector::with_metrics(VadConfig::default(), capture.clone()).unwrap();
        detector.detect(&tone_burst(0.5, 1.0)).unwrap();

        let stats = capture.0.lock().unwrap()[0];
        assert_eq!(
            stats.frames_processed,
            stats.speech_frames + stats.silence_frames
        );
        // Roughly half of the 2 s input is the tone.
        let ratio = stats.speech_ratio();
        assert!((0.4..0.6).contains(&ratio), "speech ratio {ratio}");
    }

    #[test]
    fn reset_restarts_timestamps() {
        let mut detector = VadDetector::new(VadConfig::default()).unwrap();
        let audio = tone_burst(0.5, 1.0);
        let first = detector.detect(&audio).unwrap();

        detector.reset(AudioTimestamp::EPOCH);
        let second = detector.detect(&audio).unwrap();
        assert_eq!(first, second);
    }
}
