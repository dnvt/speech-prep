//! Core VAD detection engine with dual-metric analysis.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::error::{Error, Result};
use crate::time::{AudioDuration, AudioInstant, AudioTimestamp};
use realfft::{RealFftPlanner, RealToComplex};

use super::config::VadConfig;
use super::metrics::{
    AdaptiveThresholdSnapshot, NoopVadMetricsCollector, VadMetricsCollector, VadMetricsSnapshot,
    VadStats,
};

/// Number of nanoseconds in one second, used for time conversion.
const NANOS_PER_SECOND: u128 = 1_000_000_000;
/// Small epsilon value for numerical stability in floating-point comparisons.
const EPSILON: f32 = 1e-12;

/// Maximum smoothing factor for baseline tracking (energy, flux, threshold).
/// Capped at 0.999 to prevent numerical instability from exponential moving
/// average converging too slowly. At 0.999, half-life ≈ 693 samples (43ms at
/// 16kHz).
const MAX_SMOOTHING_FACTOR: f32 = 0.999;

/// Maximum normalized value for energy and spectral flux metrics.
/// Caps outliers at 10x the baseline to prevent extreme transients from
/// dominating the detection logic while still allowing headroom for loud
/// signals.
const MAX_NORMALIZED_METRIC: f32 = 10.0;

/// Relative energy ratio below which we consider a frame near-silence.
const SILENCE_RELATIVE_GATE: f32 = 1.7;

/// Streaming voice activity detector. See the [module docs](super) for an
/// example and [`VadConfig`] for tuning.
pub struct VadDetector {
    config: VadConfig,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    metrics: Arc<dyn VadMetricsCollector>,
    energy_weight: f32,
    flux_weight: f32,
    state: Mutex<DetectorState>,
}

impl fmt::Debug for VadDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let processed_samples = self.lock_state().processed_samples;
        f.debug_struct("VadDetector")
            .field("config", &self.config)
            .field("window_length", &self.window.len())
            .field("energy_weight", &self.energy_weight)
            .field("flux_weight", &self.flux_weight)
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

        let frame_length = config.frame_length_samples()?;
        let window = hann_window(frame_length);

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(config.fft_size()?);

        let total_weight = config.energy_weight + config.flux_weight;
        let (energy_weight, flux_weight) = (
            config.energy_weight / total_weight,
            config.flux_weight / total_weight,
        );

        let state = DetectorState::new(&config, fft.make_output_vec().len());

        Ok(Self {
            config,
            fft,
            window,
            metrics,
            energy_weight,
            flux_weight,
            state: Mutex::new(state),
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

    /// Clear all streaming state, including the adaptive baselines, and set a
    /// new stream start time.
    pub fn reset(&mut self, stream_start_time: AudioTimestamp) {
        self.config.stream_start_time = stream_start_time;
        let bins = self.fft.make_output_vec().len();
        *self.lock_state() = DetectorState::new(&self.config, bins);
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
    /// Returns [`Error::Processing`] if framing or the FFT fails.
    pub fn detect(&self, samples: &[f32]) -> Result<Vec<SpeechChunk>> {
        self.run(samples, false)
    }

    /// Analyze any samples still waiting for a full frame and close the open
    /// speech segment, if there is one. Call this at the end of a stream;
    /// call [`reset`](Self::reset) before starting another.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Processing`] if the FFT fails.
    pub fn flush(&self) -> Result<Vec<SpeechChunk>> {
        self.run(&[], true)
    }

    /// First stream sample whose classification can still change: the start
    /// of the open speech segment, or the first sample not yet covered by a
    /// complete analysis frame.
    pub(crate) fn undecided_from(&self) -> usize {
        let state = self.lock_state();
        state
            .active_segment
            .as_ref()
            .map_or(state.carry_start, |seg| {
                seg.start_sample.min(state.carry_start)
            })
    }

    /// First stream sample not yet covered by a complete analysis frame.
    pub(crate) fn analyzed_until(&self) -> usize {
        self.lock_state().carry_start
    }

    /// The open speech segment so far, reported as ending at `end_sample`.
    pub(crate) fn open_segment(&self, end_sample: usize) -> Option<SpeechChunk> {
        let state = self.lock_state();
        let seg = state.active_segment.as_ref()?;
        let frames = seg.frame_count.max(1) as f32;
        Some(SpeechChunk {
            start_time: self.absolute_time_for_sample(seg.start_sample).ok()?,
            end_time: self.absolute_time_for_sample(end_sample).ok()?,
            confidence: (seg.score_sum / frames).clamp(0.0, 1.0),
            avg_energy: seg.energy_sum / frames,
            frame_count: seg.frame_count,
        })
    }

    fn run(&self, samples: &[f32], finish: bool) -> Result<Vec<SpeechChunk>> {
        let detection_start = AudioInstant::now();
        let mut detector_state = self.lock_state();

        detector_state.processed_samples += samples.len();
        let chunk_end_sample = detector_state.processed_samples;

        let frames = self.frame_signal(samples, finish, &mut detector_state)?;

        let (mut chunks, stats) = if frames.is_empty() {
            (Vec::new(), VadStats::default())
        } else {
            let energy = Self::compute_energy(&frames);
            let flux = self.compute_spectral_flux(&frames, &mut detector_state)?;
            self.merge_metrics(
                &frames,
                &energy,
                &flux,
                chunk_end_sample,
                &mut detector_state,
            )?
        };

        if finish {
            if let Some(segment) = detector_state.active_segment.take() {
                self.finalize_segment(&segment, chunk_end_sample, &mut chunks)?;
                detector_state.last_segment_end = segment.last_end_sample.min(chunk_end_sample);
            }
        }

        let adaptive = AdaptiveThresholdSnapshot {
            energy_baseline: detector_state.energy_baseline,
            flux_baseline: detector_state.flux_baseline,
            dynamic_threshold: detector_state.dynamic_threshold,
        };
        drop(detector_state);

        let latency = AudioInstant::now().duration_since(detection_start);
        let snapshot = VadMetricsSnapshot {
            stats,
            latency,
            adaptive,
        };
        self.metrics.record_vad_metrics(&snapshot);

        Ok(chunks)
    }

    /// Split the stream into analysis frames.
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
    ) -> Result<Vec<Frame>> {
        let processed = self.preprocess_signal(samples, state);
        state.carry.extend_from_slice(&processed);

        let frame_length = self.config.frame_length_samples()?;
        let hop_length = self.config.hop_length_samples()?;

        let mut frames = Vec::new();
        let mut start = 0usize;
        loop {
            let available = state.carry.len().saturating_sub(start);
            let complete = available >= frame_length;
            if !(complete || (finish && available > 0)) {
                break;
            }

            let slice = state.carry.get(start..start + available.min(frame_length));
            let mut frame: Vec<f32> = slice
                .unwrap_or_default()
                .iter()
                .zip(&self.window)
                .map(|(sample, window)| sample * window)
                .collect();
            frame.resize(frame_length, 0.0);

            frames.push(Frame {
                data: frame,
                start_sample: state.carry_start + start,
                valid_len: available.min(frame_length),
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

    fn preprocess_signal(&self, samples: &[f32], state: &mut DetectorState) -> Vec<f32> {
        match self.config.pre_emphasis {
            Some(coeff) if coeff > 0.0 => {
                let mut processed = Vec::with_capacity(samples.len());
                let mut previous = state.pre_emphasis_prev;
                for &sample in samples {
                    let emphasized = coeff.mul_add(-previous, sample);
                    processed.push(emphasized);
                    previous = sample;
                }
                if let Some(&last) = samples.last() {
                    state.pre_emphasis_prev = last;
                }
                processed
            }
            _ => {
                if let Some(&last) = samples.last() {
                    state.pre_emphasis_prev = last;
                }
                samples.to_vec()
            }
        }
    }

    fn compute_energy(frames: &[Frame]) -> Vec<f32> {
        let mut values = Vec::with_capacity(frames.len());

        for frame in frames {
            debug_assert!(!frame.data.is_empty(), "frame data should never be empty");
            let sum_sq: f32 = frame.data.iter().map(|sample| sample * sample).sum();
            let len = frame.data.len();
            let rms = (sum_sq / len as f32).sqrt();

            values.push(rms);
        }

        values
    }

    fn compute_spectral_flux(
        &self,
        frames: &[Frame],
        state: &mut DetectorState,
    ) -> Result<Vec<f32>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        if state.previous_spectrum.len() != spectrum.len() {
            state.previous_spectrum.resize(spectrum.len(), 0.0);
        }
        let previous = &mut state.previous_spectrum;

        let mut values = Vec::with_capacity(frames.len());

        for frame in frames {
            debug_assert!(!frame.data.is_empty(), "frame data should never be empty");
            input.fill(0.0);
            let len = frame.data.len().min(input.len());
            for (dst, &src) in input.iter_mut().zip(frame.data.iter()).take(len) {
                *dst = src;
            }

            self.fft
                .process_with_scratch(&mut input, &mut spectrum, &mut scratch)
                .map_err(|err| Error::Processing(format!("FFT processing failed: {err}")))?;

            let mut flux = 0.0f32;
            for (bin, prev) in spectrum.iter().zip(previous.iter_mut()) {
                let magnitude = bin.re.hypot(bin.im);
                let diff = (magnitude - *prev).max(0.0);
                flux += diff;
                *prev = magnitude;
            }

            values.push(flux);
        }

        Ok(values)
    }

    fn merge_metrics(
        &self,
        frames: &[Frame],
        energy: &[f32],
        flux: &[f32],
        chunk_end_sample: usize,
        detector_state: &mut DetectorState,
    ) -> Result<(Vec<SpeechChunk>, VadStats)> {
        let mut stats = VadStats::default();
        let mut segments = Vec::new();

        let mut dynamic_threshold = detector_state.dynamic_threshold.max(EPSILON);
        let mut energy_baseline = detector_state
            .energy_baseline
            .max(self.config.energy_floor)
            .max(EPSILON);
        let mut flux_baseline = detector_state
            .flux_baseline
            .max(self.config.flux_floor)
            .max(EPSILON);

        let silence_energy_smoothing = self.config.energy_smoothing.min(MAX_SMOOTHING_FACTOR);
        let silence_flux_smoothing = self.config.flux_smoothing.min(MAX_SMOOTHING_FACTOR);
        let silence_threshold_smoothing = self.config.threshold_smoothing.min(MAX_SMOOTHING_FACTOR);

        let dynamic_threshold_min =
            (self.config.base_threshold * self.config.release_margin).max(EPSILON);
        let dynamic_threshold_max =
            self.config.base_threshold * self.config.activation_margin * 2.0;

        let mut active_segment = detector_state.active_segment.take();
        let mut last_segment_end = detector_state.last_segment_end;
        let mut silence_run = active_segment
            .as_ref()
            .map_or(0usize, |state| state.silence_run);

        for (idx, frame) in frames.iter().enumerate() {
            let raw_energy = energy.get(idx).copied().ok_or_else(|| {
                Error::Processing(format!("energy array length mismatch at index {idx}"))
            })?;
            let raw_flux = flux.get(idx).copied().ok_or_else(|| {
                Error::Processing(format!("flux array length mismatch at index {idx}"))
            })?;

            // Seed the baselines from the start of the stream, assuming it
            // begins with background. Frame 0 has no previous spectrum, so
            // its flux is meaningless; seed flux from frame 1.
            match detector_state.frames_seen {
                0 => energy_baseline = raw_energy.max(self.config.energy_floor),
                1 => flux_baseline = raw_flux.max(self.config.flux_floor),
                _ => {}
            }
            detector_state.frames_seen += 1;

            let energy_denominator = energy_baseline.max(self.config.energy_floor).max(EPSILON);
            let normalized_energy =
                (raw_energy / energy_denominator).clamp(0.0, MAX_NORMALIZED_METRIC);
            let flux_denominator = flux_baseline.max(self.config.flux_floor).max(EPSILON);
            let normalized_flux = (raw_flux / flux_denominator).clamp(0.0, MAX_NORMALIZED_METRIC);
            let energy_ratio = raw_energy / energy_denominator;

            let combined = self
                .energy_weight
                .mul_add(normalized_energy, self.flux_weight * normalized_flux);

            let base_threshold = if active_segment.is_some() {
                dynamic_threshold * self.config.release_margin
            } else {
                dynamic_threshold * self.config.activation_margin
            };
            let threshold =
                base_threshold.max(self.config.base_threshold * self.config.release_margin);
            let low_energy = raw_energy < self.config.min_speech_energy;
            let low_relative_energy = energy_ratio < SILENCE_RELATIVE_GATE;
            let mut raw_is_speech = combined >= threshold;
            if raw_is_speech && (low_energy || low_relative_energy) {
                raw_is_speech = false;
            }

            let is_speech = if active_segment.is_some() {
                if raw_is_speech {
                    silence_run = 0;
                    true
                } else {
                    silence_run += 1;
                    silence_run <= self.config.hangover_frames
                }
            } else {
                silence_run = 0;
                raw_is_speech
            };

            if is_speech {
                // Overlapping frames can make a new segment start before the
                // previous one ended; never let segments overlap.
                let start = frame.start_sample.max(last_segment_end);
                let segment_state =
                    active_segment.get_or_insert_with(|| ActiveSegmentState::new(start));
                segment_state.score_sum += combined;
                segment_state.energy_sum += raw_energy;
                segment_state.frame_count += 1;
                segment_state.last_end_sample = frame.start_sample + frame.valid_len.max(1);
                segment_state.silence_run = silence_run;
            } else if let Some(segment_state) = active_segment.take() {
                let finalize_result =
                    self.finalize_segment(&segment_state, chunk_end_sample, &mut segments);
                if let Err(err) = finalize_result {
                    detector_state.active_segment = Some(segment_state);
                    return Err(err);
                }
                last_segment_end = segment_state.last_end_sample.min(chunk_end_sample);
                silence_run = 0;
            }

            stats.frames_processed += 1;
            if is_speech {
                stats.speech_frames += 1;
            } else {
                stats.silence_frames += 1;
            }

            if is_speech {
                // Track the noise floor through speech as well: follow dips
                // quickly and rise slowly. Without this, a recording that
                // starts with noise above the gate is classified as speech
                // from the first frame and the baselines never adapt.
                energy_baseline =
                    track_floor(energy_baseline, raw_energy, silence_energy_smoothing);
                flux_baseline = track_floor(flux_baseline, raw_flux, silence_flux_smoothing);
            } else {
                dynamic_threshold = silence_threshold_smoothing.mul_add(
                    dynamic_threshold,
                    (1.0 - silence_threshold_smoothing) * combined,
                );
                energy_baseline = silence_energy_smoothing.mul_add(
                    energy_baseline,
                    (1.0 - silence_energy_smoothing) * raw_energy,
                );
                flux_baseline = silence_flux_smoothing
                    .mul_add(flux_baseline, (1.0 - silence_flux_smoothing) * raw_flux);
            }

            dynamic_threshold =
                dynamic_threshold.clamp(dynamic_threshold_min, dynamic_threshold_max);
            energy_baseline = energy_baseline.max(self.config.energy_floor).max(EPSILON);
            flux_baseline = flux_baseline.max(self.config.flux_floor).max(EPSILON);
        }

        detector_state.dynamic_threshold = dynamic_threshold;
        detector_state.energy_baseline = energy_baseline;
        detector_state.flux_baseline = flux_baseline;
        detector_state.last_segment_end = last_segment_end;

        // Keep the open segment for the next call; `flush` closes it.
        if let Some(mut segment_state) = active_segment {
            segment_state.silence_run = silence_run;
            detector_state.active_segment = Some(segment_state);
        } else {
            detector_state.active_segment = None;
        }

        Ok((segments, stats))
    }

    fn finalize_segment(
        &self,
        segment: &ActiveSegmentState,
        chunk_end_sample: usize,
        segments: &mut Vec<SpeechChunk>,
    ) -> Result<()> {
        if segment.last_end_sample <= segment.start_sample {
            return Ok(());
        }

        if segment.frame_count < self.config.min_speech_frames {
            return Ok(());
        }

        let clamped_end = segment
            .last_end_sample
            .min(chunk_end_sample.max(segment.start_sample + 1));
        let start_time = self.absolute_time_for_sample(segment.start_sample)?;
        let end_time = self.absolute_time_for_sample(clamped_end)?;

        if end_time <= start_time {
            return Ok(());
        }

        let confidence = (segment.score_sum / segment.frame_count as f32).clamp(0.0, 1.0);
        let avg_energy = if segment.frame_count > 0 {
            segment.energy_sum / segment.frame_count as f32
        } else {
            0.0
        };

        segments.push(SpeechChunk {
            start_time,
            end_time,
            confidence,
            avg_energy,
            frame_count: segment.frame_count,
        });

        Ok(())
    }

    fn absolute_time_for_sample(&self, sample_index: usize) -> Result<AudioTimestamp> {
        let offset = samples_to_duration(sample_index, self.config.sample_rate);
        Ok(self.config.stream_start_time.add_duration(offset))
    }
}

/// Per-frame factor for raising a baseline during speech. With the default
/// 10 ms hop the baseline needs a few seconds of sustained level to rise,
/// while words and syllables only nudge it.
const SPEECH_FLOOR_RISE: f32 = 0.998;

/// Noise-floor update during speech: fall like a normal baseline update when
/// the frame is quieter, rise slowly when it is louder.
fn track_floor(baseline: f32, value: f32, fall_smoothing: f32) -> f32 {
    let smoothing = if value < baseline {
        fall_smoothing
    } else {
        SPEECH_FLOOR_RISE
    };
    smoothing.mul_add(baseline, (1.0 - smoothing) * value)
}

fn hann_window(length: usize) -> Vec<f32> {
    if length == 0 {
        return Vec::new();
    }

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

struct Frame {
    data: Vec<f32>,
    start_sample: usize,
    valid_len: usize,
}

pub(super) struct DetectorState {
    /// Samples received so far.
    pub(super) processed_samples: usize,
    /// Analysis frames processed so far.
    pub(super) frames_seen: u64,
    /// Pre-emphasized samples not yet consumed by a complete frame.
    pub(super) carry: Vec<f32>,
    /// Stream index of `carry[0]`.
    pub(super) carry_start: usize,
    /// End of the last closed segment; new segments start at or after it.
    pub(super) last_segment_end: usize,
    pub(super) energy_baseline: f32,
    pub(super) flux_baseline: f32,
    pub(super) dynamic_threshold: f32,
    pub(super) previous_spectrum: Vec<f32>,
    pub(super) pre_emphasis_prev: f32,
    pub(super) active_segment: Option<ActiveSegmentState>,
}

pub(super) struct ActiveSegmentState {
    pub(super) start_sample: usize,
    pub(super) last_end_sample: usize,
    pub(super) score_sum: f32,
    pub(super) energy_sum: f32,
    pub(super) frame_count: usize,
    pub(super) silence_run: usize,
}

impl DetectorState {
    fn new(config: &VadConfig, spectrum_bins: usize) -> Self {
        Self {
            processed_samples: 0,
            frames_seen: 0,
            carry: Vec::new(),
            carry_start: 0,
            last_segment_end: 0,
            energy_baseline: config.energy_floor.max(EPSILON),
            flux_baseline: config.flux_floor.max(EPSILON),
            dynamic_threshold: config.base_threshold.max(EPSILON),
            previous_spectrum: vec![0.0; spectrum_bins],
            pre_emphasis_prev: 0.0,
            active_segment: None,
        }
    }
}

impl ActiveSegmentState {
    pub(super) fn new(start_sample: usize) -> Self {
        Self {
            start_sample,
            last_end_sample: start_sample,
            score_sum: 0.0,
            energy_sum: 0.0,
            frame_count: 0,
            silence_run: 0,
        }
    }
}

/// Speech segment with temporal metadata emitted by the detector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeechChunk {
    /// Start time of the detected speech segment.
    pub start_time: AudioTimestamp,
    /// End time of the detected speech segment.
    pub end_time: AudioTimestamp,
    /// Aggregated confidence score derived from combined metrics.
    pub confidence: f32,
    /// Average energy observed within the segment.
    pub avg_energy: f32,
    /// Number of frames that contributed to the segment.
    pub frame_count: usize,
}

impl SpeechChunk {
    /// Duration of the speech segment.
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

        assert_eq!(segments.len(), 1);
        let seg = segments[0];
        assert!((seg.start_time.as_secs() - 0.5).abs() < 0.05, "{seg:?}");
        assert!((seg.end_time.as_secs() - 1.5).abs() < 0.1, "{seg:?}");
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
            assert!((seg.start_time.as_secs() - start).abs() < 0.05, "{seg:?}");
            assert!(
                (seg.end_time.as_secs() - (start + 0.6)).abs() < 0.1,
                "{seg:?}"
            );
        }
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
