//! Core VAD detection engine with dual-metric analysis.

use std::fmt;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::monitoring::{AtomicCounter, VADStats};
use crate::time::{AudioDuration, AudioInstant, AudioTimestamp};
use parking_lot::Mutex;
use realfft::{RealFftPlanner, RealToComplex};

use super::config::VadConfig;
use super::metrics::{AdaptiveThresholdSnapshot, VadMetricsCollector, VadMetricsSnapshot};

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

/// Energy level under which we consider a frame near-silence regardless of
/// normalization.
const SILENCE_ENERGY_GATE: f32 = 0.02;
/// Relative energy ratio below which we consider a frame near-silence.
const SILENCE_RELATIVE_GATE: f32 = 1.7;

/// Real-time voice activity detector combining energy and spectral flux
/// metrics.
pub struct VadDetector {
    config: VadConfig,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    metrics: Arc<dyn VadMetricsCollector>,
    processed_samples: AtomicCounter,
    energy_weight: f32,
    flux_weight: f32,
    state: Mutex<DetectorState>,
}

impl fmt::Debug for VadDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let processed_samples = self.processed_samples.get();
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
    /// Construct a new detector instance.
    pub fn new(config: VadConfig, metrics: Arc<dyn VadMetricsCollector>) -> Result<Self> {
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

        let previous_spectrum = {
            let tmp = fft.make_output_vec();
            vec![0.0; tmp.len()]
        };

        let state = DetectorState {
            energy_baseline: config.energy_floor.max(EPSILON),
            flux_baseline: config.flux_floor.max(EPSILON),
            dynamic_threshold: config.base_threshold.max(EPSILON),
            previous_spectrum,
            pre_emphasis_prev: 0.0,
            active_segment: None,
        };

        Ok(Self {
            config,
            fft,
            window,
            metrics,
            processed_samples: AtomicCounter::new(0),
            energy_weight,
            flux_weight,
            state: Mutex::new(state),
        })
    }

    /// Access detector configuration.
    #[must_use]
    pub fn config(&self) -> &VadConfig {
        &self.config
    }

    /// Return the start sample of the currently active speech segment, if any.
    #[must_use]
    pub fn active_segment_start_sample(&self) -> Option<usize> {
        let state = self.state.lock();
        state
            .active_segment
            .as_ref()
            .map(|segment| segment.start_sample)
    }

    /// Reset processed sample count and stream start time.
    pub fn reset(&mut self, stream_start_time: AudioTimestamp) {
        self.config.stream_start_time = stream_start_time;
        self.processed_samples.reset();
        let mut state = self.state.lock();
        state.active_segment = None;
        state.pre_emphasis_prev = 0.0;
    }

    /// Run detection on a slice of samples, returning detected speech segments.
    pub fn detect(&self, samples: &[f32]) -> Result<Vec<SpeechChunk>> {
        let detection_start = AudioInstant::now();
        let chunk_len = samples.len() as u64;

        let mut detector_state = self.state.lock();

        let chunk_start_sample = self.processed_samples.fetch_add(chunk_len) as usize;
        let chunk_end_sample = chunk_start_sample + samples.len();

        let frames = match self.frame_signal(samples, chunk_start_sample, &mut detector_state) {
            Ok(frames) => frames,
            Err(err) => {
                let _ = self.processed_samples.fetch_sub(chunk_len);
                drop(detector_state);
                return Err(err);
            }
        };

        if frames.is_empty() {
            let latency = AudioInstant::now().duration_since(detection_start);
            let adaptive = AdaptiveThresholdSnapshot {
                energy_baseline: detector_state.energy_baseline,
                flux_baseline: detector_state.flux_baseline,
                dynamic_threshold: detector_state.dynamic_threshold,
            };
            let snapshot = VadMetricsSnapshot::new(VADStats::new(), latency, adaptive);
            self.metrics.record_vad_metrics(&snapshot);
            drop(detector_state);
            return Ok(Vec::new());
        }

        let energy = Self::compute_energy(&frames);
        let flux = self.compute_spectral_flux(&frames, &mut detector_state)?;
        let (chunks, mut stats) = self.merge_metrics(
            &frames,
            &energy,
            &flux,
            chunk_end_sample,
            &mut detector_state,
        )?;
        stats.speech_frames = chunks.len() as u64;
        let adaptive = AdaptiveThresholdSnapshot {
            energy_baseline: detector_state.energy_baseline,
            flux_baseline: detector_state.flux_baseline,
            dynamic_threshold: detector_state.dynamic_threshold,
        };
        drop(detector_state);

        let latency = AudioInstant::now().duration_since(detection_start);
        let snapshot = VadMetricsSnapshot::new(stats, latency, adaptive);
        self.metrics.record_vad_metrics(&snapshot);

        Ok(chunks)
    }

    fn frame_signal(
        &self,
        samples: &[f32],
        absolute_start: usize,
        state: &mut DetectorState,
    ) -> Result<Vec<Frame>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let processed = self.preprocess_signal(samples, state);
        let frame_length = self.config.frame_length_samples()?;
        let hop_length = self.config.hop_length_samples()?;

        if frame_length == 0 {
            return Err(Error::Processing("frame length resolved to zero".into()));
        }

        let mut frames = Vec::new();
        let mut start = 0usize;

        while start + frame_length <= processed.len() {
            #[allow(clippy::indexing_slicing)] // bounds checked by while condition
            let slice = &processed[start..start + frame_length];
            let mut frame = Vec::with_capacity(frame_length);
            frame.extend(
                slice
                    .iter()
                    .zip(&self.window)
                    .map(|(sample, window)| sample * window),
            );
            frames.push(Frame {
                data: frame,
                start_sample: absolute_start + start,
                valid_len: frame_length,
            });
            start += hop_length;
        }

        if start < processed.len() {
            if let Some(slice) = processed.get(start..) {
                let available = slice.len().min(frame_length);
                let mut frame = Vec::with_capacity(frame_length);
                frame.extend(
                    slice
                        .iter()
                        .zip(&self.window)
                        .map(|(sample, window)| sample * window),
                );
                frame.resize(frame_length, 0.0);
                frames.push(Frame {
                    data: frame,
                    start_sample: absolute_start + start,
                    valid_len: available,
                });
            }
        }

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
    ) -> Result<(Vec<SpeechChunk>, VADStats)> {
        let mut stats = VADStats::new();
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
        let mut silence_run = active_segment
            .as_ref()
            .map_or(0usize, |state| state.silence_run);

        for (idx, frame) in frames.iter().enumerate() {
            let frame_start = AudioInstant::now();
            let raw_energy = energy.get(idx).copied().ok_or_else(|| {
                Error::Processing(format!("energy array length mismatch at index {idx}"))
            })?;
            let raw_flux = flux.get(idx).copied().ok_or_else(|| {
                Error::Processing(format!("flux array length mismatch at index {idx}"))
            })?;

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
            let low_energy = raw_energy < SILENCE_ENERGY_GATE;
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
                let segment_state = active_segment
                    .get_or_insert_with(|| ActiveSegmentState::new(frame.start_sample));
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
                silence_run = 0;
            }

            let _frame_processing = AudioInstant::now().duration_since(frame_start);
            stats.frames_processed += 1;

            // Update baselines only during silence to avoid noise floor contamination
            if !is_speech {
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

        // Preserve active segment for streaming continuity; flush to finalize
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
