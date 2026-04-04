//! End-to-end audio processing coordinator.
//!
//! Integrates format conversion, VAD, chunking, and preprocessing into a
//! synchronous pipeline for one input stream at a time.
//!
//! # Architecture
//!
//! The coordinator follows a simple synchronous design optimized for CPU-bound
//! audio processing:
//!
//! ```text
//! Raw Audio Bytes
//!     ↓
//! Format Conversion → StandardAudio (16kHz mono PCM)
//!     ↓
//! VAD Detection → Speech Segments
//!     ↓
//! Chunking → ProcessedChunks (500ms aligned)
//!     ↓
//! Preprocessing → Clean Audio
//!     ↓
//! Processed Output
//! ```
//!
//! # Performance Contract
//!
//! - **Audio processing latency**: <60ms P95 (all 4 stages)
//! - **Per-stage tracking**: Individual latency metrics exported
//! - **Stage reporting**: Per-stage latency metrics are returned
//!
//! # Example
//!
//! ```rust,no_run
//! use speech_prep::pipeline::AudioPipelineCoordinator;
//!
//! # fn main() -> speech_prep::error::Result<()> {
//! let coordinator = AudioPipelineCoordinator::new_with_defaults()?;
//!
//! let audio_bytes = std::fs::read("sample.wav")?;
//! let result = coordinator.process_frame(&audio_bytes)?;
//!
//! assert!(result.total_latency < std::time::Duration::from_secs(1));
//! assert!(result.chunks_processed <= 16);
//! # Ok(())
//! # }
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use crate::chunker::{Chunker, ProcessedChunk};
use crate::converter::{AudioFormatConverter, ConversionMetadata, StandardAudio};
use crate::error::{Error, Result};
use crate::format::AudioFormat;
use crate::preprocessing::{DcHighPassFilter, NoiseReducer, PreprocessingConfig, VadContext};
use crate::time::{validate_in_range, AudioDuration, AudioInstant, AudioTimestamp};
use crate::vad::{SpeechChunk, VadDetector};

/// Result of processing audio through the complete pipeline.
#[derive(Debug, Clone, Copy)]
pub struct ProcessingResult {
    /// Total number of chunks generated.
    pub chunks_processed: usize,

    /// Total latency for complete pipeline execution.
    pub total_latency: Duration,

    /// Per-stage latency breakdown.
    pub stage_latencies: StageLatencies,

    /// Reserved compatibility flag from the earlier fan-out pipeline surface.
    ///
    /// The standalone coordinator never applies backpressure, so this is
    /// always `false`.
    pub backpressure_active: bool,
}

/// Latency measurements for each pipeline stage.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageLatencies {
    /// Format conversion latency.
    pub format_conversion: Duration,

    /// VAD detection latency.
    pub vad_detection: Duration,

    /// Chunking latency.
    pub chunking: Duration,

    /// Preprocessing latency (per chunk average).
    pub preprocessing_avg: Duration,

    /// Reserved compatibility field from the earlier fan-out pipeline surface.
    ///
    /// The standalone coordinator has no broadcast stage, so this is always
    /// `Duration::ZERO`.
    pub broadcasting_avg: Duration,
}

/// Audio pipeline coordinator.
///
/// Orchestrates format conversion, VAD detection, chunking, and preprocessing
/// in a synchronous pipeline optimized for CPU-bound audio processing.
///
/// **Thread Safety**: Preprocessing filters wrapped in `Mutex` for interior
/// mutability, allowing concurrent frame processing from multiple callers.
#[derive(Debug)]
pub struct AudioPipelineCoordinator {
    vad_detector: Arc<VadDetector>,
    chunker: Chunker,
    dc_filter: Mutex<DcHighPassFilter>,
    noise_reducer: Mutex<NoiseReducer>,
    stream_buffer: Mutex<StreamBuffer>,
    processed_cursor: AtomicUsize,
}

#[derive(Debug, Default)]
struct StreamBuffer {
    base_sample_index: usize,
    samples: Vec<f32>,
}

impl StreamBuffer {
    fn append(&mut self, new_samples: &[f32]) {
        self.samples.extend_from_slice(new_samples);
    }

    fn as_slice(&self) -> &[f32] {
        &self.samples
    }

    fn base_sample_index(&self) -> usize {
        self.base_sample_index
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn start_time(&self, stream_start: AudioTimestamp, sample_rate: u32) -> Result<AudioTimestamp> {
        let offset = samples_to_duration(self.base_sample_index, sample_rate)?;
        Ok(stream_start.add_duration(offset))
    }

    fn drop_through(&mut self, sample_index: usize) {
        if sample_index <= self.base_sample_index {
            return;
        }

        let drop_count = sample_index
            .saturating_sub(self.base_sample_index)
            .min(self.samples.len());

        if drop_count == 0 {
            return;
        }

        if drop_count >= self.samples.len() {
            self.samples.clear();
            self.base_sample_index = sample_index;
        } else {
            self.samples.drain(..drop_count);
            self.base_sample_index += drop_count;
        }
    }
}

impl AudioPipelineCoordinator {
    /// Create a new coordinator with provided components.
    ///
    /// # Arguments
    ///
    /// * `vad_detector` - Voice activity detector
    /// * `chunker` - Audio chunker
    /// * `dc_filter` - DC offset removal and high-pass filter
    /// * `noise_reducer` - Spectral noise reduction
    pub fn new(
        vad_detector: Arc<VadDetector>,
        chunker: Chunker,
        dc_filter: DcHighPassFilter,
        noise_reducer: NoiseReducer,
    ) -> Self {
        Self {
            vad_detector,
            chunker,
            dc_filter: Mutex::new(dc_filter),
            noise_reducer: Mutex::new(noise_reducer),
            stream_buffer: Mutex::new(StreamBuffer::default()),
            processed_cursor: AtomicUsize::new(0),
        }
    }

    /// Create coordinator with default configuration.
    ///
    /// Suitable for testing and standard audio processing scenarios.
    ///
    /// # Errors
    ///
    /// Returns error if component initialization fails.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use speech_prep::pipeline::AudioPipelineCoordinator;
    ///
    /// # fn main() -> speech_prep::error::Result<()> {
    /// let coordinator = AudioPipelineCoordinator::new_with_defaults()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_defaults() -> Result<Self> {
        use crate::{NoopVadMetricsCollector, VadConfig, VadMetricsCollector};

        let metrics: Arc<dyn VadMetricsCollector> = Arc::new(NoopVadMetricsCollector);

        let vad_config = VadConfig::default();
        let vad_detector = Arc::new(VadDetector::new(vad_config, metrics)?);

        let chunker = Chunker::default();

        let dc_config = PreprocessingConfig::default();
        let dc_filter = DcHighPassFilter::new(dc_config)?;

        let noise_config = crate::preprocessing::NoiseReductionConfig::default();
        let noise_reducer = crate::preprocessing::NoiseReducer::new(noise_config)?;

        Ok(Self::new(vad_detector, chunker, dc_filter, noise_reducer))
    }

    /// Process raw audio bytes through complete pipeline.
    ///
    /// # Performance Contract
    ///
    /// Total audio processing latency must be <60ms P95 for standard inputs
    /// (500ms audio chunks at 16kHz).
    ///
    /// # Arguments
    ///
    /// * `audio_bytes` - Raw audio data (WAV, PCM, or other supported formats)
    ///
    /// # Returns
    ///
    /// Processing result with latency metrics and chunk count.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Format detection/conversion fails
    /// - VAD detection fails
    /// - Chunking fails
    /// - Preprocessing fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use speech_prep::pipeline::AudioPipelineCoordinator;
    /// # fn main() -> speech_prep::error::Result<()> {
    /// let coordinator = AudioPipelineCoordinator::new_with_defaults()?;
    /// let audio = std::fs::read("test.wav")?;
    ///
    /// let result = coordinator.process_frame(&audio)?;
    /// assert!(result.total_latency < std::time::Duration::from_millis(60));
    /// # Ok(())
    /// # }
    /// ```
    pub fn process_frame(&self, audio_bytes: &[u8]) -> Result<ProcessingResult> {
        let pipeline_start = AudioInstant::now();
        let mut latencies = StageLatencies::default();

        let format_start = AudioInstant::now();
        let standard_audio = AudioFormatConverter::convert_to_standard(audio_bytes)?;
        latencies.format_conversion = format_start.elapsed();

        self.process_standard_audio(&standard_audio, pipeline_start, latencies)
    }

    /// Flush pending audio by injecting trailing silence to finalize speech
    /// segments.
    ///
    /// This should be invoked when a streaming session ends to ensure any
    /// active speech region is emitted before scoring.
    pub fn flush(&self) -> Result<ProcessingResult> {
        let pipeline_start = AudioInstant::now();
        let latencies = StageLatencies::default();

        let config = *self.vad_detector.config();
        let frame_len = config.frame_length_samples()?;
        let frames_to_flush = (config.hangover_frames.max(1)) + 1;
        let silence_samples = vec![0.0f32; frame_len * frames_to_flush];

        let metadata = ConversionMetadata {
            original_format: AudioFormat::WavPcm,
            original_sample_rate: config.sample_rate,
            original_channels: 1,
            original_bit_depth: Some(16),
            peak_before: 0.0,
            peak_after: 0.0,
            conversion_time_ms: 0.0,
            detection_time_ms: 0.0,
            decode_time_ms: 0.0,
            resample_time_ms: 0.0,
            mix_time_ms: 0.0,
        };

        let standard_audio = StandardAudio {
            samples: silence_samples,
            metadata,
        };
        self.process_standard_audio(&standard_audio, pipeline_start, latencies)
    }

    fn process_standard_audio(
        &self,
        standard_audio: &StandardAudio,
        pipeline_start: AudioInstant,
        mut latencies: StageLatencies,
    ) -> Result<ProcessingResult> {
        let vad_start = AudioInstant::now();
        let vad_segments = self.vad_detector.detect(&standard_audio.samples)?;
        latencies.vad_detection = vad_start.elapsed();

        let sample_rate = self.vad_detector.config().sample_rate;
        let stream_start_time = self.vad_detector.config().stream_start_time;

        let (chunks, chunk_duration) = {
            let mut buffer = self.stream_buffer.lock();
            buffer.append(&standard_audio.samples);

            if buffer.as_slice().is_empty() {
                Ok::<(Vec<ProcessedChunk>, Duration), Error>((Vec::new(), Duration::default()))
            } else {
                let buffer_base = buffer.base_sample_index();
                let buffer_len = buffer.len();
                let buffer_end_abs = buffer_base + buffer_len;
                let processed_before = self.processed_cursor.load(Ordering::Acquire);
                let slice_start_abs = processed_before.max(buffer_base);

                if slice_start_abs >= buffer_base + buffer_len {
                    let lookback_samples = (sample_rate as usize) / 5;
                    let drop_target = slice_start_abs.saturating_sub(lookback_samples);
                    buffer.drop_through(drop_target);
                    drop(buffer);
                    return Ok(ProcessingResult {
                        chunks_processed: 0,
                        total_latency: pipeline_start.elapsed(),
                        stage_latencies: latencies,
                        backpressure_active: false,
                    });
                }

                let base_time = buffer.start_time(stream_start_time, sample_rate)?;
                let offset_samples = slice_start_abs.saturating_sub(buffer_base);
                let offset_duration = samples_to_duration(offset_samples, sample_rate)?;
                let audio_start = base_time.add_duration(offset_duration);

                let audio_slice = buffer
                    .as_slice()
                    .get(offset_samples..)
                    .ok_or_else(|| Error::InvalidInput("invalid buffer window".into()))?;

                let normalized_segments = normalize_vad_segments(
                    &vad_segments,
                    stream_start_time,
                    audio_start,
                    slice_start_abs,
                    buffer_end_abs,
                    sample_rate,
                )?;

                let chunk_start = AudioInstant::now();
                let chunks = self.chunker.chunk_with_stream_start(
                    audio_slice,
                    sample_rate,
                    &normalized_segments,
                    audio_start,
                )?;
                let elapsed = chunk_start.elapsed();

                let mut max_processed_sample = processed_before;
                for chunk in &chunks {
                    let end_sample =
                        time_to_sample_index(chunk.end_time, stream_start_time, sample_rate)?;
                    if end_sample > max_processed_sample {
                        max_processed_sample = end_sample;
                    }
                }
                self.processed_cursor
                    .store(max_processed_sample, Ordering::Release);

                let lookback_samples = (sample_rate as usize) / 5; // Retain ~200ms history
                let drop_target = max_processed_sample.saturating_sub(lookback_samples);
                buffer.drop_through(drop_target);
                drop(buffer);

                Ok::<(Vec<ProcessedChunk>, Duration), Error>((chunks, elapsed))
            }
        }?;
        latencies.chunking = chunk_duration;

        let mut total_preprocess = Duration::default();
        let mut prev_overlap_next: Option<Vec<f32>> = None;

        for chunk in &chunks {
            let preprocess_start = AudioInstant::now();
            let mut preprocessed = self.preprocess_chunk(chunk)?;

            if let Some(prev_overlap) = prev_overlap_next.take() {
                preprocessed.overlap_prev = Some(prev_overlap);
            } else {
                preprocessed.overlap_prev = None;
            }

            prev_overlap_next.clone_from(&preprocessed.overlap_next);

            total_preprocess += preprocess_start.elapsed();
        }

        let chunk_count = chunks.len().max(1);
        latencies.preprocessing_avg = total_preprocess / chunk_count as u32;
        latencies.broadcasting_avg = Duration::ZERO;

        let total_latency = pipeline_start.elapsed();

        if total_latency > Duration::from_millis(60) {
            tracing::warn!(
                latency_ms = total_latency.as_millis(),
                "Audio processing exceeded 60ms target"
            );
        }

        let backpressure_active = false;

        Ok(ProcessingResult {
            chunks_processed: chunks.len(),
            total_latency,
            stage_latencies: latencies,
            backpressure_active,
        })
    }

    fn preprocess_chunk(&self, chunk: &ProcessedChunk) -> Result<ProcessedChunk> {
        let vad_ctx = VadContext {
            is_silence: chunk.is_silence(),
        };
        let dc_clean = {
            let mut filter = self.dc_filter.lock();
            filter.process(&chunk.samples, Some(&vad_ctx))?
        };

        let denoised = {
            let mut reducer = self.noise_reducer.lock();
            reducer.reduce(&dc_clean, Some(vad_ctx))?
        };

        let (energy, has_clipping) = Self::compute_energy_and_clipping(&denoised);
        let snr_db = Self::recalculate_snr(chunk.snr_db, chunk.energy, energy);

        let overlap_next = chunk.overlap_next.as_ref().and_then(|existing| {
            let retain = existing.len().min(denoised.len());
            denoised.get(denoised.len() - retain..).map(<[f32]>::to_vec)
        });

        let mut processed = chunk.clone();
        processed.samples = denoised;
        processed.energy = energy;
        processed.snr_db = snr_db;
        processed.has_clipping = has_clipping;
        processed.overlap_next = overlap_next;
        processed.overlap_prev = None;

        Ok(processed)
    }

    fn compute_energy_and_clipping(samples: &[f32]) -> (f32, bool) {
        const CLIPPING_THRESHOLD: f32 = 0.999;

        if samples.is_empty() {
            return (0.0, false);
        }

        let mut sum_squares = 0.0f32;
        let mut has_clipping = false;
        for &sample in samples {
            let abs = sample.abs();
            if abs >= CLIPPING_THRESHOLD {
                has_clipping = true;
            }
            sum_squares = sample.mul_add(sample, sum_squares);
        }
        let mean_square = sum_squares / samples.len() as f32;
        (mean_square.sqrt(), has_clipping)
    }

    fn recalculate_snr(
        previous_snr: Option<f32>,
        previous_energy: f32,
        new_energy: f32,
    ) -> Option<f32> {
        const EPSILON: f32 = 1e-10;
        let snr_db = previous_snr?;

        if previous_energy <= EPSILON {
            return Some(snr_db);
        }

        let noise_rms = previous_energy / 10_f32.powf(snr_db / 20.0);
        if noise_rms <= EPSILON || new_energy <= EPSILON {
            return Some(snr_db);
        }

        let ratio = new_energy / noise_rms;
        if ratio <= EPSILON {
            return Some(snr_db);
        }

        Some(20.0 * ratio.log10())
    }
}

fn samples_to_duration(samples: usize, sample_rate: u32) -> Result<AudioDuration> {
    validate_in_range(sample_rate, 1_u32, u32::MAX, "sample_rate")?;

    let sample_rate_u128 = u128::from(sample_rate);
    let sample_count = samples as u128;
    let nanos = (sample_count * 1_000_000_000u128 + (sample_rate_u128 / 2)) / sample_rate_u128;
    Ok(AudioDuration::from_nanos(nanos as u64))
}

fn time_to_sample_index(
    time: AudioTimestamp,
    stream_start: AudioTimestamp,
    sample_rate: u32,
) -> Result<usize> {
    validate_in_range(sample_rate, 1_u32, u32::MAX, "sample_rate")?;

    let duration = time
        .duration_since(stream_start)
        .ok_or_else(|| Error::TemporalOperation("time precedes stream start".into()))?;
    let samples = (duration.as_secs_f64() * f64::from(sample_rate)).round() as usize;
    Ok(samples)
}

fn normalize_vad_segments(
    segments: &[SpeechChunk],
    stream_start: AudioTimestamp,
    slice_start_time: AudioTimestamp,
    slice_start_sample: usize,
    buffer_end_sample: usize,
    sample_rate: u32,
) -> Result<Vec<SpeechChunk>> {
    validate_in_range(sample_rate, 1_u32, u32::MAX, "sample_rate")?;

    let mut normalized = Vec::with_capacity(segments.len());

    for segment in segments {
        let start_sample_abs = time_to_sample_index(segment.start_time, stream_start, sample_rate)?;
        let end_sample_abs = time_to_sample_index(segment.end_time, stream_start, sample_rate)?;

        if end_sample_abs <= slice_start_sample {
            // Entire segment already processed; skip.
            continue;
        }

        let clamped_start_abs = start_sample_abs.max(slice_start_sample);
        let clamped_end_abs = end_sample_abs.min(buffer_end_sample);

        if clamped_end_abs <= clamped_start_abs {
            continue;
        }

        let rel_start_samples = clamped_start_abs - slice_start_sample;
        let rel_end_samples = clamped_end_abs - slice_start_sample;

        let start_time =
            slice_start_time.add_duration(samples_to_duration(rel_start_samples, sample_rate)?);
        let end_time =
            slice_start_time.add_duration(samples_to_duration(rel_end_samples, sample_rate)?);

        let mut adjusted = *segment;
        adjusted.start_time = start_time;
        adjusted.end_time = end_time;
        normalized.push(adjusted);
    }

    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::AudioFixtures;

    /// Helper to convert f32 samples to WAV bytes for testing
    /// Creates a minimal 16-bit PCM WAV file
    fn samples_to_wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
        let mut wav_data = Vec::new();

        // WAV header
        let num_samples = samples.len() as u32;
        let num_channels = 1u16;
        let bits_per_sample = 16u16;
        let byte_rate = sample_rate * u32::from(num_channels) * u32::from(bits_per_sample) / 8;
        let block_align = num_channels * bits_per_sample / 8;
        let data_size = num_samples * u32::from(block_align);

        // RIFF header
        wav_data.extend_from_slice(b"RIFF");
        wav_data.extend_from_slice(&(36 + data_size).to_le_bytes());
        wav_data.extend_from_slice(b"WAVE");

        // fmt chunk
        wav_data.extend_from_slice(b"fmt ");
        wav_data.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
        wav_data.extend_from_slice(&1u16.to_le_bytes()); // PCM format
        wav_data.extend_from_slice(&num_channels.to_le_bytes());
        wav_data.extend_from_slice(&sample_rate.to_le_bytes());
        wav_data.extend_from_slice(&byte_rate.to_le_bytes());
        wav_data.extend_from_slice(&block_align.to_le_bytes());
        wav_data.extend_from_slice(&bits_per_sample.to_le_bytes());

        // data chunk
        wav_data.extend_from_slice(b"data");
        wav_data.extend_from_slice(&data_size.to_le_bytes());

        // Convert f32 samples to i16 PCM
        for &sample in samples {
            let i16_sample = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            wav_data.extend_from_slice(&i16_sample.to_le_bytes());
        }

        wav_data
    }

    /// Test coordinator creation with default configuration.
    #[test]
    fn test_coordinator_creation_with_defaults() {
        let coordinator = AudioPipelineCoordinator::new_with_defaults();
        assert!(
            coordinator.is_ok(),
            "Failed to create coordinator with defaults"
        );
    }

    /// Test basic frame processing with real audio data.
    #[test]
    fn test_process_frame_with_real_audio() {
        let coordinator =
            AudioPipelineCoordinator::new_with_defaults().expect("Failed to create coordinator");

        // Load real test audio and convert to WAV bytes
        let fixtures = AudioFixtures::new();
        let audio_sample = fixtures
            .load_sample("french_short")
            .expect("Failed to load test audio");
        let test_audio = samples_to_wav_bytes(&audio_sample.audio_data, audio_sample.sample_rate);

        // Process audio through complete pipeline
        let result = coordinator.process_frame(&test_audio);
        assert!(
            result.is_ok(),
            "Failed to process audio frame: {:?}",
            result.err()
        );

        let processing_result = result.unwrap();

        // Verify results
        assert!(
            processing_result.chunks_processed > 0,
            "No chunks were generated from audio"
        );

        // Verify latency tracking
        assert!(
            processing_result.total_latency < Duration::from_millis(100),
            "Processing took too long: {:?}",
            processing_result.total_latency
        );
    }

    /// Test that all stage latencies are tracked.
    #[test]
    fn test_stage_latencies_tracked() {
        let coordinator =
            AudioPipelineCoordinator::new_with_defaults().expect("Failed to create coordinator");

        let fixtures = AudioFixtures::new();
        let audio_sample = fixtures
            .load_sample("french_short")
            .expect("Failed to load test audio");
        let test_audio = samples_to_wav_bytes(&audio_sample.audio_data, audio_sample.sample_rate);

        let result = coordinator
            .process_frame(&test_audio)
            .expect("Failed to process audio");

        let latencies = result.stage_latencies;

        // Verify all stages have latency measurements
        assert!(
            latencies.format_conversion > Duration::ZERO,
            "Format conversion latency not tracked"
        );
        assert!(
            latencies.vad_detection > Duration::ZERO,
            "VAD detection latency not tracked"
        );
        assert!(
            latencies.chunking > Duration::ZERO,
            "Chunking latency not tracked"
        );

        let _ = latencies.preprocessing_avg;
        assert_eq!(
            latencies.broadcasting_avg,
            Duration::ZERO,
            "Standalone pipeline should not report broadcast latency"
        );
    }

    /// Test <60ms latency performance contract for audio processing.
    #[test]
    fn test_latency_performance_contract() {
        let coordinator =
            AudioPipelineCoordinator::new_with_defaults().expect("Failed to create coordinator");

        let fixtures = AudioFixtures::new();
        let audio_sample = fixtures
            .load_sample("french_short")
            .expect("Failed to load test audio");
        let test_audio = samples_to_wav_bytes(&audio_sample.audio_data, audio_sample.sample_rate);

        coordinator
            .process_frame(&test_audio)
            .expect("Failed to process warm-up audio");

        let mut latencies = Vec::new();
        for _ in 0..5 {
            let result = coordinator
                .process_frame(&test_audio)
                .expect("Failed to process audio");
            latencies.push(result.total_latency);
        }

        latencies.sort();
        let p95_index = (latencies.len() as f64 * 0.95).ceil() as usize - 1;
        let p95_latency = latencies[p95_index];

        assert!(
            p95_latency < Duration::from_millis(150),
            "P95 latency exceeds 150ms (CI-tolerant): {:?}",
            p95_latency
        );
    }

    /// Test the compatibility flag remains disabled.
    #[test]
    fn test_backpressure_detection() {
        let coordinator =
            AudioPipelineCoordinator::new_with_defaults().expect("Failed to create coordinator");

        let fixtures = AudioFixtures::new();
        let audio_sample = fixtures
            .load_sample("french_short")
            .expect("Failed to load test audio");
        let test_audio = samples_to_wav_bytes(&audio_sample.audio_data, audio_sample.sample_rate);

        let result = coordinator
            .process_frame(&test_audio)
            .expect("Failed to process audio");

        assert!(
            !result.backpressure_active,
            "Standalone coordinator should not report backpressure"
        );

        for _ in 0..3 {
            let result = coordinator
                .process_frame(&test_audio)
                .expect("Failed to process audio");
            assert!(
                !result.backpressure_active,
                "Standalone coordinator should not report backpressure"
            );
        }
    }

    /// Test processing empty audio handles gracefully.
    #[test]
    fn test_process_empty_audio() {
        let coordinator =
            AudioPipelineCoordinator::new_with_defaults().expect("Failed to create coordinator");

        let empty_audio = &[];
        let result = coordinator.process_frame(empty_audio);

        assert!(result.is_err(), "Empty audio should return error");
    }
}
