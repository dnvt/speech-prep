//! High-level audio format conversion pipeline.
//!
//! This module provides a unified API for converting arbitrary audio formats
//! to standard format: mono, 16kHz, normalized f32 samples.
//!
//! ## Pipeline Stages
//!
//! 1. **Format Detection**: Identify audio container format (WAV, MP3, FLAC,
//!    etc.)
//! 2. **Decoding**: Extract PCM samples from container (supports 16/24-bit)
//! 3. **Resampling**: Convert to 16kHz standard rate (linear interpolation)
//! 4. **Channel Mixing**: Downmix to mono (simple averaging)
//!
//! ## Performance Contract
//!
//! - **Target Latency**: <10ms for 3-second audio clip
//! - **Memory**: Streaming-friendly, minimal allocations
//! - **Quality**: RMS error <0.01, zero clipping
//!
//! ## Example
//!
//! ```rust,no_run
//! use speech_prep::converter::AudioFormatConverter;
//!
//! let audio_bytes = std::fs::read("recording.wav")?;
//! let standard = AudioFormatConverter::convert_to_standard(&audio_bytes)?;
//!
//! println!(
//!     "Converted {} samples from {} to mono 16kHz",
//!     standard.samples.len(),
//!     standard.metadata.original_format
//! );
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{Error, Result};
use crate::time::{AudioDuration, AudioInstant};

use crate::decoder::{ChannelMixer, SampleRateConverter, WavDecoder};
use crate::format::{AudioFormat, FormatDetector};

/// Standardized audio output: mono, 16kHz, normalized samples.
///
/// This is the canonical format for all audio processing,
/// designed to be consumed by downstream scoring and analysis, and other
/// downstream components.
#[derive(Debug, Clone, PartialEq)]
pub struct StandardAudio {
    /// Mono audio samples at 16kHz, normalized to [-1.0, 1.0].
    pub samples: Vec<f32>,
    /// Metadata tracking the conversion journey and quality metrics.
    pub metadata: ConversionMetadata,
}

impl StandardAudio {
    /// Total number of mono samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Duration in seconds at 16kHz.
    #[must_use]
    pub fn duration_sec(&self) -> f64 {
        self.samples.len() as f64 / 16000.0
    }

    /// Check if the audio is effectively silent (all samples near zero).
    #[must_use]
    pub fn is_silent(&self) -> bool {
        self.samples.iter().all(|&s| s.abs() < 1e-4)
    }
}

/// Metadata tracking the complete conversion pipeline journey.
///
/// Captures information from all 4 pipeline stages to enable debugging,
/// quality validation, and observability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConversionMetadata {
    /// Detected audio format (WAV, MP3, FLAC, etc.).
    pub original_format: AudioFormat,
    /// Original sample rate in Hz before resampling.
    pub original_sample_rate: u32,
    /// Original number of channels before mixing.
    pub original_channels: u8,
    /// Original bit depth (if applicable, e.g., 16, 24 for PCM).
    pub original_bit_depth: Option<u16>,
    /// Peak amplitude in original audio before any processing.
    pub peak_before: f32,
    /// Peak amplitude after complete conversion pipeline.
    pub peak_after: f32,
    /// Total time spent in conversion pipeline (all 4 stages).
    pub conversion_time_ms: f64,
    /// Time spent in format detection stage.
    pub detection_time_ms: f64,
    /// Time spent in decoding stage.
    pub decode_time_ms: f64,
    /// Time spent in resampling stage.
    pub resample_time_ms: f64,
    /// Time spent in channel mixing stage.
    pub mix_time_ms: f64,
}

impl ConversionMetadata {
    /// Check if any stage exceeded expected latency budget.
    ///
    /// Expected budget breakdown for 3s clip:
    /// - Detection: <1ms
    /// - Decoding: <3ms
    /// - Resampling: <5ms
    /// - Mixing: <1ms
    /// - **Total: <10ms**
    #[must_use]
    pub fn has_performance_issue(&self) -> bool {
        self.conversion_time_ms > 10.0
            || self.detection_time_ms > 1.0
            || self.decode_time_ms > 3.0
            || self.resample_time_ms > 5.0
            || self.mix_time_ms > 1.0
    }

    /// Calculate the peak amplitude reduction ratio from conversion.
    ///
    /// Returns the ratio of final peak to original peak. Values:
    /// - 1.0 = no amplitude change
    /// - <1.0 = amplitude reduced (common with averaging/resampling)
    /// - >1.0 = amplitude increased (rare, may indicate issue)
    #[must_use]
    pub fn peak_ratio(&self) -> f32 {
        if self.peak_before.abs() < f32::EPSILON {
            1.0 // Avoid division by zero for silent input
        } else {
            self.peak_after / self.peak_before
        }
    }
}

/// High-level audio format converter.
///
/// Provides a unified API for converting arbitrary audio formats to the
/// standard format: mono, 16kHz, normalized f32 samples.
///
/// ## Pipeline Architecture
///
/// ```text
/// Input Bytes
///     ↓
/// [Format Detection] ← 6 formats: WAV, MP3, FLAC, Opus, WebM, AAC
///     ↓
/// [WAV Decoding] ← 16/24-bit PCM normalization
///     ↓
/// [Resampling] ← Arbitrary rate → 16kHz (linear interpolation)
///     ↓
/// [Channel Mixing] ← Multi-channel → Mono (simple averaging)
///     ↓
/// StandardAudio (mono, 16kHz, f32)
/// ```
///
/// ## Current Scope Limitations
///
/// - **Formats**: Only WAV decoding implemented; other formats detected but not
///   decoded
/// - **Channel Counts**: 1, 2, 4, 6 channels supported
/// - **Bit Depths**: 16-bit and 24-bit PCM only
/// - **Resampling**: Linear interpolation (sinc reserved for future)
///
/// ## Example
///
/// ```rust,no_run
/// use speech_prep::converter::AudioFormatConverter;
///
/// // Convert any supported audio to standard format
/// let wav_bytes = std::fs::read("audio.wav")?;
/// let standard = AudioFormatConverter::convert_to_standard(&wav_bytes)?;
///
/// // Access standardized samples and metadata
/// println!("Format: {}", standard.metadata.original_format);
/// println!("Converted {} Hz → 16kHz", standard.metadata.original_sample_rate);
/// println!("Converted {} ch → mono", standard.metadata.original_channels);
/// println!("Conversion time: {:.2}ms", standard.metadata.conversion_time_ms);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct AudioFormatConverter;

impl AudioFormatConverter {
    /// Create a new audio format converter instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Convert arbitrary audio bytes to standard format: mono, 16kHz, f32.
    ///
    /// This is the primary entry point for the audio normalization pipeline.
    /// It composes all 4 stages: format detection, decoding, resampling, and
    /// channel mixing.
    ///
    /// # Arguments
    ///
    /// * `audio_bytes` - Raw audio file bytes (any supported format)
    ///
    /// # Returns
    ///
    /// `StandardAudio` with mono 16kHz samples and complete conversion
    /// metadata.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if:
    /// - Format detection fails (not a recognized audio format)
    /// - Format is detected but not WAV (only WAV decoding supported)
    /// - WAV decoding fails (malformed file, unsupported codec)
    /// - Resampling fails (invalid sample rates)
    /// - Channel mixing fails (unsupported channel count)
    ///
    /// # Performance
    ///
    /// Target: <10ms for 3-second audio clip on reference hardware.
    /// Actual timing captured in `ConversionMetadata.conversion_time_ms`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use speech_prep::converter::AudioFormatConverter;
    ///
    /// let audio_bytes = std::fs::read("recording.wav")?;
    /// let standard = AudioFormatConverter::convert_to_standard(&audio_bytes)?;
    ///
    /// assert_eq!(standard.metadata.original_format.as_str(), "wav");
    /// assert!(standard.samples.len() > 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[allow(clippy::cognitive_complexity)] // Large function — audio format conversion pipeline
    pub fn convert_to_standard(audio_bytes: &[u8]) -> Result<StandardAudio> {
        let pipeline_start = AudioInstant::now();

        tracing::debug!(
            audio_bytes_len = audio_bytes.len(),
            "Starting audio format conversion pipeline"
        );

        let detection_start = AudioInstant::now();
        let format_metadata = FormatDetector::detect(audio_bytes)?;
        let detection_duration = elapsed_since(detection_start);
        let detection_time_ms = detection_duration.as_secs_f64() * 1000.0;

        tracing::debug!(
            format = %format_metadata.format,
            detection_time_ms,
            "Format detection complete"
        );

        if format_metadata.format != AudioFormat::WavPcm {
            return Err(Error::InvalidInput(format!(
                "unsupported format for decoding: {} (only WAV supported)",
                format_metadata.format.as_str()
            )));
        }

        let decode_start = AudioInstant::now();
        let decoded = WavDecoder::decode(audio_bytes)?;
        let decode_duration = elapsed_since(decode_start);
        let decode_time_ms = decode_duration.as_secs_f64() * 1000.0;

        tracing::debug!(
            sample_rate = decoded.sample_rate,
            channels = decoded.channels,
            bit_depth = decoded.bit_depth,
            sample_count = decoded.samples.len(),
            decode_time_ms,
            "WAV decoding complete"
        );

        let peak_before = decoded
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        let resample_start = AudioInstant::now();
        let resampled = SampleRateConverter::resample(
            &decoded.samples,
            decoded.channels,
            decoded.sample_rate,
            16000,
        )?;
        let resample_duration = elapsed_since(resample_start);
        let resample_time_ms = resample_duration.as_secs_f64() * 1000.0;

        tracing::debug!(
            input_rate = decoded.sample_rate,
            output_rate = 16000,
            output_samples = resampled.len(),
            resample_time_ms,
            "Sample rate conversion complete"
        );

        let mix_start = AudioInstant::now();
        let mixed = ChannelMixer::mix_to_mono(&resampled, decoded.channels)?;
        let mix_duration = elapsed_since(mix_start);
        let mix_time_ms = mix_duration.as_secs_f64() * 1000.0;

        tracing::debug!(
            input_channels = decoded.channels,
            output_samples = mixed.samples.len(),
            peak_before_mix = mixed.peak_before_mix,
            peak_after_mix = mixed.peak_after_mix,
            mix_time_ms,
            "Channel mixing complete"
        );

        let conversion_duration = elapsed_since(pipeline_start);
        let conversion_time_ms = conversion_duration.as_secs_f64() * 1000.0;

        if conversion_time_ms > 10.0 {
            tracing::warn!(
                conversion_time_ms,
                detection_time_ms,
                decode_time_ms,
                resample_time_ms,
                mix_time_ms,
                "Audio conversion exceeded 10ms target latency"
            );
        } else {
            tracing::debug!(conversion_time_ms, "Audio conversion pipeline complete");
        }

        let metadata = ConversionMetadata {
            original_format: format_metadata.format,
            original_sample_rate: decoded.sample_rate,
            original_channels: decoded.channels,
            original_bit_depth: Some(decoded.bit_depth),
            peak_before,
            peak_after: mixed.peak_after_mix,
            conversion_time_ms,
            detection_time_ms,
            decode_time_ms,
            resample_time_ms,
            mix_time_ms,
        };

        Ok(StandardAudio {
            samples: mixed.samples,
            metadata,
        })
    }
}

fn elapsed_since(start: AudioInstant) -> AudioDuration {
    AudioInstant::now().duration_since(start)
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult<T> = std::result::Result<T, String>;

    /// Create a minimal valid WAV file for testing.
    fn create_test_wav(sample_rate: u32, channels: u16, samples: &[i16]) -> TestResult<Vec<u8>> {
        let spec = hound::WavSpec {
            sample_rate,
            channels,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut cursor = std::io::Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|e| format!("failed to create WAV writer: {e}"))?;

        for &sample in samples {
            writer
                .write_sample(sample)
                .map_err(|e| format!("failed to write sample: {e}"))?;
        }

        writer
            .finalize()
            .map_err(|e| format!("failed to finalize WAV: {e}"))?;

        Ok(cursor.into_inner())
    }

    #[test]
    fn test_convert_mono_16khz_identity() -> TestResult<()> {
        // Already in standard format: mono, 16kHz
        let samples = vec![100i16, 200, -100, -200]; // Small amplitude
        let wav = create_test_wav(16000, 1, &samples)?;

        let standard =
            AudioFormatConverter::convert_to_standard(&wav).map_err(|e| e.to_string())?;

        // Should have 4 samples (already 16kHz mono)
        assert_eq!(standard.samples.len(), 4);
        assert_eq!(standard.metadata.original_sample_rate, 16000);
        assert_eq!(standard.metadata.original_channels, 1);
        assert_eq!(standard.metadata.original_format, AudioFormat::WavPcm);

        Ok(())
    }

    #[test]
    fn test_convert_stereo_44100_to_standard() -> TestResult<()> {
        // Stereo 44.1kHz → mono 16kHz
        let samples = vec![1000i16, -1000, 2000, -2000]; // 2 stereo frames
        let wav = create_test_wav(44100, 2, &samples)?;

        let standard =
            AudioFormatConverter::convert_to_standard(&wav).map_err(|e| e.to_string())?;

        // Should have ~1 sample after downsampling 44.1kHz → 16kHz and mixing stereo →
        // mono Original: 2 frames at 44.1kHz = ~0.045ms
        // At 16kHz: ~0.045ms * 16000 = ~0.72 samples (rounds to 1)
        assert!(!standard.samples.is_empty());
        assert_eq!(standard.metadata.original_sample_rate, 44100);
        assert_eq!(standard.metadata.original_channels, 2);

        Ok(())
    }

    #[test]
    fn test_convert_tracks_timing() -> TestResult<()> {
        let samples = vec![0i16; 1000]; // 1000 samples
        let wav = create_test_wav(16000, 1, &samples)?;

        let standard =
            AudioFormatConverter::convert_to_standard(&wav).map_err(|e| e.to_string())?;

        // All timing fields should be populated
        assert!(standard.metadata.detection_time_ms >= 0.0);
        assert!(standard.metadata.decode_time_ms >= 0.0);
        assert!(standard.metadata.resample_time_ms >= 0.0);
        assert!(standard.metadata.mix_time_ms >= 0.0);
        assert!(standard.metadata.conversion_time_ms >= 0.0);

        // Total time should be sum of stages (approximately, with measurement overhead)
        let stage_sum = standard.metadata.detection_time_ms
            + standard.metadata.decode_time_ms
            + standard.metadata.resample_time_ms
            + standard.metadata.mix_time_ms;

        assert!(
            (standard.metadata.conversion_time_ms - stage_sum).abs() < 1.0,
            "total time {} should approximately equal stage sum {}",
            standard.metadata.conversion_time_ms,
            stage_sum
        );

        Ok(())
    }

    #[test]
    fn test_convert_tracks_peaks() -> TestResult<()> {
        // Create audio with known peak
        let samples = vec![10000i16, -10000, 5000, -5000]; // Peak: 10000/32768 ≈ 0.305
        let wav = create_test_wav(16000, 1, &samples)?;

        let standard =
            AudioFormatConverter::convert_to_standard(&wav).map_err(|e| e.to_string())?;

        // Should track peaks
        assert!(standard.metadata.peak_before > 0.0);
        assert!(standard.metadata.peak_after > 0.0);

        // Peak should be approximately 0.305
        assert!(
            (standard.metadata.peak_before - 0.305).abs() < 0.01,
            "expected peak ~0.305, got {}",
            standard.metadata.peak_before
        );

        Ok(())
    }

    #[test]
    fn test_convert_rejects_non_wav() {
        // Create fake MP3 header
        let mp3_bytes = vec![0xFF, 0xFB, 0x90, 0x00]; // Valid MP3 frame start

        let result = AudioFormatConverter::convert_to_standard(&mp3_bytes);

        // Should detect as MP3 but reject for decoding (only WAV supported)
        assert!(result.is_err());
        if let Err(err) = result {
            let err_msg = err.to_string();
            assert!(err_msg.contains("MP3") || err_msg.contains("unsupported"));
        }
    }

    #[test]
    fn test_standard_audio_duration_calculation() -> TestResult<()> {
        let samples = vec![0i16; 16000]; // 1 second at 16kHz
        let wav = create_test_wav(16000, 1, &samples)?;

        let standard =
            AudioFormatConverter::convert_to_standard(&wav).map_err(|e| e.to_string())?;

        // Should be 1.0 second
        assert!((standard.duration_sec() - 1.0).abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_standard_audio_is_silent_detection() -> TestResult<()> {
        let silent_samples = vec![0i16; 100];
        let wav = create_test_wav(16000, 1, &silent_samples)?;

        let standard =
            AudioFormatConverter::convert_to_standard(&wav).map_err(|e| e.to_string())?;

        assert!(standard.is_silent());

        Ok(())
    }

    #[test]
    fn test_conversion_metadata_peak_ratio() -> TestResult<()> {
        let samples = vec![10000i16, -10000];
        let wav = create_test_wav(16000, 1, &samples)?;

        let standard =
            AudioFormatConverter::convert_to_standard(&wav).map_err(|e| e.to_string())?;

        // Peak ratio should be close to 1.0 for mono 16kHz (no mixing/resampling
        // changes)
        assert!(
            (standard.metadata.peak_ratio() - 1.0).abs() < 0.1,
            "expected peak ratio ~1.0, got {}",
            standard.metadata.peak_ratio()
        );

        Ok(())
    }
}
