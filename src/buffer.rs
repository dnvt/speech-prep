//! Audio buffer types for batch audio processing.
//!
//! Provides `AudioBuffer` for owned sample buffers, complementing streaming
//! audio types.
//!
//! ## Architecture
//!
//! - **AudioChunk**: Streaming-oriented transport
//! - **AudioBuffer** (this module): Owned samples for batch-style APIs
//!
//! ## Design Principles
//!
//! - Zero-copy conversions from AudioChunk where possible
//! - Temporal type integration (AudioDuration)
//! - Validation and normalization helpers

use crate::error::{Error, Result};
use crate::time::AudioDuration;
use crate::types::AudioChunk;

/// Sample rates approved for ingestion across the stack.
const VALID_SAMPLE_RATES: [u32; 5] = [8000, 16000, 22050, 44100, 48000];

/// Audio buffer for batch processing.
///
/// Provides utilities for normalization, resampling, and format conversion.
///
/// # Example
///
/// ```rust
/// use speech_prep::buffer::AudioBuffer;
///
/// let samples = vec![0.1, 0.2, -0.1, -0.2];
/// let buffer = AudioBuffer::from_samples(samples, 16000)?;
///
/// assert_eq!(buffer.sample_rate(), 16000);
/// assert_eq!(buffer.len(), 4);
/// # Ok::<(), speech_prep::error::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Audio samples (f32, normalized to [-1.0, 1.0])
    samples: Vec<f32>,
    /// Sample rate in Hz
    sample_rate: u32,
    /// Optional metadata
    metadata: Option<AudioBufferMetadata>,
}

/// Audio buffer metadata for tracking processing history.
#[derive(Debug, Clone, Default)]
pub struct AudioBufferMetadata {
    /// Source identifier (file, stream, etc.)
    pub source: Option<String>,
    /// Original sample rate (before resampling)
    pub original_sr: Option<u32>,
    /// Duration in seconds
    pub duration: Option<AudioDuration>,
    /// Whether audio has been normalized
    pub normalized: bool,
    /// Processing operations applied
    pub processing_chain: Vec<String>,
}

impl AudioBuffer {
    /// Create `AudioBuffer` from f32 samples
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples (will be validated)
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Errors
    ///
    /// Returns error if samples are empty or sample rate is invalid
    ///
    /// # Example
    ///
    /// ```rust
    /// use speech_prep::buffer::AudioBuffer;
    ///
    /// let samples = vec![0.1, 0.2, -0.1];
    /// let buffer = AudioBuffer::from_samples(samples, 16000)?;
    /// assert_eq!(buffer.len(), 3);
    /// # Ok::<(), speech_prep::error::Error>(())
    /// ```
    pub fn from_samples(samples: Vec<f32>, sample_rate: u32) -> Result<Self> {
        Self::validate_sample_rate(sample_rate)?;

        if samples.is_empty() {
            return Err(Error::empty_input("audio samples are empty"));
        }

        Self::validate_sample_values(&samples)?;

        Ok(Self {
            samples,
            sample_rate,
            metadata: Some(AudioBufferMetadata::default()),
        })
    }

    /// Create `AudioBuffer` from `AudioChunk` (zero-copy data)
    ///
    /// Converts a streaming `AudioChunk` into an owned `AudioBuffer`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use speech_prep::buffer::AudioBuffer;
    /// use speech_prep::types::AudioChunk;
    ///
    /// let chunk = AudioChunk::new(vec![0.1, 0.2], 0, 0.0, 16000);
    /// let buffer = AudioBuffer::from_chunk(chunk)?;
    /// assert_eq!(buffer.len(), 2);
    /// # Ok::<(), speech_prep::error::Error>(())
    /// ```
    pub fn from_chunk(chunk: AudioChunk) -> Result<Self> {
        let sample_rate = chunk.sample_rate;
        let samples = chunk.data;

        Self::from_samples(samples, sample_rate)
    }

    /// Get sample rate in Hz
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get number of samples
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get duration as `AudioDuration`
    ///
    /// # Example
    ///
    /// ```rust
    /// use speech_prep::buffer::AudioBuffer;
    /// use speech_prep::time::AudioDuration;
    ///
    /// let buffer = AudioBuffer::from_samples(vec![0.0; 16000], 16000)?;
    /// let duration = buffer.duration();
    /// assert_eq!(duration.as_secs(), 1);
    /// # Ok::<(), speech_prep::error::Error>(())
    /// ```
    #[must_use]
    pub fn duration(&self) -> AudioDuration {
        let duration_secs = self.samples.len() as f64 / f64::from(self.sample_rate);
        let duration_nanos = (duration_secs * 1_000_000_000.0) as u64;
        AudioDuration::from_nanos(duration_nanos)
    }

    /// Get immutable slice of samples
    #[must_use]
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Get mutable slice of samples
    pub fn samples_mut(&mut self) -> &mut [f32] {
        &mut self.samples
    }

    /// Consume buffer and return samples
    #[must_use]
    pub fn into_samples(self) -> Vec<f32> {
        self.samples
    }

    /// Normalize samples to [-1.0, 1.0] range
    ///
    /// Applies peak normalization to ensure samples are within valid range.
    ///
    /// # Example
    ///
    /// ```rust
    /// use speech_prep::buffer::AudioBuffer;
    ///
    /// let mut buffer = AudioBuffer::from_samples(vec![2.0, -2.0, 1.0], 16000)?;
    /// buffer.normalize();
    ///
    /// // Samples now scaled to [-1.0, 1.0]
    /// assert!(buffer.samples().iter().all(|&s| s >= -1.0 && s <= 1.0));
    /// # Ok::<(), speech_prep::error::Error>(())
    /// ```
    pub fn normalize(&mut self) {
        let max_abs = self.samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);

        if max_abs > 0.0 {
            let scale = 1.0 / max_abs;
            for sample in &mut self.samples {
                *sample *= scale;
            }
        }

        if let Some(ref mut meta) = self.metadata {
            meta.normalized = true;
            meta.processing_chain.push("normalize".to_owned());
        }
    }

    /// Validate sample values are within expected range
    ///
    /// Checks that all samples are finite and within reasonable bounds.
    pub fn validate_samples(&self) -> Result<()> {
        Self::validate_sample_values(&self.samples)
    }

    fn validate_sample_values(samples: &[f32]) -> Result<()> {
        for &sample in samples {
            if !sample.is_finite() {
                return Err(Error::invalid_format("sample value is not finite"));
            }
        }

        Ok(())
    }

    /// Get metadata
    #[must_use]
    pub fn metadata(&self) -> Option<&AudioBufferMetadata> {
        self.metadata.as_ref()
    }

    /// Set metadata
    pub fn set_metadata(&mut self, metadata: AudioBufferMetadata) {
        self.metadata = Some(metadata);
    }

    fn validate_sample_rate(sample_rate: u32) -> Result<()> {
        if VALID_SAMPLE_RATES.contains(&sample_rate) {
            Ok(())
        } else {
            Err(Error::invalid_format(format!(
                "unsupported sample rate: {sample_rate}"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_samples_valid() {
        let samples = vec![0.1, 0.2, -0.1, -0.2];
        let buffer = AudioBuffer::from_samples(samples.clone(), 16000);

        assert!(buffer.is_ok());
        let buffer = buffer.expect("buffer creation");
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.sample_rate(), 16000);
        assert_eq!(buffer.samples(), &samples[..]);
    }

    #[test]
    fn test_from_samples_rejects_non_finite_values() {
        let samples = vec![0.0, f32::NAN, 0.2];
        let buffer = AudioBuffer::from_samples(samples, 16000);

        assert!(buffer.is_err(), "NaN samples must be rejected");
    }

    #[test]
    fn test_sample_rate_policy() {
        assert!(AudioBuffer::from_samples(vec![0.0; 10], 16000).is_ok());
        assert!(
            AudioBuffer::from_samples(vec![0.0; 10], 32_000).is_err(),
            "Unexpected sample rates must be rejected"
        );
    }

    #[test]
    fn test_from_samples_empty_fails() {
        let samples: Vec<f32> = vec![];
        let result = AudioBuffer::from_samples(samples, 16000);

        assert!(result.is_err());
    }

    #[test]
    fn test_from_samples_invalid_sample_rate() {
        let samples = vec![0.1, 0.2];

        assert!(AudioBuffer::from_samples(samples.clone(), 1000).is_err());

        assert!(AudioBuffer::from_samples(samples, 100_000).is_err());
    }

    #[test]
    fn test_from_chunk() {
        let chunk = AudioChunk::new(vec![0.1, 0.2, -0.1], 0, 0.0, 16000);
        let buffer = AudioBuffer::from_chunk(chunk);

        assert!(buffer.is_ok());
        let buffer = buffer.expect("buffer from chunk");
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.sample_rate(), 16000);
    }

    #[test]
    fn test_duration() {
        let buffer = AudioBuffer::from_samples(vec![0.0; 16000], 16000).expect("buffer creation");
        let duration = buffer.duration();

        assert_eq!(duration.as_secs(), 1);
        assert!(duration.as_millis() >= 1000 && duration.as_millis() <= 1001); // Allow for rounding
    }

    #[test]
    fn test_normalize() {
        let mut buffer =
            AudioBuffer::from_samples(vec![2.0, -2.0, 1.0, -1.0], 16000).expect("buffer creation");

        buffer.normalize();

        let max_abs = buffer
            .samples()
            .iter()
            .map(|&s| s.abs())
            .fold(0.0f32, f32::max);
        assert!((max_abs - 1.0).abs() < 1e-6);

        assert!(buffer.metadata().expect("metadata").normalized);
    }

    #[test]
    fn test_normalize_zero_samples() {
        let mut buffer =
            AudioBuffer::from_samples(vec![0.0, 0.0, 0.0], 16000).expect("buffer creation");

        buffer.normalize();
    }

    #[test]
    fn test_validate_samples_valid() {
        let buffer =
            AudioBuffer::from_samples(vec![0.1, -0.5, 0.9], 16000).expect("buffer creation");

        assert!(buffer.validate_samples().is_ok());
    }

    #[test]
    fn test_validate_samples_infinite() {
        let mut buffer =
            AudioBuffer::from_samples(vec![0.1, 0.2, 0.9], 16000).expect("buffer creation");
        buffer.samples_mut()[1] = f32::INFINITY;
        assert!(buffer.validate_samples().is_err());
    }

    #[test]
    fn test_validate_samples_nan() {
        let mut buffer =
            AudioBuffer::from_samples(vec![0.1, 0.2, 0.9], 16000).expect("buffer creation");
        buffer.samples_mut()[1] = f32::NAN;
        assert!(buffer.validate_samples().is_err());
    }

    #[test]
    fn test_into_samples() {
        let samples = vec![0.1, 0.2, 0.3];
        let buffer = AudioBuffer::from_samples(samples.clone(), 16000).expect("buffer creation");

        let extracted = buffer.into_samples();
        assert_eq!(extracted, samples);
    }

    #[test]
    fn test_samples_mut() {
        let mut buffer =
            AudioBuffer::from_samples(vec![0.1, 0.2, 0.3], 16000).expect("buffer creation");

        buffer.samples_mut()[0] = 0.5;
        assert_eq!(buffer.samples()[0], 0.5);
    }

    #[test]
    fn test_metadata_operations() {
        let mut buffer = AudioBuffer::from_samples(vec![0.1, 0.2], 16000).expect("buffer creation");

        let metadata = AudioBufferMetadata {
            source: Some("test.wav".to_owned()),
            original_sr: Some(44100),
            duration: Some(AudioDuration::from_millis(125)),
            normalized: true,
            processing_chain: vec!["resample".to_owned(), "normalize".to_owned()],
        };

        buffer.set_metadata(metadata);

        let retrieved = buffer.metadata().expect("metadata");
        assert_eq!(retrieved.source.as_deref(), Some("test.wav"));
        assert_eq!(retrieved.original_sr, Some(44100));
        assert!(retrieved.normalized);
        assert_eq!(retrieved.processing_chain.len(), 2);
    }
}
