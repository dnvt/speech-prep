//! Audio normalization with RMS targeting and peak limiting.
//!
//! This module provides intelligent audio level normalization that:
//! - Targets a configurable RMS level (e.g., -20 dBFS)
//! - Prevents clipping with hard peak limiting at ±1.0
//! - Enforces maximum gain ceiling to avoid over-amplification
//! - Handles silence and very quiet audio gracefully
//!
//! # Example
//!
//! ```rust,no_run
//! use speech_prep::preprocessing::Normalizer;
//!
//! # fn main() -> speech_prep::error::Result<()> {
//! let normalizer = Normalizer::new(0.5, 10.0)?;
//! let quiet_audio = vec![0.1f32; 1000];
//! let normalized = normalizer.normalize(&quiet_audio)?;
//! # Ok(())
//! # }
//! ```

use crate::error::{Error, Result};
use crate::time::{AudioDuration, AudioInstant};
use tracing::{debug, trace};

/// Audio normalizer with RMS targeting and peak limiting.
///
/// # Configuration
///
/// - `target_rms`: Desired RMS level in range [0.0, 1.0] (e.g., 0.5 for -6
///   dBFS)
/// - `max_gain`: Maximum gain multiplier to prevent over-amplification (e.g.,
///   10.0)
///
/// # Safety
///
/// All output samples are hard-clamped to [-1.0, 1.0] to prevent clipping.
#[derive(Debug, Clone, Copy)]
pub struct Normalizer {
    target_rms: f32,
    max_gain: f32,
}

impl Normalizer {
    /// Creates a new normalizer with the given target RMS and max gain.
    ///
    /// # Arguments
    ///
    /// - `target_rms`: Target RMS level in [0.0, 1.0]
    /// - `max_gain`: Maximum gain multiplier (must be positive)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if:
    /// - `target_rms` is not in [0.0, 1.0]
    /// - `max_gain` is not positive
    pub fn new(target_rms: f32, max_gain: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&target_rms) {
            return Err(Error::InvalidInput(
                "target_rms must be in range [0.0, 1.0]".into(),
            ));
        }
        if max_gain <= 0.0 {
            return Err(Error::InvalidInput("max_gain must be positive".into()));
        }
        Ok(Self {
            target_rms,
            max_gain,
        })
    }

    /// Normalizes audio samples to the target RMS level.
    ///
    /// # Arguments
    ///
    /// - `samples`: Input audio samples
    ///
    /// # Returns
    ///
    /// Normalized audio with target RMS level, all samples in [-1.0, 1.0].
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if samples are empty.
    #[tracing::instrument(skip(samples), fields(sample_count = samples.len()))]
    pub fn normalize(self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.is_empty() {
            return Err(Error::InvalidInput("cannot normalize empty audio".into()));
        }

        let processing_start = AudioInstant::now();
        let current_rms = Self::calculate_rms(samples);

        if current_rms < 1e-10 {
            trace!(
                current_rms,
                "audio is silence or near-silence, no normalization applied"
            );
            let _elapsed = elapsed_duration(processing_start);
            return Ok(samples.to_vec());
        }

        let raw_gain = self.target_rms / current_rms;
        let gain = raw_gain.min(self.max_gain);
        let gain_limited = raw_gain > self.max_gain;

        let (output, clipped_samples) = Self::apply_gain_with_limiting(samples, gain);

        Self::log_normalization_metrics(
            current_rms,
            self.target_rms,
            gain,
            gain_limited,
            clipped_samples,
        );

        let _elapsed = elapsed_duration(processing_start);

        Ok(output)
    }

    /// Applies gain to samples with peak limiting at [-1.0, 1.0].
    ///
    /// Returns the normalized samples and count of clipped samples.
    fn apply_gain_with_limiting(samples: &[f32], gain: f32) -> (Vec<f32>, usize) {
        let mut clipped_samples = 0usize;
        let output: Vec<f32> = samples
            .iter()
            .map(|&s| {
                let amplified = s * gain;
                if amplified.abs() > 1.0 {
                    clipped_samples += 1;
                }
                amplified.clamp(-1.0, 1.0)
            })
            .collect();
        (output, clipped_samples)
    }

    fn log_normalization_metrics(
        current_rms: f32,
        target_rms: f32,
        gain: f32,
        gain_limited: bool,
        clipped_samples: usize,
    ) {
        let gain_db = 20.0 * gain.log10();

        if gain_db > 6.0 {
            debug!(
                current_rms,
                target_rms,
                gain,
                gain_db,
                gain_limited,
                clipped_samples,
                "high gain applied during normalization"
            );
        } else {
            trace!(
                current_rms,
                target_rms,
                gain,
                gain_db,
                gain_limited,
                clipped_samples,
                "normalization complete"
            );
        }
    }

    /// Calculates the root-mean-square (RMS) of the input samples.
    fn calculate_rms(samples: &[f32]) -> f32 {
        let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
        let mean_square = sum_squares / samples.len() as f32;
        mean_square.sqrt()
    }
}

fn elapsed_duration(start: AudioInstant) -> AudioDuration {
    AudioInstant::now().duration_since(start)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_to_target_rms() {
        let normalizer = Normalizer::new(0.5, 10.0).unwrap();
        let quiet_audio = vec![0.1f32; 1000];

        let normalized = normalizer.normalize(&quiet_audio).unwrap();
        let result_rms = Normalizer::calculate_rms(&normalized);

        // Within 5% tolerance
        assert!(
            (result_rms - 0.5).abs() < 0.025,
            "RMS {result_rms} not within tolerance of 0.5"
        );
    }

    #[test]
    fn test_no_clipping() {
        let normalizer = Normalizer::new(0.9, 100.0).unwrap();
        let audio = vec![0.8f32; 1000];

        let normalized = normalizer.normalize(&audio).unwrap();

        for &sample in &normalized {
            assert!(
                (-1.0..=1.0).contains(&sample),
                "Sample {sample} outside [-1.0, 1.0]"
            );
        }
    }

    #[test]
    fn test_max_gain_limit() {
        let normalizer = Normalizer::new(0.5, 2.0).unwrap();
        let very_quiet = vec![0.01f32; 1000];

        let normalized = normalizer.normalize(&very_quiet).unwrap();

        // Verify gain didn't exceed max_gain
        let actual_gain = normalized[0] / very_quiet[0];
        assert!(
            actual_gain <= 2.0 + 1e-6,
            "Gain {actual_gain} exceeded max_gain 2.0"
        );
    }

    #[test]
    fn test_silence_handling() {
        let normalizer = Normalizer::new(0.5, 10.0).unwrap();
        let silence = vec![0.0f32; 1000];

        let normalized = normalizer.normalize(&silence).unwrap();

        assert_eq!(normalized, silence, "Silence should remain unchanged");
    }

    #[test]
    fn test_near_silence_handling() {
        let normalizer = Normalizer::new(0.5, 10.0).unwrap();
        let near_silence = vec![1e-11f32; 1000];

        let normalized = normalizer.normalize(&near_silence).unwrap();

        // Near-silence should be preserved (no gain applied)
        assert_eq!(
            normalized, near_silence,
            "Near-silence should remain unchanged"
        );
    }

    #[test]
    fn test_invalid_target_rms_above() {
        let result = Normalizer::new(1.5, 10.0);
        assert!(result.is_err(), "Should reject target_rms > 1.0");
    }

    #[test]
    fn test_invalid_target_rms_below() {
        let result = Normalizer::new(-0.1, 10.0);
        assert!(result.is_err(), "Should reject negative target_rms");
    }

    #[test]
    fn test_invalid_max_gain_zero() {
        let result = Normalizer::new(0.5, 0.0);
        assert!(result.is_err(), "Should reject zero max_gain");
    }

    #[test]
    fn test_invalid_max_gain_negative() {
        let result = Normalizer::new(0.5, -1.0);
        assert!(result.is_err(), "Should reject negative max_gain");
    }

    #[test]
    fn test_empty_audio() {
        let normalizer = Normalizer::new(0.5, 10.0).unwrap();
        let result = normalizer.normalize(&[]);
        assert!(result.is_err(), "Should reject empty audio");
    }

    #[test]
    fn test_loud_audio_reduction() {
        let normalizer = Normalizer::new(0.3, 10.0).unwrap();
        let loud_audio = vec![0.9f32; 1000];

        let normalized = normalizer.normalize(&loud_audio).unwrap();
        let result_rms = Normalizer::calculate_rms(&normalized);

        // Should reduce loud audio to target
        assert!(
            (result_rms - 0.3).abs() < 0.02,
            "RMS {result_rms} not within tolerance of 0.3"
        );
    }

    #[test]
    fn test_varied_amplitude() {
        let normalizer = Normalizer::new(0.5, 10.0).unwrap();
        let varied: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0) * 0.1).collect();

        let normalized = normalizer.normalize(&varied).unwrap();

        // Check all samples are valid
        for &sample in &normalized {
            assert!(
                (-1.0..=1.0).contains(&sample),
                "Sample {sample} outside valid range"
            );
        }

        let result_rms = Normalizer::calculate_rms(&normalized);
        assert!(
            (result_rms - 0.5).abs() < 0.05,
            "RMS {result_rms} not within tolerance of 0.5"
        );
    }

    #[test]
    fn test_peak_limiting_preserves_bounds() {
        let normalizer = Normalizer::new(0.8, 20.0).unwrap();

        // Mostly quiet samples with a single full-scale peak that would clip after
        // gain.
        let mut audio = vec![0.1f32; 999];
        audio.insert(0, 1.0);

        let normalized = normalizer.normalize(&audio).unwrap();
        let result_rms = Normalizer::calculate_rms(&normalized);

        // The peak should be clamped to the hard limit.
        assert!(
            normalized
                .iter()
                .all(|sample| (-1.0..=1.0).contains(sample)),
            "Samples exceeded normalized bounds: {:?}",
            normalized
        );
        assert!(
            normalized[0] <= 1.0 && normalized[0] >= 0.999,
            "Peak sample should be hard-limited to ~1.0, got {}",
            normalized[0]
        );

        // RMS cannot quite hit the target because of limiting, but should still be
        // elevated.
        assert!(
            result_rms > 0.7 && result_rms <= normalizer.target_rms + 0.05,
            "RMS {result_rms} outside expected post-limiting range"
        );
    }
}
