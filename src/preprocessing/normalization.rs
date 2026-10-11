//! RMS loudness normalization with a gain cap and hard limiting.

use crate::error::{Error, Result};

/// Scales audio towards a target RMS level.
///
/// The gain is `target_rms / rms(input)`, capped at `max_gain` so that quiet
/// noise is not boosted into something loud. Samples that would exceed
/// ±1.0 after gain are hard-clipped, so pick a target with headroom (0.1,
/// about -20 dBFS, is a reasonable choice for speech).
///
/// ```
/// use speech_prep::preprocessing::Normalizer;
///
/// # fn main() -> speech_prep::Result<()> {
/// let normalizer = Normalizer::new(0.1, 10.0)?;
/// let quiet = vec![0.02f32; 1000];
/// let louder = normalizer.normalize(&quiet)?;
/// assert!((louder[0] - 0.1).abs() < 1e-6);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Normalizer {
    target_rms: f32,
    max_gain: f32,
}

impl Normalizer {
    /// Create a normalizer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] if `target_rms` is outside
    /// `[0.0, 1.0]` or `max_gain` is not positive.
    pub fn new(target_rms: f32, max_gain: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&target_rms) {
            return Err(Error::Configuration(
                "target_rms must be in [0.0, 1.0]".into(),
            ));
        }
        if max_gain.is_nan() || max_gain <= 0.0 {
            return Err(Error::Configuration("max_gain must be positive".into()));
        }
        Ok(Self {
            target_rms,
            max_gain,
        })
    }

    /// Return a gain-adjusted copy of `samples`. Digital silence is returned
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if `samples` is empty.
    pub fn normalize(self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.is_empty() {
            return Err(Error::InvalidInput("cannot normalize empty audio".into()));
        }

        let current_rms = Self::calculate_rms(samples);
        if current_rms < 1e-10 {
            return Ok(samples.to_vec());
        }

        let gain = (self.target_rms / current_rms).min(self.max_gain);
        let mut clipped = 0usize;
        let output = samples
            .iter()
            .map(|&s| {
                let amplified = s * gain;
                if amplified.abs() > 1.0 {
                    clipped += 1;
                }
                amplified.clamp(-1.0, 1.0)
            })
            .collect();

        tracing::trace!(current_rms, gain, clipped, "normalized audio");
        Ok(output)
    }

    fn calculate_rms(samples: &[f32]) -> f32 {
        let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }
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
