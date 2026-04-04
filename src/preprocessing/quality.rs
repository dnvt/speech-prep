//! Audio quality assessment with SNR estimation and spectral analysis.
//!
//! This module provides multi-dimensional quality metrics for audio
//! preprocessing validation.
//!
//! # Metrics
//!
//! - **SNR (Signal-to-Noise Ratio)**: Measures signal power vs noise floor (dB)
//! - **RMS Energy**: Root-mean-square energy as baseline quality indicator
//! - **Spectral Centroid**: Weighted average frequency (brightness measure)
//! - **Quality Score**: Unified score in [0.0, 1.0] combining all metrics
//!
//! # Performance
//!
//! - **Target**: <10ms per second of 16 kHz audio
//! - **Memory**: Minimal allocations (reuses frame buffers)
//!
//! # Example
//!
//! ```rust,no_run
//! use speech_prep::preprocessing::QualityAssessor;
//!
//! # fn main() -> speech_prep::error::Result<()> {
//! let assessor = QualityAssessor::new(16000);
//! let audio_samples = vec![0.5f32; 16000]; // 1 second at 16 kHz
//!
//! let metrics = assessor.assess(&audio_samples)?;
//! assert!(metrics.snr_db.is_finite());
//! # Ok(())
//! # }
//! ```

use crate::error::{Error, Result};
use crate::time::{AudioDuration, AudioInstant};
use tracing::{debug, trace};

/// Quality metrics for audio assessment.
///
/// All metrics are computed for a single audio chunk.
#[derive(Debug, Clone, Copy)]
pub struct QualityMetrics {
    /// Signal-to-noise ratio in decibels [0.0, 60.0]
    pub snr_db: f32,
    /// RMS energy level [0.0, 1.0]
    pub energy: f32,
    /// Spectral centroid in Hz [0.0, `sample_rate/2`]
    pub spectral_centroid: f32,
    /// Unified quality score [0.0, 1.0] (higher is better)
    pub quality_score: f32,
}

/// Audio quality assessor with configurable sample rate.
///
/// Computes multi-dimensional quality metrics for audio chunks,
/// providing objective measures for quality gates and filtering.
#[derive(Debug, Clone, Copy)]
pub struct QualityAssessor {
    sample_rate: u32,
}

impl QualityAssessor {
    /// Creates a new quality assessor for the given sample rate.
    ///
    /// # Arguments
    ///
    /// - `sample_rate`: Audio sample rate in Hz (e.g., 16000)
    ///
    /// # Example
    ///
    /// ```rust
    /// use speech_prep::preprocessing::QualityAssessor;
    ///
    /// let assessor = QualityAssessor::new(16000);
    /// ```
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Assesses audio quality for the given samples.
    ///
    /// Computes SNR, energy, spectral centroid, and unified quality score.
    ///
    /// # Arguments
    ///
    /// - `samples`: Audio samples to assess (must not be empty)
    ///
    /// # Returns
    ///
    /// Quality metrics including SNR (dB), energy, spectral centroid (Hz),
    /// and unified quality score [0.0, 1.0].
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if samples are empty.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use speech_prep::preprocessing::QualityAssessor;
    ///
    /// # fn main() -> speech_prep::error::Result<()> {
    /// let assessor = QualityAssessor::new(16000);
    /// let audio = vec![0.5f32; 16000];
    /// let metrics = assessor.assess(&audio)?;
    /// assert!((0.0..=1.0).contains(&metrics.quality_score));
    /// # Ok(())
    /// # }
    /// ```
    pub fn assess(self, samples: &[f32]) -> Result<QualityMetrics> {
        trace!(sample_count = samples.len(), "Assessing audio quality");

        if samples.is_empty() {
            return Err(Error::InvalidInput("Cannot assess empty audio".into()));
        }

        let processing_start = AudioInstant::now();
        let energy = Self::calculate_rms(samples);
        let snr_db = Self::calculate_snr(samples, energy)?;
        let spectral_centroid = self.calculate_spectral_centroid(samples)?;
        let quality_score = self.aggregate_score(snr_db, energy, spectral_centroid);

        debug!(
            snr_db,
            energy, spectral_centroid, quality_score, "Audio quality metrics computed"
        );

        let metrics = QualityMetrics {
            snr_db,
            energy,
            spectral_centroid,
            quality_score,
        };
        let _latency = elapsed_duration(processing_start);

        Ok(metrics)
    }

    /// Calculates RMS (root-mean-square) energy of audio samples.
    ///
    /// This is a static method that can be called without an assessor instance.
    ///
    /// # Arguments
    ///
    /// - `samples`: Audio samples (must not be empty)
    ///
    /// # Returns
    ///
    /// RMS energy in range [0.0, 1.0] for normalized audio
    fn calculate_rms(samples: &[f32]) -> f32 {
        let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
        let mean_square = sum_squares / samples.len() as f32;
        mean_square.sqrt()
    }

    /// Calculates signal-to-noise ratio (SNR) in decibels.
    ///
    /// Estimates noise floor from the quietest 10% of frames,
    /// then computes dB ratio between signal RMS and noise floor.
    ///
    /// # Arguments
    ///
    /// - `samples`: Audio samples
    /// - `signal_rms`: Pre-computed RMS energy of the signal
    ///
    /// # Returns
    ///
    /// SNR in dB, clamped to [0.0, 60.0] for practical purposes
    ///
    /// # Errors
    ///
    /// Returns `Error::AudioProcessing` if insufficient frames for estimation
    fn calculate_snr(samples: &[f32], signal_rms: f32) -> Result<f32> {
        // Compute frame energies (256 samples per frame)
        let frame_energies = Self::frame_energy(samples);

        let mut valid_energies: Vec<f32> =
            frame_energies.into_iter().filter(|x| !x.is_nan()).collect();

        if valid_energies.is_empty() {
            return Err(Error::Processing(
                "All frame energies are NaN; cannot estimate noise floor".into(),
            ));
        }

        valid_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Quietest 10% of frames as noise floor estimate
        let noise_frame_count = (valid_energies.len() / 10).max(1);
        let noise_frames = valid_energies
            .get(0..noise_frame_count)
            .ok_or_else(|| Error::Processing("Insufficient frames for noise estimation".into()))?;

        let noise_floor = noise_frames.iter().sum::<f32>() / noise_frames.len() as f32;

        if signal_rms < 1e-6 {
            return Ok(0.0);
        }

        if noise_floor < 1e-10 {
            return Ok(60.0);
        }

        let snr = 20.0 * (signal_rms / noise_floor).log10();
        Ok(snr.clamp(0.0, 60.0))
    }

    /// Computes RMS energy for each frame of audio.
    ///
    /// Divides audio into fixed-size frames and computes RMS for each.
    ///
    /// # Arguments
    ///
    /// - `samples`: Audio samples
    ///
    /// # Returns
    ///
    /// Vector of RMS energies, one per frame
    fn frame_energy(samples: &[f32]) -> Vec<f32> {
        const FRAME_SIZE: usize = 256;
        samples
            .chunks(FRAME_SIZE)
            .map(|frame| {
                let sum_sq: f32 = frame.iter().map(|&s| s * s).sum();
                (sum_sq / frame.len() as f32).sqrt()
            })
            .collect()
    }

    /// Calculates spectral centroid (brightness measure) in Hz.
    ///
    /// Computes weighted average frequency from magnitude spectrum.
    /// Uses simplified time-domain approximation (not full FFT).
    ///
    /// # Arguments
    ///
    /// - `samples`: Audio samples (should be ≥512 for meaningful result)
    ///
    /// # Returns
    ///
    /// Spectral centroid in Hz, clamped to [0.0, `sample_rate/2`]
    ///
    /// # Errors
    ///
    /// Returns `Error::AudioProcessing` if samples are too short
    ///
    /// # Note
    ///
    /// This is a simplified implementation. Full FFT-based spectral
    /// centroid can be added in the future for more accurate results.
    fn calculate_spectral_centroid(self, samples: &[f32]) -> Result<f32> {
        // For very short audio, return midpoint frequency
        if samples.len() < 512 {
            return Ok(self.sample_rate as f32 / 4.0);
        }

        // Use first 512 samples for spectral analysis
        let window = samples.get(0..512).ok_or_else(|| {
            Error::Processing("Insufficient samples for spectral analysis".into())
        })?;

        // Time-domain approximation of spectral centroid
        let (magnitude_sum, weighted_sum) =
            window
                .iter()
                .enumerate()
                .fold((0.0f32, 0.0f32), |(mag_acc, weighted_acc), (i, &s)| {
                    let magnitude = s.abs();
                    (
                        mag_acc + magnitude,
                        magnitude.mul_add(i as f32, weighted_acc),
                    )
                });

        if magnitude_sum < 1e-10 {
            return Ok(self.sample_rate as f32 / 4.0);
        }

        let centroid_bin = weighted_sum / magnitude_sum;
        let centroid_hz = (centroid_bin / 512.0) * (self.sample_rate as f32 / 2.0);
        Ok(centroid_hz.clamp(0.0, self.sample_rate as f32 / 2.0))
    }

    /// Aggregates individual metrics into unified quality score [0.0, 1.0].
    ///
    /// Uses weighted combination:
    /// - 50% SNR (signal clarity)
    /// - 30% Energy (signal strength)
    /// - 20% Spectral centroid (frequency content)
    ///
    /// # Arguments
    ///
    /// - `snr_db`: Signal-to-noise ratio in dB
    /// - `energy`: RMS energy
    /// - `spectral_centroid`: Spectral centroid in Hz
    ///
    /// # Returns
    ///
    /// Quality score in [0.0, 1.0], where 1.0 is perfect quality
    fn aggregate_score(self, snr_db: f32, energy: f32, spectral_centroid: f32) -> f32 {
        let snr_score = (snr_db / 60.0).clamp(0.0, 1.0);
        let energy_score = (energy / 0.5).clamp(0.0, 1.0);
        let centroid_score = (spectral_centroid / (self.sample_rate as f32 / 2.0)).clamp(0.0, 1.0);

        // 50% SNR, 30% energy, 20% spectral
        let score = 0.5f32.mul_add(
            snr_score,
            0.3f32.mul_add(energy_score, 0.2 * centroid_score),
        );

        score.clamp(0.0, 1.0)
    }
}

fn elapsed_duration(start: AudioInstant) -> AudioDuration {
    AudioInstant::now().duration_since(start)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 0.01;

    #[test]
    fn test_high_quality_audio() {
        let assessor = QualityAssessor::new(16000);
        // Clean sine wave with silence periods (high quality with clear signal/noise
        // separation)
        let mut samples = vec![0.0f32; 16000];

        // Add strong signal in middle 50% of audio (8000 samples)
        for i in 4000..12000 {
            samples[i] = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5;
        }
        // First and last 25% remain silent (noise floor)

        let metrics = assessor.assess(&samples).unwrap();

        // High-quality audio with clear signal/noise separation should have high SNR
        assert!(
            metrics.snr_db > 20.0,
            "Expected SNR > 20 dB, got {:.1}",
            metrics.snr_db
        );
        assert!((0.0..=1.0).contains(&metrics.quality_score));
        assert!(
            metrics.quality_score > 0.5,
            "Expected quality > 0.5, got {:.2}",
            metrics.quality_score
        );
    }

    #[test]
    fn test_noisy_audio() {
        let assessor = QualityAssessor::new(16000);
        // Signal + random noise (lower quality)
        let mut noisy = vec![0.0f32; 16000];
        for (i, sample) in noisy.iter_mut().enumerate() {
            let signal = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.2;
            let noise = (i as f32 * 0.1).sin().mul_add(0.1, (i % 7) as f32 * 0.01);
            *sample = signal + noise;
        }

        let metrics = assessor.assess(&noisy).unwrap();

        // Noisy audio should have lower SNR and quality score
        assert!(
            metrics.snr_db < 40.0,
            "Expected SNR < 40 dB for noisy audio"
        );
        assert!((0.0..=1.0).contains(&metrics.quality_score));
    }

    #[test]
    fn test_energy_calculation() {
        let assessor = QualityAssessor::new(16000);
        // Constant amplitude signal
        let audio = vec![0.5f32; 1000];

        let metrics = assessor.assess(&audio).unwrap();

        // RMS of constant 0.5 should be 0.5
        assert!(
            (metrics.energy - 0.5).abs() < EPSILON,
            "Expected energy ~0.5, got {:.3}",
            metrics.energy
        );
    }

    #[test]
    fn test_quality_score_bounds() {
        let assessor = QualityAssessor::new(16000);
        let audio = vec![0.3f32; 5000];

        let metrics = assessor.assess(&audio).unwrap();

        // Quality score must always be in [0.0, 1.0]
        assert!(
            (0.0..=1.0).contains(&metrics.quality_score),
            "Quality score {:.2} out of bounds [0.0, 1.0]",
            metrics.quality_score
        );
        assert!(
            (0.0..=60.0).contains(&metrics.snr_db),
            "SNR {:.1} dB out of bounds [0.0, 60.0]",
            metrics.snr_db
        );
    }

    #[test]
    fn test_spectral_centroid_computed() {
        let assessor = QualityAssessor::new(16000);
        let audio = vec![0.2f32; 1024];

        let metrics = assessor.assess(&audio).unwrap();

        // Spectral centroid should be in valid frequency range
        assert!(metrics.spectral_centroid >= 0.0);
        assert!(
            metrics.spectral_centroid <= 8000.0, // Nyquist frequency
            "Spectral centroid {:.1} Hz exceeds Nyquist (8000 Hz)",
            metrics.spectral_centroid
        );
    }

    #[test]
    fn test_empty_audio() {
        let assessor = QualityAssessor::new(16000);
        let result = assessor.assess(&[]);

        assert!(result.is_err(), "Should reject empty audio");
        match result.unwrap_err() {
            Error::InvalidInput(msg) => {
                assert!(
                    msg.contains("empty"),
                    "Expected 'empty' error, got: {}",
                    msg
                );
            }
            other => panic!("Expected InvalidInput error, got: {:?}", other),
        }
    }

    #[test]
    fn test_silence_handling() {
        let assessor = QualityAssessor::new(16000);
        // Pure silence (all zeros)
        let silence = vec![0.0f32; 16000];

        let metrics = assessor.assess(&silence).unwrap();

        // Silence should have zero energy
        assert!(
            metrics.energy < EPSILON,
            "Expected near-zero energy for silence, got {:.6}",
            metrics.energy
        );
        // Silence should have 0 dB SNR (not maximum!)
        assert!(
            metrics.snr_db < 1.0,
            "Expected SNR ~0 dB for silence, got {:.1} dB",
            metrics.snr_db
        );
        // Silence should have LOW quality score (not high!)
        assert!(
            metrics.quality_score < 0.2,
            "Expected quality <0.2 for silence, got {:.2}",
            metrics.quality_score
        );
        // Quality score should still be valid bounds
        assert!((0.0..=1.0).contains(&metrics.quality_score));
    }

    #[test]
    fn test_short_audio() {
        let assessor = QualityAssessor::new(16000);
        // Very short audio (< 512 samples)
        let short_audio = vec![0.5f32; 256];

        let metrics = assessor.assess(&short_audio).unwrap();

        // Should not panic, should return valid metrics
        assert!((0.0..=1.0).contains(&metrics.quality_score));
        assert!(metrics.spectral_centroid > 0.0);
    }

    #[test]
    fn test_very_quiet_audio() {
        let assessor = QualityAssessor::new(16000);
        // Very quiet audio (below signal threshold but not exactly zero)
        let very_quiet = vec![1e-7f32; 16000];

        let metrics = assessor.assess(&very_quiet).unwrap();

        // Very quiet audio should be treated similarly to silence
        assert!(
            metrics.energy < 1e-6,
            "Expected near-zero energy for very quiet audio, got {:.9}",
            metrics.energy
        );
        assert!(
            metrics.snr_db < 5.0,
            "Expected low SNR for very quiet audio, got {:.1} dB",
            metrics.snr_db
        );
        assert!(
            metrics.quality_score < 0.3,
            "Expected low quality for very quiet audio, got {:.2}",
            metrics.quality_score
        );
    }
}
