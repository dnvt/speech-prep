//! Simple signal-quality metrics for a block of audio.
//!
//! ```
//! use speech_prep::preprocessing::QualityAssessor;
//!
//! # fn main() -> speech_prep::Result<()> {
//! let audio: Vec<f32> = (0..16_000)
//!     .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / 16_000.0).sin())
//!     .collect();
//!
//! let metrics = QualityAssessor::new(16_000).assess(&audio)?;
//! assert!((metrics.spectral_centroid - 1_000.0).abs() < 50.0);
//! # Ok(())
//! # }
//! ```

use realfft::RealFftPlanner;

use crate::error::{Error, Result};

/// Metrics returned by [`QualityAssessor::assess`].
#[derive(Debug, Clone, Copy)]
pub struct QualityMetrics {
    /// Overall RMS divided by the RMS of the quietest 10% of 16 ms frames,
    /// in dB, clamped to `[0, 60]`. A rough SNR that assumes the input
    /// contains some pauses.
    pub snr_db: f32,
    /// RMS level of the whole block.
    pub energy: f32,
    /// Power-weighted mean frequency in Hz. Speech is typically 500-2500 Hz;
    /// much higher values suggest hiss.
    pub spectral_centroid: f32,
    /// Heuristic score in `[0, 1]`: 60% SNR (0-60 dB) and 40% level
    /// (-50 to -20 dBFS RMS). Useful for ranking or gating clips, not as an
    /// absolute measure.
    pub quality_score: f32,
}

/// Computes [`QualityMetrics`].
#[derive(Debug, Clone, Copy)]
pub struct QualityAssessor {
    sample_rate: u32,
}

/// Frame length for the SNR and spectral estimates.
const FRAME: usize = 256;
const SPECTRUM_FRAME: usize = 512;

impl QualityAssessor {
    /// Create an assessor for audio at `sample_rate` Hz.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Measure `samples`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if `samples` is empty, and
    /// [`Error::Processing`] if the FFT fails.
    pub fn assess(self, samples: &[f32]) -> Result<QualityMetrics> {
        if samples.is_empty() {
            return Err(Error::InvalidInput("cannot assess empty audio".into()));
        }

        let energy = rms(samples);
        let snr_db = Self::estimate_snr(samples, energy);
        let spectral_centroid = self.spectral_centroid(samples)?;

        let snr_score = (snr_db / 60.0).clamp(0.0, 1.0);
        let level_db = 20.0 * energy.max(1e-10).log10();
        let level_score = ((level_db + 50.0) / 30.0).clamp(0.0, 1.0);
        let quality_score = 0.6f32.mul_add(snr_score, 0.4 * level_score);

        tracing::trace!(
            snr_db,
            energy,
            spectral_centroid,
            quality_score,
            "assessed audio quality"
        );

        Ok(QualityMetrics {
            snr_db,
            energy,
            spectral_centroid,
            quality_score,
        })
    }

    fn estimate_snr(samples: &[f32], signal_rms: f32) -> f32 {
        if signal_rms < 1e-6 {
            return 0.0;
        }

        let mut frames: Vec<f32> = samples.chunks(FRAME).map(rms).collect();
        frames.sort_by(f32::total_cmp);
        let quietest = &frames[..(frames.len() / 10).max(1)];
        let noise_floor = quietest.iter().sum::<f32>() / quietest.len() as f32;

        if noise_floor < 1e-10 {
            return 60.0;
        }
        (20.0 * (signal_rms / noise_floor).log10()).clamp(0.0, 60.0)
    }

    /// Average the power spectrum over Hann-windowed frames and return its
    /// centroid.
    fn spectral_centroid(self, samples: &[f32]) -> Result<f32> {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(SPECTRUM_FRAME);
        let mut input = fft.make_input_vec();
        let mut spectrum = fft.make_output_vec();
        let mut power = vec![0.0f32; spectrum.len()];

        let window: Vec<f32> = (0..SPECTRUM_FRAME)
            .map(|n| {
                let phase = 2.0 * std::f32::consts::PI * n as f32 / SPECTRUM_FRAME as f32;
                0.5f32.mul_add(-phase.cos(), 0.5)
            })
            .collect();

        for frame in samples.chunks(SPECTRUM_FRAME) {
            input.fill(0.0);
            for ((dst, &s), &w) in input.iter_mut().zip(frame).zip(&window) {
                *dst = s * w;
            }
            fft.process(&mut input, &mut spectrum)
                .map_err(|e| Error::Processing(format!("FFT failed: {e}")))?;
            for (p, bin) in power.iter_mut().zip(&spectrum) {
                *p += bin.norm_sqr();
            }
        }

        let total: f32 = power.iter().sum();
        if total < 1e-12 {
            return Ok(0.0);
        }

        let bin_hz = self.sample_rate as f32 / SPECTRUM_FRAME as f32;
        let weighted: f32 = power
            .iter()
            .enumerate()
            .map(|(k, &p)| k as f32 * bin_hz * p)
            .sum();
        Ok(weighted / total)
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
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
        for (i, sample) in samples.iter_mut().enumerate().take(12000).skip(4000) {
            *sample = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5;
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

        assert!((0.0..=1.0).contains(&metrics.quality_score));
        // A constant signal has all its energy near 0 Hz.
        assert!(metrics.spectral_centroid < 100.0);
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

    #[test]
    fn test_spectral_centroid_tracks_frequency() {
        let assessor = QualityAssessor::new(16000);
        for freq in [300.0f32, 1_000.0, 3_000.0] {
            let tone: Vec<f32> = (0..16000)
                .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / 16000.0).sin() * 0.5)
                .collect();
            let centroid = assessor.assess(&tone).unwrap().spectral_centroid;
            assert!(
                (centroid - freq).abs() < freq * 0.05,
                "{freq} Hz tone gave centroid {centroid:.1} Hz"
            );
        }
    }
}
