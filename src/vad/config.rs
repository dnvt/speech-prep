//! VAD configuration and validation.

use crate::error::{Error, Result};
use crate::time::{AudioDuration, AudioTimestamp};

const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// Configuration for [`VadDetector`](super::VadDetector).
///
/// The defaults target 16 kHz mono speech with 20 ms frames and 50% overlap.
/// Most callers only need to adjust sensitivity:
///
/// ```
/// use speech_prep::VadConfig;
///
/// // Noisy room: require a stronger signal and tolerate longer pauses.
/// let noisy = VadConfig {
///     activation_margin: 1.4,
///     hangover_frames: 5,
///     ..VadConfig::default()
/// };
/// assert!(noisy.validate().is_ok());
///
/// // Quiet speaker: trigger earlier.
/// let quiet = VadConfig {
///     activation_margin: 1.05,
///     min_speech_frames: 2,
///     ..VadConfig::default()
/// };
/// assert!(quiet.validate().is_ok());
/// ```
///
/// # How detection works
///
/// Each frame gets an RMS energy and a spectral-flux value. Both are divided
/// by running baselines (updated only during silence), combined with
/// `energy_weight` and `flux_weight`, and compared with an adaptive
/// threshold. `activation_margin` and `release_margin` add hysteresis, and
/// `hangover_frames` keeps a segment open through short pauses.
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    /// Sample rate of the input in Hz. Default: 16000.
    pub sample_rate: u32,

    /// Analysis frame length. Default: 20 ms.
    ///
    /// Shorter frames react faster; longer frames are more stable in noise.
    pub frame_duration: AudioDuration,

    /// Overlap between consecutive frames, in `[0.0, 1.0)`. Default: 0.5.
    pub frame_overlap: f32,

    /// Smoothing of the energy baseline, in `[0.0, 1.0)`. Default: 0.85.
    ///
    /// Higher values adapt more slowly to changes in background level.
    pub energy_smoothing: f32,

    /// Smoothing of the spectral-flux baseline, in `[0.0, 1.0)`.
    /// Default: 0.8.
    pub flux_smoothing: f32,

    /// Lower bound for the energy baseline, to avoid dividing by zero on
    /// digital silence. Default: 1e-4.
    pub energy_floor: f32,

    /// Lower bound for the spectral-flux baseline. Default: 1e-4.
    pub flux_floor: f32,

    /// Frames quieter than this RMS level are never speech. Default: 0.0005.
    ///
    /// Measured on the pre-emphasized, windowed frame, so it sits well below
    /// the raw signal level. Background noise is handled by the adaptive
    /// baselines, not by this gate; raise it only to ignore faint sounds in
    /// otherwise silent recordings.
    pub min_speech_energy: f32,

    /// Smoothing of the adaptive threshold, in `[0.0, 1.0)`. Default: 0.9.
    pub threshold_smoothing: f32,

    /// Threshold multiplier for entering speech, `>= 1.0`. Default: 1.1.
    ///
    /// Raise it (1.2-1.5) if noise triggers false detections; lower it
    /// (1.05) to catch soft speech.
    pub activation_margin: f32,

    /// Threshold multiplier for staying in speech, `<= activation_margin`.
    /// Default: 0.9.
    pub release_margin: f32,

    /// Starting threshold before the adaptive threshold has settled.
    /// Default: 0.4.
    pub base_threshold: f32,

    /// Weight of normalized energy in the combined score. Default: 0.6.
    ///
    /// The two weights are normalized to sum to 1.
    pub energy_weight: f32,

    /// Weight of normalized spectral flux in the combined score.
    /// Default: 0.4.
    pub flux_weight: f32,

    /// Silent frames tolerated before a speech segment is closed.
    /// Default: 3 (60 ms with the default frame settings).
    pub hangover_frames: usize,

    /// Minimum number of speech frames for a segment to be reported.
    /// Default: 3. Shorter blips are dropped.
    pub min_speech_frames: usize,

    /// Timestamp assigned to the first sample the detector sees.
    /// Default: [`AudioTimestamp::EPOCH`].
    pub stream_start_time: AudioTimestamp,

    /// Pre-emphasis coefficient for `y[n] = x[n] - a * x[n-1]`, or `None` to
    /// disable. Default: `Some(0.97)`.
    pub pre_emphasis: Option<f32>,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            frame_duration: AudioDuration::from_millis(20),
            frame_overlap: 0.5,
            energy_smoothing: 0.85,
            flux_smoothing: 0.8,
            energy_floor: 1e-4,
            flux_floor: 1e-4,
            min_speech_energy: 0.0005,
            threshold_smoothing: 0.9,
            activation_margin: 1.1,
            release_margin: 0.9,
            base_threshold: 0.4,
            energy_weight: 0.6,
            flux_weight: 0.4,
            hangover_frames: 3,
            min_speech_frames: 3,
            stream_start_time: AudioTimestamp::EPOCH,
            pre_emphasis: Some(0.97),
        }
    }
}

impl VadConfig {
    /// Check that the configuration is usable.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] describing the first invalid field.
    pub fn validate(&self) -> Result<()> {
        const EPSILON: f32 = 1e-6;

        if self.sample_rate == 0 {
            return Err(invalid_input("sample_rate must be greater than zero"));
        }

        if self.frame_duration.as_nanos() as u64 == 0 {
            return Err(invalid_input("frame_duration must be non-zero"));
        }

        if !(0.0..1.0).contains(&self.frame_overlap) {
            return Err(invalid_input("frame_overlap must be within [0.0, 1.0)"));
        }

        if !(0.0..1.0).contains(&self.energy_smoothing) {
            return Err(invalid_input("energy_smoothing must be within [0.0, 1.0)"));
        }

        if !(0.0..1.0).contains(&self.flux_smoothing) {
            return Err(invalid_input("flux_smoothing must be within [0.0, 1.0)"));
        }

        if !(0.0..1.0).contains(&self.threshold_smoothing) {
            return Err(invalid_input(
                "threshold_smoothing must be within [0.0, 1.0)",
            ));
        }

        if self.min_speech_energy.is_nan() || self.min_speech_energy < 0.0 {
            return Err(invalid_input("min_speech_energy must be non-negative"));
        }

        if self.activation_margin < 1.0 {
            return Err(invalid_input("activation_margin must be >= 1.0"));
        }

        if self.release_margin <= 0.0 {
            return Err(invalid_input("release_margin must be positive"));
        }

        if self.release_margin > self.activation_margin {
            return Err(invalid_input("release_margin must be <= activation_margin"));
        }

        if self.base_threshold <= 0.0 {
            return Err(invalid_input("base_threshold must be positive"));
        }

        if self.energy_weight < 0.0 || self.flux_weight < 0.0 {
            return Err(invalid_input("metric weights must be non-negative"));
        }

        let weight_sum = self.energy_weight + self.flux_weight;
        if weight_sum.abs() < EPSILON {
            return Err(invalid_input("metric weights must not both be zero"));
        }

        if self.min_speech_frames == 0 {
            return Err(invalid_input("min_speech_frames must be greater than zero"));
        }

        if let Some(coeff) = self.pre_emphasis {
            if !(0.0..1.0).contains(&coeff) {
                return Err(invalid_input(
                    "pre_emphasis coefficient must be in [0.0, 1.0)",
                ));
            }
        }

        Ok(())
    }

    /// Frame length in samples (at least 1).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] if the length does not fit in `usize`.
    pub fn frame_length_samples(&self) -> Result<usize> {
        let sr = u128::from(self.sample_rate);
        let nanos = self.frame_duration.as_nanos();
        let numerator = nanos
            .saturating_mul(sr)
            .saturating_add(NANOS_PER_SECOND / 2);
        let samples = usize::try_from(numerator / NANOS_PER_SECOND)
            .map_err(|_| invalid_input("frame duration too large for platform"))?;
        Ok(samples.max(1))
    }

    /// Hop between frames in samples (at least 1).
    ///
    /// # Errors
    ///
    /// Same as [`frame_length_samples`](Self::frame_length_samples).
    pub fn hop_length_samples(&self) -> Result<usize> {
        let frame_length = self.frame_length_samples()?;
        let hop = (frame_length as f32 * (1.0 - self.frame_overlap)).round() as usize;
        Ok(hop.max(1))
    }

    /// FFT size: the next power of two at or above the frame length.
    ///
    /// # Errors
    ///
    /// Same as [`frame_length_samples`](Self::frame_length_samples).
    pub fn fft_size(&self) -> Result<usize> {
        Ok(self.frame_length_samples()?.next_power_of_two())
    }
}

fn invalid_input(message: impl Into<String>) -> Error {
    Error::Configuration(message.into())
}
