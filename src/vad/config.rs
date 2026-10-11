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
///     activation_ratio: 3.5,
///     hangover_frames: 30,
///     ..VadConfig::default()
/// };
/// assert!(noisy.validate().is_ok());
///
/// // Quiet speaker: trigger earlier.
/// let quiet = VadConfig {
///     activation_ratio: 1.8,
///     release_ratio: 1.2,
///     ..VadConfig::default()
/// };
/// assert!(quiet.validate().is_ok());
/// ```
///
/// # How detection works
///
/// Each frame's RMS energy is compared with a running estimate of the
/// background level, the noise floor. A frame is loud when its energy is at
/// least `activation_ratio` times the floor (`release_ratio` once a segment
/// is open) and above `min_speech_energy`. A segment opens on the first loud
/// frame and closes after more than `hangover_frames` quiet frames in a row.
///
/// The floor is seeded from the quietest frame of the first 300 ms. During
/// silence it follows the frame energy with `noise_smoothing`; during speech
/// it still falls quickly but rises only slowly, so steady noise that starts
/// with the stream is absorbed within a few seconds while words are not.
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

    /// Pre-emphasis coefficient for `y[n] = x[n] - a * x[n-1]`, or `None` to
    /// disable. Default: `Some(0.97)`. It tilts the spectrum towards the
    /// frequencies that carry speech and away from low-frequency hum.
    pub pre_emphasis: Option<f32>,

    /// How many times louder than the noise floor a frame must be to start a
    /// segment, `>= release_ratio`. Default: 2.5 (about 8 dB).
    ///
    /// Raise it (3-4) if noise triggers false detections; lower it (1.7-2)
    /// to catch soft speech.
    pub activation_ratio: f32,

    /// How many times louder than the noise floor a frame must be to count
    /// as speech inside an open segment, `>= 1.0`. Default: 1.3.
    ///
    /// Keeping it below `activation_ratio` lets a segment ride through the
    /// quieter ends of words without starting on every small bump.
    pub release_ratio: f32,

    /// Frames quieter than this RMS level are never speech. Default: 0.0005.
    ///
    /// Measured on the pre-emphasized, windowed frame, so it sits well below
    /// the raw signal level. Background noise is handled by the noise floor,
    /// not by this gate; raise it only to ignore faint sounds in otherwise
    /// silent recordings.
    pub min_speech_energy: f32,

    /// Smoothing of the noise floor during silence, in `[0.0, 1.0)`.
    /// Default: 0.85. Higher values adapt more slowly to a changing
    /// background.
    pub noise_smoothing: f32,

    /// Lower bound for the noise floor, so digital silence does not make
    /// every faint sound look loud. Must be positive. Default: 1e-4.
    pub energy_floor: f32,

    /// Quiet frames tolerated before a speech segment is closed.
    /// Default: 20 (200 ms with the default 10 ms hop), which keeps the
    /// short gaps between words inside one segment. Lower it to split speech
    /// at every brief pause.
    pub hangover_frames: usize,

    /// Extra audio included before and after each segment, so soft word
    /// onsets and endings are not cut off. Default: 50 ms. Padding never
    /// makes segments overlap.
    pub speech_padding: AudioDuration,

    /// Minimum number of speech frames for a segment to be reported.
    /// Default: 3. Shorter blips are dropped.
    pub min_speech_frames: usize,

    /// Timestamp assigned to the first sample the detector sees.
    /// Default: [`AudioTimestamp::EPOCH`].
    pub stream_start_time: AudioTimestamp,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            frame_duration: AudioDuration::from_millis(20),
            frame_overlap: 0.5,
            pre_emphasis: Some(0.97),
            activation_ratio: 2.5,
            release_ratio: 1.3,
            min_speech_energy: 0.0005,
            noise_smoothing: 0.85,
            energy_floor: 1e-4,
            hangover_frames: 20,
            speech_padding: AudioDuration::from_millis(50),
            min_speech_frames: 3,
            stream_start_time: AudioTimestamp::EPOCH,
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
        if self.sample_rate == 0 {
            return Err(invalid_input("sample_rate must be greater than zero"));
        }

        if self.frame_duration.as_nanos() == 0 {
            return Err(invalid_input("frame_duration must be non-zero"));
        }

        if !(0.0..1.0).contains(&self.frame_overlap) {
            return Err(invalid_input("frame_overlap must be within [0.0, 1.0)"));
        }

        if let Some(coeff) = self.pre_emphasis {
            if !(0.0..1.0).contains(&coeff) {
                return Err(invalid_input(
                    "pre_emphasis coefficient must be in [0.0, 1.0)",
                ));
            }
        }

        if self.release_ratio.is_nan() || self.release_ratio < 1.0 {
            return Err(invalid_input("release_ratio must be >= 1.0"));
        }

        if !self.activation_ratio.is_finite() || self.activation_ratio < self.release_ratio {
            return Err(invalid_input(
                "activation_ratio must be finite and >= release_ratio",
            ));
        }

        if self.min_speech_energy.is_nan() || self.min_speech_energy < 0.0 {
            return Err(invalid_input("min_speech_energy must be non-negative"));
        }

        if !(0.0..1.0).contains(&self.noise_smoothing) {
            return Err(invalid_input("noise_smoothing must be within [0.0, 1.0)"));
        }

        if !self.energy_floor.is_finite() || self.energy_floor <= 0.0 {
            return Err(invalid_input("energy_floor must be positive and finite"));
        }

        if self.min_speech_frames == 0 {
            return Err(invalid_input("min_speech_frames must be greater than zero"));
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
}

fn invalid_input(message: impl Into<String>) -> Error {
    Error::Configuration(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_valid() {
        assert!(VadConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_bad_ratios() {
        for (activation_ratio, release_ratio) in [
            (2.0, 0.9),
            (1.2, 1.5),
            (f32::NAN, 1.3),
            (2.0, f32::NAN),
            (f32::INFINITY, 1.3),
        ] {
            let config = VadConfig {
                activation_ratio,
                release_ratio,
                ..VadConfig::default()
            };
            assert!(
                config.validate().is_err(),
                "{activation_ratio} / {release_ratio}"
            );
        }
    }

    #[test]
    fn rejects_bad_noise_settings() {
        for config in [
            VadConfig {
                noise_smoothing: 1.0,
                ..VadConfig::default()
            },
            VadConfig {
                energy_floor: 0.0,
                ..VadConfig::default()
            },
            VadConfig {
                energy_floor: f32::NAN,
                ..VadConfig::default()
            },
            VadConfig {
                min_speech_energy: -1.0,
                ..VadConfig::default()
            },
        ] {
            assert!(config.validate().is_err(), "{config:?}");
        }
    }
}
