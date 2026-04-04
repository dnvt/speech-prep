/// Decoded audio data with normalized samples and metadata.
///
/// All samples are normalized to the range [-1.0, 1.0] regardless of the
/// source bit depth. This normalization enables consistent processing across
/// different audio formats and bit depths.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedAudio {
    /// Normalized audio samples in the range [-1.0, 1.0].
    /// For stereo audio, samples are interleaved (L, R, L, R, ...).
    pub samples: Vec<f32>,
    /// Sample rate in Hz (e.g., 44100, 48000).
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo).
    pub channels: u8,
    /// Bit depth of the source PCM data (e.g., 16, 24).
    pub bit_depth: u16,
    /// Total duration in seconds, calculated from sample count and rate.
    pub duration_sec: f64,
}

impl DecodedAudio {
    /// Total number of audio frames (samples per channel).
    ///
    /// For stereo audio with 1000 total samples, this returns 500 frames.
    #[cfg(test)]
    #[must_use]
    pub fn frame_count(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }

    /// Verify all samples are within normalized bounds [-1.0, 1.0].
    ///
    /// Returns `true` if all samples are properly normalized.
    #[cfg(test)]
    #[must_use]
    pub fn is_normalized(&self) -> bool {
        self.samples.iter().all(|&s| (-1.0..=1.0).contains(&s))
    }
}

/// Mono audio with metadata from channel mixing.
///
/// Contains the mixed mono samples plus diagnostic information about the
/// original channel layout and peak amplitudes before/after mixing.
#[derive(Debug, Clone, PartialEq)]
pub struct MixedAudio {
    /// Mono audio samples in the range [-1.0, 1.0].
    pub samples: Vec<f32>,
    /// Original number of channels before mixing (1 = already mono).
    pub original_channels: u8,
    /// Peak amplitude in the original multi-channel audio.
    pub peak_before_mix: f32,
    /// Peak amplitude after mixing to mono.
    pub peak_after_mix: f32,
}

impl MixedAudio {
    /// Total number of mono samples.
    #[cfg(test)]
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Check if any clipping occurred during mixing.
    ///
    /// Returns `true` if the peak amplitude equals 1.0 (indicating potential
    /// clipping at the boundaries).
    #[cfg(test)]
    #[must_use]
    pub fn is_clipped(&self) -> bool {
        (self.peak_after_mix - 1.0).abs() < f32::EPSILON
    }

    /// Calculate the peak reduction ratio from mixing.
    ///
    /// Returns the ratio of post-mix peak to pre-mix peak. A value of 1.0
    /// means no amplitude change, <1.0 means reduction, >1.0 means
    /// amplification (rare with averaging).
    #[cfg(test)]
    #[must_use]
    pub fn peak_ratio(&self) -> f32 {
        if self.peak_before_mix.abs() < f32::EPSILON {
            1.0 // Avoid division by zero for silent input
        } else {
            self.peak_after_mix / self.peak_before_mix
        }
    }
}
