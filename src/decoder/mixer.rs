use crate::error::{Error, Result};

use super::MixedAudio;

/// Channel mixer for converting multi-channel audio to mono.
#[derive(Debug, Default, Clone, Copy)]
pub struct ChannelMixer;

impl ChannelMixer {
    /// Mix multi-channel audio into a mono buffer via averaging.
    pub fn mix_to_mono(samples: &[f32], channels: u8) -> Result<MixedAudio> {
        if channels == 0 {
            return Err(Error::InvalidInput("channel count cannot be zero".into()));
        }
        if ![1, 2, 4, 6].contains(&channels) {
            return Err(Error::InvalidInput(format!(
                "unsupported channel count: {channels} (supports 1, 2, 4, 6 only)"
            )));
        }
        if !samples.len().is_multiple_of(usize::from(channels)) {
            let sample_len = samples.len();
            return Err(Error::InvalidInput(format!(
                "sample count {sample_len} not divisible by channel count {channels}"
            )));
        }

        let peak_before = Self::calculate_peak(samples);

        if channels == 1 {
            return Ok(MixedAudio {
                samples: samples.to_vec(),
                original_channels: 1,
                peak_before_mix: peak_before,
                peak_after_mix: peak_before,
            });
        }

        let frame_count = samples.len() / usize::from(channels);
        let mut mixed = Vec::with_capacity(frame_count);

        for frame in samples.chunks_exact(usize::from(channels)) {
            let sum: f32 = frame.iter().sum();
            let avg = sum / f32::from(channels);
            mixed.push(avg.clamp(-1.0, 1.0));
        }

        let peak_after = Self::calculate_peak(&mixed);

        Ok(MixedAudio {
            samples: mixed,
            original_channels: channels,
            peak_before_mix: peak_before,
            peak_after_mix: peak_after,
        })
    }

    #[inline]
    fn calculate_peak(samples: &[f32]) -> f32 {
        samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult<T> = std::result::Result<T, String>;

    #[test]
    fn test_mix_identity_for_mono() -> TestResult<()> {
        let mono = vec![0.1, -0.2, 0.3];
        let mixed = ChannelMixer::mix_to_mono(&mono, 1).map_err(|e| e.to_string())?;
        assert_eq!(mixed.samples, mono);
        Ok(())
    }

    #[test]
    fn test_mix_stereo_to_mono() -> TestResult<()> {
        let stereo = vec![0.5, -0.5, 0.8, 0.2];
        let mixed = ChannelMixer::mix_to_mono(&stereo, 2).map_err(|e| e.to_string())?;
        assert_eq!(mixed.samples.len(), 2);
        Ok(())
    }

    #[test]
    fn test_mix_reject_invalid_channels() {
        let samples = vec![0.0, 0.0, 0.0];
        assert!(ChannelMixer::mix_to_mono(&samples, 3).is_err());
    }

    #[test]
    fn test_mix_reject_misaligned_samples() {
        let samples = vec![0.0, 0.0, 0.0];
        assert!(ChannelMixer::mix_to_mono(&samples, 2).is_err());
    }

    #[test]
    fn test_mix_empty_input() -> TestResult<()> {
        let empty: Vec<f32> = Vec::new();
        let mixed = ChannelMixer::mix_to_mono(&empty, 1).map_err(|e| e.to_string())?;
        assert_eq!(mixed.samples.len(), 0);
        assert_eq!(mixed.sample_count(), 0);
        Ok(())
    }

    #[test]
    fn test_mix_single_frame_stereo() -> TestResult<()> {
        let stereo = vec![0.6, 0.4];
        let mixed = ChannelMixer::mix_to_mono(&stereo, 2).map_err(|e| e.to_string())?;
        assert_eq!(mixed.samples.len(), 1);
        assert!((mixed.samples[0] - 0.5).abs() < f32::EPSILON);
        Ok(())
    }

    #[test]
    fn test_mix_is_clipped_detection() -> TestResult<()> {
        let clipped = vec![1.0, 1.0];
        let unclipped = vec![0.5, 0.5];

        let mixed_clipped = ChannelMixer::mix_to_mono(&clipped, 2).map_err(|e| e.to_string())?;
        assert!(mixed_clipped.is_clipped());

        let mixed_unclipped =
            ChannelMixer::mix_to_mono(&unclipped, 2).map_err(|e| e.to_string())?;
        assert!(!mixed_unclipped.is_clipped());
        Ok(())
    }

    #[test]
    fn test_mix_peak_ratio_behavior() -> TestResult<()> {
        let stereo = vec![0.8, -0.8, 0.4, -0.4];
        let mixed = ChannelMixer::mix_to_mono(&stereo, 2).map_err(|e| e.to_string())?;
        assert!(mixed.peak_before_mix > 0.0);
        assert!(mixed.peak_ratio() <= 1.0);
        Ok(())
    }
}
