use crate::error::{Error, Result};

/// Downmixes interleaved multi-channel audio to mono.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChannelMixer;

impl ChannelMixer {
    /// Average each frame of `channels` interleaved samples into one sample.
    pub(crate) fn mix_to_mono(samples: &[f32], channels: u16) -> Result<Vec<f32>> {
        let channels = usize::from(channels);
        if channels == 0 {
            return Err(Error::InvalidInput("channel count cannot be zero".into()));
        }
        if channels == 1 {
            return Ok(samples.to_vec());
        }
        if !samples.len().is_multiple_of(channels) {
            return Err(Error::InvalidInput(format!(
                "sample count {} is not a multiple of channel count {channels}",
                samples.len()
            )));
        }

        let scale = 1.0 / channels as f32;
        Ok(samples
            .chunks_exact(channels)
            .map(|frame| (frame.iter().sum::<f32>() * scale).clamp(-1.0, 1.0))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_is_unchanged() {
        let mono = vec![0.1, -0.2, 0.3];
        assert_eq!(ChannelMixer::mix_to_mono(&mono, 1).unwrap(), mono);
    }

    #[test]
    fn averages_stereo_frames() {
        let stereo = vec![0.6, 0.4, 1.0, -1.0];
        assert_eq!(
            ChannelMixer::mix_to_mono(&stereo, 2).unwrap(),
            vec![0.5, 0.0]
        );
    }

    #[test]
    fn handles_odd_channel_counts() {
        let three = vec![0.3, 0.3, 0.3, 0.0, 0.0, 0.9];
        let mixed = ChannelMixer::mix_to_mono(&three, 3).unwrap();
        assert_eq!(mixed.len(), 2);
        assert!((mixed[0] - 0.3).abs() < 1e-6);
        assert!((mixed[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(ChannelMixer::mix_to_mono(&[0.0; 3], 0).is_err());
        assert!(ChannelMixer::mix_to_mono(&[0.0; 3], 2).is_err());
    }

    #[test]
    fn empty_input() {
        assert!(ChannelMixer::mix_to_mono(&[], 2).unwrap().is_empty());
    }
}
