use crate::error::{Error, Result};

/// Sample rate converter using linear interpolation.
#[derive(Debug, Default, Clone, Copy)]
pub struct SampleRateConverter;

impl SampleRateConverter {
    /// Standard target sample rate (16kHz) used across the pipeline.
    pub const TARGET_SAMPLE_RATE: u32 = 16_000;

    /// Construct a new converter.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Resample audio between arbitrary rates using linear interpolation.
    pub fn resample(
        samples: &[f32],
        channels: u8,
        from_rate: u32,
        to_rate: u32,
    ) -> Result<Vec<f32>> {
        if channels == 0 {
            return Err(Error::InvalidInput("channel count cannot be zero".into()));
        }
        if from_rate == 0 {
            return Err(Error::InvalidInput(
                "input sample rate cannot be zero".into(),
            ));
        }
        if to_rate == 0 {
            return Err(Error::InvalidInput(
                "output sample rate cannot be zero".into(),
            ));
        }

        if from_rate == to_rate {
            return Ok(samples.to_vec());
        }
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let channel_count = channels as usize;
        if !samples.len().is_multiple_of(channel_count) {
            return Err(Error::InvalidInput(format!(
                "sample count {} not divisible by channel count {}",
                samples.len(),
                channels
            )));
        }

        let frames_in = samples.len() / channel_count;
        if frames_in == 0 {
            return Ok(Vec::new());
        }

        let ratio = f64::from(to_rate) / f64::from(from_rate);
        let frames_out = (frames_in as f64 * ratio).ceil() as usize;
        let output_len = frames_out * channel_count;
        let mut output = Vec::with_capacity(output_len);

        for frame_out in 0..frames_out {
            let input_pos = (frame_out as f64) / ratio;
            let idx = input_pos.floor() as usize;
            let frac = (input_pos - idx as f64) as f32;

            for channel_idx in 0..channel_count {
                let sample = Self::interpolate_channel(
                    samples,
                    channel_count,
                    frames_in,
                    channel_idx,
                    idx,
                    frac,
                );
                output.push(sample);
            }
        }

        Ok(output)
    }

    /// Convenience helper that resamples directly to 16kHz.
    pub fn resample_to_16khz(samples: &[f32], channels: u8, from_rate: u32) -> Result<Vec<f32>> {
        Self::resample(samples, channels, from_rate, Self::TARGET_SAMPLE_RATE)
    }

    #[inline]
    #[allow(clippy::indexing_slicing)]
    fn interpolate_channel(
        samples: &[f32],
        channel_count: usize,
        frames_in: usize,
        channel_idx: usize,
        frame_idx: usize,
        frac: f32,
    ) -> f32 {
        if frames_in == 0 {
            return 0.0;
        }

        let idx_clamped = frame_idx.min(frames_in - 1);
        let base = idx_clamped * channel_count + channel_idx;
        let s0 = samples[base];

        let next_frame = idx_clamped + 1;
        let s1 = if next_frame < frames_in {
            samples[next_frame * channel_count + channel_idx]
        } else {
            s0
        };

        frac.mul_add(s1 - s0, s0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult<T> = std::result::Result<T, String>;

    #[test]
    fn test_resample_identity_16khz() -> TestResult<()> {
        let input = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let output =
            SampleRateConverter::resample(&input, 1, 16_000, 16_000).map_err(|e| e.to_string())?;

        assert_eq!(output, input);
        Ok(())
    }

    #[test]
    fn test_resample_44100_to_16000() -> TestResult<()> {
        let input = vec![0.0f32; 44_100];
        let output =
            SampleRateConverter::resample(&input, 1, 44_100, 16_000).map_err(|e| e.to_string())?;

        assert_eq!(output.len(), 16_000);
        assert!(output.iter().all(|&s| s.abs() < f32::EPSILON));
        Ok(())
    }

    #[test]
    fn test_resample_48000_to_16000() -> TestResult<()> {
        let input = vec![0.0f32; 48_000];
        let output =
            SampleRateConverter::resample(&input, 1, 48_000, 16_000).map_err(|e| e.to_string())?;

        assert_eq!(output.len(), 16_000);
        Ok(())
    }

    #[test]
    fn test_resample_8000_to_16000() -> TestResult<()> {
        let input = vec![0.0f32; 8_000];
        let output =
            SampleRateConverter::resample(&input, 1, 8_000, 16_000).map_err(|e| e.to_string())?;

        assert_eq!(output.len(), 16_000);
        Ok(())
    }

    #[test]
    fn test_resample_preserves_amplitude() -> TestResult<()> {
        let input = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
        let output = SampleRateConverter::resample(&input, 1, 8, 16).map_err(|e| e.to_string())?;

        let max_input = input.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let max_output = output.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

        assert!((max_input - max_output).abs() < 0.05);
        Ok(())
    }

    #[test]
    fn test_resample_to_16khz_helper() -> TestResult<()> {
        let input = vec![0.0f32; 8_000];
        let output =
            SampleRateConverter::resample_to_16khz(&input, 1, 8_000).map_err(|e| e.to_string())?;
        assert_eq!(output.len(), 16_000);
        Ok(())
    }

    #[test]
    fn test_resample_reject_zero_channels() {
        let samples = vec![0.0, 0.0];
        let result = SampleRateConverter::resample(&samples, 0, 16_000, 8_000);
        assert!(result.is_err());
    }

    #[test]
    fn test_resample_reject_misaligned_samples() {
        let samples = vec![0.0, 0.0, 0.0];
        let result = SampleRateConverter::resample(&samples, 2, 16_000, 8_000);
        assert!(result.is_err());
    }
}
