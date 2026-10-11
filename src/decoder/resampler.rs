use crate::error::{Error, Result};

/// Linear-interpolation resampler for mono audio.
///
/// Cheap and dependency-free, but it does not low-pass filter before
/// downsampling, so content above the new Nyquist frequency aliases. For
/// 44.1/48 kHz speech going to 16 kHz this is usually acceptable; use a
/// dedicated resampler when it is not.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SampleRateConverter;

impl SampleRateConverter {
    /// The crate's standard sample rate.
    pub(crate) const TARGET_SAMPLE_RATE: u32 = 16_000;

    pub(crate) fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
        if from_rate == 0 || to_rate == 0 {
            return Err(Error::InvalidInput("sample rate cannot be zero".into()));
        }
        if from_rate == to_rate || samples.is_empty() {
            return Ok(samples.to_vec());
        }

        let step = f64::from(from_rate) / f64::from(to_rate);
        let frames_out = (samples.len() as f64 / step).ceil() as usize;
        let last = samples.len() - 1;

        Ok((0..frames_out)
            .map(|n| {
                let pos = n as f64 * step;
                let idx = (pos.floor() as usize).min(last);
                let frac = (pos - idx as f64) as f32;
                let s0 = samples.get(idx).copied().unwrap_or(0.0);
                let s1 = samples.get(idx + 1).copied().unwrap_or(s0);
                frac.mul_add(s1 - s0, s0)
            })
            .collect())
    }

    pub(crate) fn resample_to_16khz(samples: &[f32], from_rate: u32) -> Result<Vec<f32>> {
        Self::resample(samples, from_rate, Self::TARGET_SAMPLE_RATE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_rate_is_identity() {
        let input = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        assert_eq!(
            SampleRateConverter::resample(&input, 16_000, 16_000).unwrap(),
            input
        );
    }

    #[test]
    fn output_length_matches_duration() {
        for rate in [8_000, 22_050, 44_100, 48_000] {
            let input = vec![0.0f32; rate as usize];
            let output = SampleRateConverter::resample_to_16khz(&input, rate).unwrap();
            assert_eq!(output.len(), 16_000, "from {rate} Hz");
        }
    }

    #[test]
    fn upsampling_interpolates() {
        let output = SampleRateConverter::resample(&[0.0, 1.0], 8, 16).unwrap();
        assert_eq!(output, vec![0.0, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn rejects_zero_rate() {
        assert!(SampleRateConverter::resample(&[0.0], 0, 16_000).is_err());
        assert!(SampleRateConverter::resample(&[0.0], 16_000, 0).is_err());
    }
}
