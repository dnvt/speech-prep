use crate::error::{Error, Result};

/// Band-limited resampler for mono audio.
///
/// Each output sample is a windowed-sinc interpolation of the input. When
/// downsampling, the sinc is stretched so it also acts as a low-pass filter
/// below the new Nyquist frequency, so high-frequency content is removed
/// instead of aliasing into the speech band.
///
/// For common rate pairs the taps for every fractional offset are
/// precomputed (a polyphase filter bank), so each output sample is a single
/// dot product.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SampleRateConverter;

/// Zero crossings of the sinc on each side of the centre. More gives a
/// sharper transition band at a higher cost.
const ZERO_CROSSINGS: usize = 24;

/// Passband edge as a fraction of the lower of the two Nyquist frequencies.
/// For 16 kHz output that is 7.2 kHz.
const CUTOFF: f64 = 0.9;

/// Kaiser window shape; 8.0 gives roughly 80 dB stopband attenuation.
const KAISER_BETA: f64 = 8.0;

/// Kernel table resolution, in points per input sample.
const OVERSAMPLE: usize = 128;

/// Largest number of precomputed filter phases (rate pairs with a larger
/// reduced output rate compute taps per sample instead).
const MAX_PHASES: u64 = 2048;

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

        let kernel = Kernel::new(from_rate, to_rate);
        let taps = 2 * kernel.half_width;

        // Output sample n sits at input position n * from / to. With
        // g = gcd(from, to) that is (n * m) / l with l = to / g and
        // m = from / g, so there are only `l` distinct fractional offsets.
        // Precompute one row of taps per offset when that table is small.
        let g = gcd(from_rate, to_rate);
        let (l, m) = (u64::from(to_rate / g), u64::from(from_rate / g));
        let frames_out = (samples.len() as u64 * l).div_ceil(m) as usize;

        let bank = (l <= MAX_PHASES).then(|| {
            (0..l)
                .map(|phase| kernel.taps(phase as f64 / l as f64))
                .collect::<Vec<_>>()
        });

        let mut scratch = Vec::new();
        Ok((0..frames_out)
            .map(|n| {
                let pos = n as u64 * m;
                let base = (pos / l) as usize;
                let phase = pos % l;
                let coeffs: &[f32] = match &bank {
                    Some(bank) => bank.get(phase as usize).map_or(&[], Vec::as_slice),
                    None => {
                        scratch = kernel.taps(phase as f64 / l as f64);
                        &scratch
                    }
                };
                // Taps cover input samples base - half_width + 1 ..= base + half_width.
                let first = (base + 1).checked_sub(kernel.half_width);
                match first.and_then(|f| samples.get(f..f + taps)) {
                    Some(window) => window.iter().zip(coeffs).map(|(x, c)| x * c).sum(),
                    None => (0..taps)
                        .filter_map(|j| {
                            let k = (base + 1 + j).checked_sub(kernel.half_width)?;
                            Some(samples.get(k)? * coeffs.get(j)?)
                        })
                        .sum(),
                }
            })
            .collect())
    }

    pub(crate) fn resample_to_16khz(samples: &[f32], from_rate: u32) -> Result<Vec<f32>> {
        Self::resample(samples, from_rate, Self::TARGET_SAMPLE_RATE)
    }
}

/// Symmetric low-pass interpolation kernel sampled at `OVERSAMPLE` points
/// per input sample.
struct Kernel {
    table: Vec<f64>,
    /// Taps on each side of the centre, in input samples.
    half_width: usize,
}

impl Kernel {
    fn new(from_rate: u32, to_rate: u32) -> Self {
        // Cutoff relative to the input Nyquist frequency.
        let fc = CUTOFF * (f64::from(to_rate) / f64::from(from_rate)).min(1.0);
        let half_width = (ZERO_CROSSINGS as f64 / fc).ceil();
        let points = half_width as usize * OVERSAMPLE + 2;

        let i0_beta = bessel_i0(KAISER_BETA);
        let table = (0..points)
            .map(|i| {
                let d = i as f64 / OVERSAMPLE as f64;
                let ratio = d / half_width;
                if ratio >= 1.0 {
                    return 0.0;
                }
                let window = bessel_i0(KAISER_BETA * (1.0 - ratio * ratio).sqrt()) / i0_beta;
                fc * sinc(fc * d) * window
            })
            .collect();

        Self {
            table,
            half_width: half_width as usize,
        }
    }

    /// Taps for an output sample `frac` input samples past input `base`,
    /// ordered from input `base - half_width + 1` to `base + half_width`.
    fn taps(&self, frac: f64) -> Vec<f32> {
        let half = self.half_width as f64;
        (0..2 * self.half_width)
            .map(|j| self.at((j as f64 - half + 1.0 - frac).abs()) as f32)
            .collect()
    }

    /// Kernel value at `distance` input samples from the centre.
    fn at(&self, distance: f64) -> f64 {
        let pos = distance * OVERSAMPLE as f64;
        let i = pos.floor() as usize;
        let t = pos - i as f64;
        match (self.table.get(i), self.table.get(i + 1)) {
            (Some(&a), Some(&b)) => a + (b - a) * t,
            _ => 0.0,
        }
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Zeroth-order modified Bessel function of the first kind (power series).
fn bessel_i0(x: f64) -> f64 {
    let half_sq = (x / 2.0) * (x / 2.0);
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..50 {
        term *= half_sq / (k * k) as f64;
        sum += term;
        if term < sum * 1e-16 {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f64, rate: u32, seconds: f64) -> Vec<f32> {
        (0..(f64::from(rate) * seconds) as usize)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    /// RMS over the middle half, away from edge transients.
    fn mid_rms(samples: &[f32]) -> f32 {
        let mid = &samples[samples.len() / 4..samples.len() * 3 / 4];
        (mid.iter().map(|s| s * s).sum::<f32>() / mid.len() as f32).sqrt()
    }

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
    fn passband_tone_is_preserved() {
        for rate in [8_000, 22_050, 44_100, 48_000] {
            let output =
                SampleRateConverter::resample_to_16khz(&tone(1_000.0, rate, 1.0), rate).unwrap();
            let gain_db = 20.0 * (mid_rms(&output) / std::f32::consts::FRAC_1_SQRT_2).log10();
            assert!(gain_db.abs() < 0.1, "from {rate} Hz: {gain_db:.3} dB");
        }
    }

    #[test]
    fn content_above_nyquist_does_not_alias() {
        // 12 kHz would alias to 4 kHz with plain interpolation.
        for rate in [44_100, 48_000] {
            let output =
                SampleRateConverter::resample_to_16khz(&tone(12_000.0, rate, 1.0), rate).unwrap();
            let level_db = 20.0 * (mid_rms(&output) / std::f32::consts::FRAC_1_SQRT_2).log10();
            assert!(level_db < -60.0, "from {rate} Hz: {level_db:.1} dB");
        }
    }

    #[test]
    fn unusual_rate_without_filter_bank() {
        // 44_101 / 16_000 reduces to 44_101 / 16_000: too many phases for the
        // bank, so taps are computed per sample.
        let output =
            SampleRateConverter::resample_to_16khz(&tone(1_000.0, 44_101, 1.0), 44_101).unwrap();
        assert_eq!(output.len(), 16_000);
        let gain_db = 20.0 * (mid_rms(&output) / std::f32::consts::FRAC_1_SQRT_2).log10();
        assert!(gain_db.abs() < 0.1, "{gain_db:.3} dB");
    }

    #[test]
    fn rejects_zero_rate() {
        assert!(SampleRateConverter::resample(&[0.0], 0, 16_000).is_err());
        assert!(SampleRateConverter::resample(&[0.0], 16_000, 0).is_err());
    }
}
