//! DC offset removal and Butterworth high-pass filtering.
//!
//! Removes DC bias and low-frequency rumble before spectral analysis. Filter
//! state is carried across calls, so a stream can be processed in chunks
//! without discontinuities at chunk boundaries.

use crate::error::{Error, Result};

/// Configuration for [`DcHighPassFilter`].
///
/// ```
/// use speech_prep::preprocessing::PreprocessingConfig;
///
/// let config = PreprocessingConfig {
///     highpass_cutoff_hz: 120.0,
///     ..Default::default()
/// };
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PreprocessingConfig {
    /// High-pass cutoff (-3 dB point) in Hz. Default: 80.
    ///
    /// 60-80 Hz suits most speech; raise it to 100-120 Hz for heavy rumble.
    pub highpass_cutoff_hz: f32,

    /// Sample rate of the input in Hz. Default: 16000.
    pub sample_rate_hz: u32,

    /// Smoothing factor for the running DC estimate, in `(0, 1)`.
    /// Default: 0.95. Higher values adapt more slowly.
    ///
    /// Each call updates the estimate as
    /// `bias = alpha * bias + (1 - alpha) * mean(chunk)`.
    pub dc_bias_alpha: f32,

    /// Subtract the running DC estimate. Default: true.
    pub enable_dc_removal: bool,

    /// Apply the high-pass filter. Default: true.
    pub enable_highpass: bool,

    /// Filter order. Default: [`HighpassOrder::FourthOrder`].
    pub highpass_order: HighpassOrder,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            highpass_cutoff_hz: 80.0,
            sample_rate_hz: 16_000,
            dc_bias_alpha: 0.95,
            enable_dc_removal: true,
            enable_highpass: true,
            highpass_order: HighpassOrder::FourthOrder,
        }
    }
}

impl PreprocessingConfig {
    /// Check that the configuration is usable.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] if the sample rate is zero, the cutoff
    /// is below 20 Hz or at/above Nyquist, or `dc_bias_alpha` is outside
    /// `(0, 1)`.
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate_hz == 0 {
            return Err(Error::Configuration(
                "sample_rate_hz must be greater than zero".into(),
            ));
        }

        // The `is_finite` guard rejects NaN and infinity, which would
        // otherwise pass every comparison and poison the filter output.
        if !(self.highpass_cutoff_hz.is_finite() && self.highpass_cutoff_hz >= 20.0) {
            return Err(Error::Configuration(format!(
                "highpass_cutoff_hz {:.1} must be a finite value of at least 20 Hz",
                self.highpass_cutoff_hz
            )));
        }

        let nyquist = self.sample_rate_hz as f32 / 2.0;
        if self.highpass_cutoff_hz >= nyquist {
            return Err(Error::Configuration(format!(
                "highpass_cutoff_hz {:.1} must be below Nyquist ({nyquist:.1} Hz)",
                self.highpass_cutoff_hz
            )));
        }

        if !(self.dc_bias_alpha > 0.0 && self.dc_bias_alpha < 1.0) {
            return Err(Error::Configuration(format!(
                "dc_bias_alpha {:.3} must be in (0, 1)",
                self.dc_bias_alpha
            )));
        }

        Ok(())
    }
}

/// High-pass filter order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HighpassOrder {
    /// 2nd-order Butterworth (12 dB/octave), one biquad.
    SecondOrder,
    /// 4th-order Butterworth (24 dB/octave), two cascaded biquads.
    #[default]
    FourthOrder,
}

impl HighpassOrder {
    /// Q factor of each biquad section.
    ///
    /// A Butterworth filter of order `n` is a cascade of biquads whose Q
    /// values come from the pole angles; reusing Q = 0.707 for every stage
    /// would put the cutoff at -6 dB instead of -3 dB.
    fn stage_q_values(self) -> &'static [f32] {
        match self {
            Self::SecondOrder => &[std::f32::consts::FRAC_1_SQRT_2],
            Self::FourthOrder => &[0.541_196_1, 1.306_563],
        }
    }
}

/// Speech/silence hint passed to the preprocessing stages.
///
/// [`DcHighPassFilter`] only updates its DC estimate during silence, and
/// [`NoiseReducer`](super::NoiseReducer) only learns its noise profile during
/// silence. Passing `None` instead of a context means "no VAD information".
#[derive(Debug, Clone, Copy)]
pub struct VadContext {
    /// `true` if the audio being processed is silence.
    pub is_silence: bool,
}

/// DC offset removal followed by a Butterworth high-pass filter.
///
/// ```
/// use speech_prep::preprocessing::{DcHighPassFilter, PreprocessingConfig};
///
/// # fn main() -> speech_prep::Result<()> {
/// let mut filter = DcHighPassFilter::new(PreprocessingConfig::default())?;
///
/// // State carries over between calls, so chunks can be fed in order.
/// for chunk in [vec![0.1; 8000], vec![0.1; 8000]] {
///     let clean = filter.process(&chunk, None)?;
///     assert_eq!(clean.len(), chunk.len());
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct DcHighPassFilter {
    config: PreprocessingConfig,
    stages: Vec<Biquad>,
    dc_bias: f32,
}

#[derive(Debug, Clone, Copy)]
struct BiquadCoefficients {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct BiquadState {
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

#[derive(Debug, Clone, Copy)]
struct Biquad {
    coeffs: BiquadCoefficients,
    state: BiquadState,
}

impl Biquad {
    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let c = &self.coeffs;
        let s = &mut self.state;
        let acc = c.b0.mul_add(input, c.b1.mul_add(s.x1, c.b2 * s.x2));
        let output = acc - c.a1.mul_add(s.y1, c.a2 * s.y2);
        s.x2 = s.x1;
        s.x1 = input;
        s.y2 = s.y1;
        s.y1 = output;
        output
    }
}

impl BiquadState {
    #[cfg(test)]
    fn is_reset(self) -> bool {
        self == Self::default()
    }
}

impl DcHighPassFilter {
    /// Create a filter from a validated configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] if `config` is invalid.
    pub fn new(config: PreprocessingConfig) -> Result<Self> {
        config.validate()?;

        let stages = config
            .highpass_order
            .stage_q_values()
            .iter()
            .map(|&q| {
                highpass_coefficients(config.highpass_cutoff_hz, config.sample_rate_hz, q).map(
                    |coeffs| Biquad {
                        coeffs,
                        state: BiquadState::default(),
                    },
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            config,
            stages,
            dc_bias: 0.0,
        })
    }

    /// Filter one chunk of samples.
    ///
    /// When `vad_context` is `Some`, the DC estimate is only updated if the
    /// chunk is silence, so speech energy does not leak into the estimate.
    ///
    /// # Errors
    ///
    /// Currently infallible; the `Result` leaves room for input validation.
    pub fn process(
        &mut self,
        samples: &[f32],
        vad_context: Option<VadContext>,
    ) -> Result<Vec<f32>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let should_update_bias = vad_context.is_none_or(|ctx| ctx.is_silence);
        if self.config.enable_dc_removal && should_update_bias {
            self.update_dc_bias(samples);
        }

        let mut output = Vec::with_capacity(samples.len());
        for &sample in samples {
            let mut next = if self.config.enable_dc_removal {
                sample - self.dc_bias
            } else {
                sample
            };
            if self.config.enable_highpass {
                for stage in &mut self.stages {
                    next = stage.process(next);
                }
            }
            output.push(next);
        }

        Ok(output)
    }

    /// Clear filter history and the DC estimate before starting a new stream.
    pub fn reset(&mut self) {
        for stage in &mut self.stages {
            stage.state = BiquadState::default();
        }
        self.dc_bias = 0.0;
    }

    /// Current DC offset estimate.
    #[must_use]
    pub fn dc_bias(&self) -> f32 {
        self.dc_bias
    }

    /// The configuration this filter was built with.
    #[must_use]
    pub fn config(&self) -> &PreprocessingConfig {
        &self.config
    }

    fn update_dc_bias(&mut self, samples: &[f32]) {
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let alpha = self.config.dc_bias_alpha;
        self.dc_bias = alpha.mul_add(self.dc_bias, (1.0 - alpha) * mean);
    }
}

/// Biquad high-pass coefficients (RBJ Audio EQ Cookbook), normalized so that
/// `a0 = 1`.
fn highpass_coefficients(
    cutoff_hz: f32,
    sample_rate_hz: u32,
    q: f32,
) -> Result<BiquadCoefficients> {
    use std::f32::consts::PI;

    let w0 = 2.0 * PI * cutoff_hz / sample_rate_hz as f32;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let a0 = 1.0 + alpha;

    let coeffs = BiquadCoefficients {
        b0: f32::midpoint(1.0, cos_w0) / a0,
        b1: -(1.0 + cos_w0) / a0,
        b2: f32::midpoint(1.0, cos_w0) / a0,
        a1: (-2.0 * cos_w0) / a0,
        a2: (1.0 - alpha) / a0,
    };

    let all_finite = [coeffs.b0, coeffs.b1, coeffs.b2, coeffs.a1, coeffs.a2]
        .iter()
        .all(|c| c.is_finite());
    if !all_finite {
        return Err(Error::Processing(format!(
            "invalid high-pass coefficients for fc={cutoff_hz:.1} Hz, fs={sample_rate_hz} Hz"
        )));
    }

    Ok(coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test helper type
    type TestResult<T> = std::result::Result<T, String>;

    #[test]
    fn non_finite_config_is_rejected() {
        let base = PreprocessingConfig::default();
        let bad = [
            PreprocessingConfig {
                highpass_cutoff_hz: f32::NAN,
                ..base
            },
            PreprocessingConfig {
                highpass_cutoff_hz: f32::INFINITY,
                ..base
            },
            PreprocessingConfig {
                dc_bias_alpha: f32::NAN,
                ..base
            },
            PreprocessingConfig {
                dc_bias_alpha: 0.0,
                ..base
            },
            PreprocessingConfig {
                dc_bias_alpha: 1.0,
                ..base
            },
        ];
        for cfg in bad {
            assert!(cfg.validate().is_err(), "should reject {cfg:?}");
        }
        assert!(base.validate().is_ok());
    }

    // Generate sine wave for testing
    fn generate_sine_wave(
        frequency: f32,
        sample_rate: u32,
        duration_secs: f32,
        amplitude: f32,
    ) -> Vec<f32> {
        use std::f32::consts::PI;
        let samples = (sample_rate as f32 * duration_secs).round() as usize;
        (0..samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * PI * frequency * t).sin() * amplitude
            })
            .collect()
    }

    // Calculate RMS (root mean square)
    fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_sq / samples.len() as f32).sqrt()
    }

    // Calculate attenuation in dB
    fn calculate_attenuation_db(input: &[f32], output: &[f32]) -> f32 {
        let rms_in = calculate_rms(input);
        let rms_out = calculate_rms(output);

        if rms_in == 0.0 || rms_out == 0.0 {
            return 0.0;
        }

        20.0 * (rms_out / rms_in).log10()
    }

    #[test]
    fn test_dc_offset_removal_synthetic_bias() -> TestResult<()> {
        // Create realistic audio with DC offset: sine wave + DC bias
        // This simulates real-world scenario better than constant DC
        let dc_offset = 0.5;
        let mut samples_with_dc: Vec<f32> = generate_sine_wave(440.0, 16000, 0.5, 0.3);
        for sample in &mut samples_with_dc {
            *sample += dc_offset;
        }

        let config = PreprocessingConfig {
            enable_dc_removal: true,
            enable_highpass: false, // DC removal only for this test
            dc_bias_alpha: 0.5,     // Fast convergence for test (vs default 0.95)
            ..Default::default()
        };
        let mut filter = DcHighPassFilter::new(config).map_err(|e| e.to_string())?;

        // Simulate streaming: process 10 chunks to allow EMA convergence
        // With α=0.5, 10 iterations gives (1 - 0.5^10) ≈ 99.9% convergence
        let mut final_output = Vec::new();
        for _ in 0..10 {
            final_output = filter
                .process(&samples_with_dc, None)
                .map_err(|e| e.to_string())?;
        }

        // After convergence (10 chunks with α=0.5), residual DC should be < 0.001 RMS
        let mean: f32 = final_output.iter().sum::<f32>() / final_output.len() as f32;
        assert!(
            mean.abs() < 0.001,
            "DC residual too high after convergence: {:.6} (expected < 0.001)",
            mean
        );

        // DC bias estimate should be close to 0.5 (within 1%)
        assert!(
            (filter.dc_bias() - dc_offset).abs() < 0.005,
            "DC bias estimate {:.6} not converged to {:.6}",
            filter.dc_bias(),
            dc_offset
        );

        Ok(())
    }

    #[test]
    fn test_highpass_frequency_response() -> TestResult<()> {
        let config = PreprocessingConfig {
            highpass_cutoff_hz: 80.0,
            enable_dc_removal: false, // Filter only for this test
            ..Default::default()
        };
        let mut filter = DcHighPassFilter::new(config).map_err(|e| e.to_string())?;

        // Test at 20 Hz (fc/4). Fourth-order should attenuate ≥30 dB
        let input_20hz = generate_sine_wave(20.0, 16000, 1.0, 1.0);
        let output_20hz = filter
            .process(&input_20hz, None)
            .map_err(|e| e.to_string())?;
        filter.reset(); // Reset state for independent test

        let attenuation_20hz = calculate_attenuation_db(&input_20hz, &output_20hz);
        assert!(
            attenuation_20hz <= -30.0,
            "Insufficient attenuation at 20Hz: {:.1} dB (expected ≤ -30 dB)",
            attenuation_20hz
        );

        // Test at 40 Hz (fc/2). Fourth-order should attenuate ≥20 dB
        let input_40hz = generate_sine_wave(40.0, 16000, 1.0, 1.0);
        let output_40hz = filter
            .process(&input_40hz, None)
            .map_err(|e| e.to_string())?;
        filter.reset(); // Reset state for independent test

        let attenuation_40hz = calculate_attenuation_db(&input_40hz, &output_40hz);
        assert!(
            attenuation_40hz <= -20.0,
            "Insufficient attenuation at 40Hz: {:.1} dB (expected ≤ -20 dB)",
            attenuation_40hz
        );

        // Test at 150 Hz (should pass with <1 dB loss)
        let input_150hz = generate_sine_wave(150.0, 16000, 1.0, 1.0);
        let output_150hz = filter
            .process(&input_150hz, None)
            .map_err(|e| e.to_string())?;

        let loss_150hz = calculate_attenuation_db(&input_150hz, &output_150hz);
        assert!(
            loss_150hz > -1.0,
            "Excessive loss at 150Hz: {:.1} dB (expected > -1 dB)",
            loss_150hz
        );

        Ok(())
    }

    #[test]
    fn test_cutoff_is_minus_3db() -> TestResult<()> {
        for order in [HighpassOrder::SecondOrder, HighpassOrder::FourthOrder] {
            let config = PreprocessingConfig {
                highpass_cutoff_hz: 80.0,
                enable_dc_removal: false,
                highpass_order: order,
                ..Default::default()
            };
            let mut filter = DcHighPassFilter::new(config).map_err(|e| e.to_string())?;
            let input = generate_sine_wave(80.0, 16000, 2.0, 1.0);
            let output = filter.process(&input, None).map_err(|e| e.to_string())?;

            // Skip the first second so the filter transient has settled.
            let gain_db = calculate_attenuation_db(&input[16000..], &output[16000..]);
            assert!(
                (gain_db + 3.0).abs() < 0.3,
                "{order:?}: gain at cutoff is {gain_db:.2} dB, expected about -3 dB"
            );
        }
        Ok(())
    }

    #[test]
    fn test_chunk_boundary_continuity() -> TestResult<()> {
        // Process as single buffer
        let long_signal = generate_sine_wave(440.0, 16000, 1.0, 0.5); // 1 second
        let config = PreprocessingConfig::default();
        let mut filter1 = DcHighPassFilter::new(config).map_err(|e| e.to_string())?;
        let output_single = filter1
            .process(&long_signal, None)
            .map_err(|e| e.to_string())?;

        // Process as two chunks
        let mut filter2 = DcHighPassFilter::new(config).map_err(|e| e.to_string())?;
        let mid = long_signal.len() / 2;
        let chunk1 = &long_signal[0..mid];
        let chunk2 = &long_signal[mid..];
        let output_chunk1 = filter2.process(chunk1, None).map_err(|e| e.to_string())?;
        let output_chunk2 = filter2.process(chunk2, None).map_err(|e| e.to_string())?;

        // Concatenate chunked output
        let output_chunked: Vec<f32> = output_chunk1.into_iter().chain(output_chunk2).collect();

        // Verify outputs match (within numerical precision)
        // Note: Using 5e-5 tolerance because EMA DC bias updates accumulate
        // floating-point drift at chunk boundaries — the filter carries state
        // across chunks via f32 accumulators, and intermediate rounding differs
        // between single-pass and chunked paths. 5e-5 still catches real
        // discontinuities while allowing normal f32 accumulation drift.
        for (i, (single, chunked)) in output_single.iter().zip(output_chunked.iter()).enumerate() {
            let diff = (single - chunked).abs();
            assert!(
                diff < 5e-5,
                "Discontinuity at sample {}: diff={:.9} (single={:.9}, chunked={:.9})",
                i,
                diff,
                single,
                chunked
            );
        }

        Ok(())
    }

    #[test]
    fn test_vad_informed_dc_update() -> TestResult<()> {
        let config = PreprocessingConfig::default();
        let mut filter = DcHighPassFilter::new(config).map_err(|e| e.to_string())?;

        // Speech chunk (don't update DC)
        let speech_samples = vec![0.1, 0.2, -0.1, 0.3];
        let speech_ctx = VadContext { is_silence: false };
        let initial_bias = filter.dc_bias();
        filter
            .process(&speech_samples, Some(speech_ctx))
            .map_err(|e| e.to_string())?;

        // DC bias should NOT change during speech
        assert_eq!(
            filter.dc_bias(),
            initial_bias,
            "DC bias changed during speech"
        );

        // Silence chunk (update DC)
        let silence_samples = vec![0.5; 1000];
        let silence_ctx = VadContext { is_silence: true };
        filter
            .process(&silence_samples, Some(silence_ctx))
            .map_err(|e| e.to_string())?;

        // DC bias should adapt toward 0.5
        assert!(
            filter.dc_bias() > initial_bias,
            "DC bias did not adapt during silence (initial={:.6}, after={:.6})",
            initial_bias,
            filter.dc_bias()
        );

        Ok(())
    }

    #[test]
    fn test_configuration_validation() {
        // Valid configuration should pass
        let valid_config = PreprocessingConfig::default();
        assert!(valid_config.validate().is_ok());

        // Cutoff too low
        let config_low = PreprocessingConfig {
            highpass_cutoff_hz: 10.0,
            ..Default::default()
        };
        assert!(config_low.validate().is_err());

        // Cutoff above Nyquist
        let config_high = PreprocessingConfig {
            highpass_cutoff_hz: 9000.0, // > 8000 Hz (Nyquist for 16kHz)
            ..Default::default()
        };
        assert!(config_high.validate().is_err());

        // Invalid EMA alpha
        let config_alpha = PreprocessingConfig {
            dc_bias_alpha: 1.0, // Must be < 1.0
            ..Default::default()
        };
        assert!(config_alpha.validate().is_err());

        // Zero sample rate
        let config_zero_sr = PreprocessingConfig {
            sample_rate_hz: 0,
            ..Default::default()
        };
        assert!(config_zero_sr.validate().is_err());
    }

    #[test]
    fn test_reset_clears_state() -> TestResult<()> {
        let config = PreprocessingConfig::default();
        let mut filter = DcHighPassFilter::new(config).map_err(|e| e.to_string())?;

        // Process some audio
        let samples = generate_sine_wave(440.0, 16000, 0.5, 0.8);
        filter.process(&samples, None).map_err(|e| e.to_string())?;

        // Verify state is non-zero
        assert_ne!(
            filter.dc_bias(),
            0.0,
            "DC bias should be non-zero after processing"
        );
        assert!(
            filter.stages.iter().any(|stage| !stage.state.is_reset()),
            "Filter stages should accumulate state after processing"
        );

        // Reset
        filter.reset();

        // Verify state cleared
        assert_eq!(filter.dc_bias(), 0.0, "DC bias should be zero after reset");
        assert!(
            filter.stages.iter().all(|stage| stage.state.is_reset()),
            "Filter stages should be reset to zero state"
        );

        Ok(())
    }

    #[test]
    fn test_empty_input() -> TestResult<()> {
        let config = PreprocessingConfig::default();
        let mut filter = DcHighPassFilter::new(config).map_err(|e| e.to_string())?;

        let output = filter.process(&[], None).map_err(|e| e.to_string())?;
        assert!(output.is_empty());

        Ok(())
    }
}
