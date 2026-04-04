//! DC offset removal and high-pass filtering.
//!
//! Removes DC bias and attenuates low-frequency rumble (<80 Hz) to prepare
//! audio for spectral analysis and noise reduction.

use crate::error::{Error, Result};
use crate::time::{AudioDuration, AudioInstant};
use tracing::{info, warn};

/// Configuration for DC offset removal and high-pass filtering.
///
/// # Examples
///
/// ```rust,no_run
/// use speech_prep::preprocessing::PreprocessingConfig;
///
/// // Default: 80 Hz high-pass, 16kHz sample rate, EMA α=0.95
/// let config = PreprocessingConfig::default();
///
/// // Custom configuration for noisy environment
/// let config = PreprocessingConfig {
///     highpass_cutoff_hz: 120.0, // More aggressive low-frequency removal
///     dc_bias_alpha: 0.98,       // Slower DC adaptation
///     ..Default::default()
/// };
/// # Ok::<(), speech_prep::error::Error>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PreprocessingConfig {
    /// High-pass filter cutoff frequency in Hz.
    ///
    /// **Range**: 60.0 - 120.0
    /// **Default**: 80.0
    ///
    /// **Effect**: Frequencies below this cutoff are attenuated (≥20 dB at
    /// fc/2). Higher cutoffs remove more low-frequency content but may
    /// affect speech naturalness.
    ///
    /// **Recommendation**:
    /// - 60-80 Hz: Standard speech (default)
    /// - 80-100 Hz: Noisy environments with HVAC/rumble
    /// - 100-120 Hz: Extreme low-frequency noise
    pub highpass_cutoff_hz: f32,

    /// Audio sample rate in Hz.
    ///
    /// **Typical Values**: 16000, 44100, 48000
    /// **Default**: 16000
    ///
    /// **Effect**: Determines filter coefficient calculation.
    /// Must match the actual sample rate of input audio.
    pub sample_rate_hz: u32,

    /// EMA smoothing factor for DC bias estimation.
    ///
    /// **Range**: 0.9 - 0.99
    /// **Default**: 0.95
    ///
    /// **Effect**: Controls adaptation speed of DC bias tracking.
    /// - Higher (0.95-0.99): Slower adaptation, smoother (recommended)
    /// - Lower (0.90-0.94): Faster adaptation, less smooth
    ///
    /// **Formula**: `bias_new = α × bias_old + (1-α) × sample_mean`
    pub dc_bias_alpha: f32,

    /// Enable DC offset removal stage.
    ///
    /// **Default**: true
    ///
    /// **Effect**: When false, DC removal is skipped (filter-only mode).
    /// Useful if audio is already DC-free (rare).
    pub enable_dc_removal: bool,

    /// Enable high-pass filtering stage.
    ///
    /// **Default**: true
    ///
    /// **Effect**: When false, high-pass filter is skipped (DC-only mode).
    /// Useful for testing or if audio already high-pass filtered.
    pub enable_highpass: bool,

    /// Order of the high-pass filter.
    ///
    /// **Default**: `FourthOrder` (two cascaded biquads)
    ///
    /// **Effect**: Higher order increases low-frequency attenuation at the cost
    /// of additional computation. `FourthOrder` meets the ≥20 dB @ 40 Hz
    /// target.
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
    /// Validate configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns `Error::Configuration` if:
    /// - `highpass_cutoff_hz` < 20 Hz (too low, ineffective)
    /// - `highpass_cutoff_hz` >= Nyquist frequency (fs/2)
    /// - `dc_bias_alpha` not in (0.0, 1.0)
    /// - `sample_rate_hz` is zero
    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate_hz == 0 {
            return Err(Error::Configuration(
                "sample_rate_hz must be greater than zero".into(),
            ));
        }

        if self.highpass_cutoff_hz < 20.0 {
            return Err(Error::Configuration(format!(
                "Cutoff {:.1} Hz too low (minimum 20 Hz)",
                self.highpass_cutoff_hz
            )));
        }

        let nyquist = self.sample_rate_hz as f32 / 2.0;
        if self.highpass_cutoff_hz >= nyquist {
            return Err(Error::Configuration(format!(
                "Cutoff {:.1} Hz exceeds Nyquist {:.1} Hz",
                self.highpass_cutoff_hz, nyquist
            )));
        }

        if self.dc_bias_alpha <= 0.0 || self.dc_bias_alpha >= 1.0 {
            return Err(Error::Configuration(format!(
                "Invalid EMA alpha: {:.3} (must be in range 0.0 < α < 1.0)",
                self.dc_bias_alpha
            )));
        }

        Ok(())
    }
}

/// Available high-pass filter orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HighpassOrder {
    /// Single biquad (2nd-order Butterworth)
    SecondOrder,
    /// Cascaded biquads (4th-order Butterworth)
    #[default]
    FourthOrder,
}

impl HighpassOrder {
    #[must_use]
    fn stage_count(self) -> usize {
        match self {
            Self::SecondOrder => 1,
            Self::FourthOrder => 2,
        }
    }
}

/// Optional VAD context for intelligent DC bias updates.
///
/// When provided, DC bias is only updated during silence periods,
/// avoiding speech distortion. This leverages the VAD engine
/// for progressive enhancement without tight coupling.
///
/// # Example
///
/// ```rust,no_run
/// use speech_prep::preprocessing::{DcHighPassFilter, PreprocessingConfig, VadContext};
///
/// # fn main() -> speech_prep::error::Result<()> {
/// let mut filter = DcHighPassFilter::new(PreprocessingConfig::default())?;
/// let samples = vec![0.0; 1600];
///
/// // Without VAD (always update DC bias)
/// let output1 = filter.process(&samples, None)?;
///
/// // With VAD (update only during silence)
/// let vad_ctx = VadContext { is_silence: true };
/// let output2 = filter.process(&samples, Some(&vad_ctx))?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct VadContext {
    /// True if the current audio is classified as silence.
    ///
    /// When true, DC bias tracking is updated.
    /// When false, DC bias is frozen (preserves speech quality).
    pub is_silence: bool,
}

/// DC offset removal and high-pass filtering with streaming state.
///
/// Implements the DC offset removal specification:
/// - Removes DC offset using exponential moving average (EMA)
/// - Applies cascaded Butterworth high-pass filtering (defaults to 4th order @
///   80 Hz)
/// - Maintains filter state across chunks for streaming continuity
/// - Achieves <2ms latency target per 500ms chunk (8000 samples @ 16kHz)
///
/// # Performance
///
/// - **Target**: <2ms per 500ms chunk
/// - **Expected**: ~0.16ms (10x headroom)
/// - **Optimization**: Precomputed coefficients, preallocated buffers
///
/// # Example
///
/// ```rust,no_run
/// use speech_prep::preprocessing::{DcHighPassFilter, PreprocessingConfig};
///
/// # fn main() -> speech_prep::error::Result<()> {
/// let mut filter = DcHighPassFilter::new(PreprocessingConfig::default())?;
/// let audio_stream = vec![vec![0.0; 8000], vec![0.1; 8000]];
///
/// // Process streaming chunks with state continuity
/// for chunk in audio_stream {
///     let clean = filter.process(&chunk, None)?;
///     // No discontinuities at boundaries!
/// }
/// # Ok(())
/// # }
/// ```
#[allow(missing_copy_implementations)]
#[derive(Debug, Clone)]
pub struct DcHighPassFilter {
    config: PreprocessingConfig,
    coeffs: BiquadCoefficients,
    stages: Vec<BiquadState>,
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

impl BiquadState {
    #[inline]
    fn process(&mut self, coeffs: &BiquadCoefficients, input: f32) -> f32 {
        let acc = coeffs
            .b0
            .mul_add(input, coeffs.b1.mul_add(self.x1, coeffs.b2 * self.x2));
        let output = acc - coeffs.a1.mul_add(self.y1, coeffs.a2 * self.y2);

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    #[cfg(test)]
    fn is_reset(self) -> bool {
        self == Self::default()
    }
}

impl DcHighPassFilter {
    /// Create a new DC offset removal and high-pass filter.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters (cutoff frequency, sample rate,
    ///   EMA alpha)
    ///
    /// # Errors
    ///
    /// Returns `Error::Configuration` if configuration is invalid.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use speech_prep::preprocessing::{DcHighPassFilter, PreprocessingConfig};
    ///
    /// let config = PreprocessingConfig {
    ///     highpass_cutoff_hz: 100.0, // More aggressive
    ///     ..Default::default()
    /// };
    /// let filter = DcHighPassFilter::new(config)?;
    /// # Ok::<(), speech_prep::error::Error>(())
    /// ```
    pub fn new(config: PreprocessingConfig) -> Result<Self> {
        config.validate()?;

        let (b0, b1, b2, a1, a2) = compute_butterworth_highpass_coefficients(
            config.highpass_cutoff_hz,
            config.sample_rate_hz,
        )?;

        let coeffs = BiquadCoefficients { b0, b1, b2, a1, a2 };
        let stage_count = config.highpass_order.stage_count();
        let stages = vec![BiquadState::default(); stage_count];

        Ok(Self {
            config,
            coeffs,
            stages,
            dc_bias: 0.0,
        })
    }

    /// Process audio samples with DC removal and high-pass filtering.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input audio samples (typically 500ms chunk = 8000 samples
    ///   @ 16kHz)
    /// * `vad_context` - Optional VAD state for intelligent DC bias updates
    ///
    /// # Returns
    ///
    /// Processed audio with DC removed and low frequencies attenuated.
    ///
    /// # Performance
    ///
    /// - Expected: ~0.16ms for 8000 samples (10x better than <2ms target)
    /// - Complexity: O(n) where n = `samples.len()`
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use speech_prep::preprocessing::{DcHighPassFilter, PreprocessingConfig, VadContext};
    ///
    /// let mut filter = DcHighPassFilter::new(PreprocessingConfig::default())?;
    ///
    /// // Chunk 1
    /// let chunk1 = vec![0.1, 0.2, -0.1, 0.15];
    /// let output1 = filter.process(&chunk1, None)?;
    ///
    /// // Chunk 2 (state preserved from chunk1 - no discontinuity!)
    /// let chunk2 = vec![0.2, 0.1, 0.3, 0.0];
    /// let output2 = filter.process(&chunk2, None)?;
    /// # Ok::<(), speech_prep::error::Error>(())
    /// ```
    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn process(
        &mut self,
        samples: &[f32],
        vad_context: Option<&VadContext>,
    ) -> Result<Vec<f32>> {
        let processing_start = AudioInstant::now();

        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let should_update_bias = vad_context.is_none_or(|ctx| ctx.is_silence);
        if self.config.enable_dc_removal && should_update_bias {
            self.update_dc_bias(samples);
        }

        let output = self.process_samples(samples);

        let elapsed = elapsed_duration(processing_start);
        let latency_ms = elapsed.as_secs_f64() * 1000.0;
        self.record_performance_metrics(samples.len(), latency_ms);

        Ok(output)
    }

    #[inline]
    fn process_samples(&mut self, samples: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(samples.len());

        for &sample in samples {
            let mut next = if self.config.enable_dc_removal {
                sample - self.dc_bias
            } else {
                sample
            };

            if self.config.enable_highpass {
                for stage in &mut self.stages {
                    next = stage.process(&self.coeffs, next);
                }
            }

            output.push(next);
        }

        output
    }

    fn record_performance_metrics(&self, sample_count: usize, latency_ms: f64) {
        if sample_count < 8000 {
            return;
        }

        if latency_ms > 2.0 {
            warn!(
                target: "audio.preprocess.highpass",
                latency_ms,
                samples = sample_count,
                cutoff_hz = self.config.highpass_cutoff_hz,
                order = ?self.config.highpass_order,
                "high-pass latency exceeded target"
            );
        }

        info!(
            target: "audio.preprocess.highpass",
            dc_bias = self.dc_bias,
            latency_ms,
            samples = sample_count,
            cutoff_hz = self.config.highpass_cutoff_hz,
            order = ?self.config.highpass_order,
            "audio preprocess high-pass metrics"
        );
    }

    /// Reset filter state for new audio stream.
    ///
    /// Clears filter history (x1, x2, y1, y2) and DC bias estimate.
    /// Use this when starting a new, independent audio stream.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use speech_prep::preprocessing::{DcHighPassFilter, PreprocessingConfig};
    ///
    /// # fn main() -> speech_prep::error::Result<()> {
    /// let mut filter = DcHighPassFilter::new(PreprocessingConfig::default())?;
    /// let audio_stream_1 = vec![0.0; 8000];
    /// let audio_stream_2 = vec![0.2; 8000];
    ///
    /// // Process stream 1
    /// filter.process(&audio_stream_1, None)?;
    ///
    /// // Switch to unrelated stream 2 - reset state
    /// filter.reset();
    /// filter.process(&audio_stream_2, None)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn reset(&mut self) {
        for stage in &mut self.stages {
            stage.reset();
        }
        self.dc_bias = 0.0;
    }

    /// Get current DC bias estimate.
    ///
    /// Useful for debugging or observability.
    #[must_use]
    pub fn dc_bias(&self) -> f32 {
        self.dc_bias
    }

    /// Get current configuration.
    #[must_use]
    pub fn config(&self) -> &PreprocessingConfig {
        &self.config
    }

    fn update_dc_bias(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }

        let sum: f32 = samples.iter().sum();
        let current_mean = sum / samples.len() as f32;

        let alpha = self.config.dc_bias_alpha;
        self.dc_bias = alpha.mul_add(self.dc_bias, (1.0 - alpha) * current_mean);
    }
}

/// Compute 2nd-order Butterworth high-pass filter coefficients.
///
/// Implements the standard biquad coefficient formulas for Butterworth
/// response.
///
/// # Arguments
///
/// * `cutoff_hz` - Cutoff frequency in Hz (e.g., 80.0)
/// * `sample_rate_hz` - Sample rate in Hz (e.g., 16000)
///
/// # Returns
///
/// Tuple of normalized coefficients: (b0, b1, b2, a1, a2)
/// where the transfer function is H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ +
/// a2·z⁻²)
///
/// # Errors
///
/// Returns error if coefficients cannot be computed (invalid parameters).
fn compute_butterworth_highpass_coefficients(
    cutoff_hz: f32,
    sample_rate_hz: u32,
) -> Result<(f32, f32, f32, f32, f32)> {
    use std::f32::consts::PI;

    let w0 = 2.0 * PI * cutoff_hz / sample_rate_hz as f32;
    let q = 0.707; // 1/sqrt(2) — Butterworth
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();

    let b0_unnorm = f32::midpoint(1.0, cos_w0);
    let b1_unnorm = -(1.0 + cos_w0);
    let b2_unnorm = f32::midpoint(1.0, cos_w0);
    let a0 = 1.0 + alpha;
    let a1_unnorm = -2.0 * cos_w0;
    let a2_unnorm = 1.0 - alpha;

    let b0 = b0_unnorm / a0;
    let b1 = b1_unnorm / a0;
    let b2 = b2_unnorm / a0;
    let a1 = a1_unnorm / a0;
    let a2 = a2_unnorm / a0;

    if !b0.is_finite() || !b1.is_finite() || !b2.is_finite() || !a1.is_finite() || !a2.is_finite() {
        return Err(Error::Processing(format!(
            "Invalid filter coefficients for fc={cutoff_hz:.1}Hz, fs={sample_rate_hz}: \
                 b0={b0:.6}, b1={b1:.6}, b2={b2:.6}, a1={a1:.6}, a2={a2:.6}"
        )));
    }

    Ok((b0, b1, b2, a1, a2))
}

fn elapsed_duration(start: AudioInstant) -> AudioDuration {
    AudioInstant::now().duration_since(start)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test helper type
    type TestResult<T> = std::result::Result<T, String>;

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
            .process(&speech_samples, Some(&speech_ctx))
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
            .process(&silence_samples, Some(&silence_ctx))
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
            filter.stages.iter().copied().any(|stage| !stage.is_reset()),
            "Filter stages should accumulate state after processing"
        );

        // Reset
        filter.reset();

        // Verify state cleared
        assert_eq!(filter.dc_bias(), 0.0, "DC bias should be zero after reset");
        assert!(
            filter.stages.iter().copied().all(BiquadState::is_reset),
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
