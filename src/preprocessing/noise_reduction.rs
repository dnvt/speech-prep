//! Stationary noise reduction by spectral subtraction.
//!
//! The reducer runs a short-time Fourier transform (STFT), learns an average
//! noise magnitude spectrum from frames marked as silence, subtracts it from
//! every frame, and resynthesizes with the original phase.
//!
//! This works for steady background noise such as fans, HVAC, hum, or hiss.
//! It does not help with non-stationary interference (other voices, music,
//! clicks) and does not remove reverb. Aggressive settings produce "musical
//! noise"; the spectral floor limits that.
//!
//! The noise profile is only learned when the caller passes a
//! [`VadContext`] with `is_silence: true`. Without that, the reducer has no
//! noise estimate and passes audio through almost unchanged.

use std::f32::consts::PI;
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use super::VadContext;
use crate::error::{Error, Result};

/// Configuration for [`NoiseReducer`].
#[derive(Debug, Clone)]
pub struct NoiseReductionConfig {
    /// Sample rate in Hz, 8000-48000. Default: 16000.
    pub sample_rate_hz: u32,

    /// STFT window length in milliseconds, 10-50. Default: 25.
    pub window_ms: f32,

    /// STFT hop in milliseconds; must be shorter than the window.
    /// Default: 10.
    pub hop_ms: f32,

    /// Over-subtraction factor (alpha), 1.0-3.0. Default: 2.0.
    ///
    /// Higher values remove more noise and introduce more artifacts.
    pub oversubtraction_factor: f32,

    /// Spectral floor (beta) as a fraction of the noise estimate,
    /// 0.001-0.1. Default: 0.02.
    pub spectral_floor: f32,

    /// Smoothing factor for the noise profile, in `[0.9, 1.0)`.
    /// Default: 0.98. Higher values adapt more slowly.
    pub noise_smoothing: f32,

    /// Set to `false` to pass audio through unchanged. Default: true.
    pub enable: bool,
}

impl Default for NoiseReductionConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 16_000,
            window_ms: 25.0,
            hop_ms: 10.0,
            oversubtraction_factor: 2.0,
            spectral_floor: 0.02,
            noise_smoothing: 0.98,
            enable: true,
        }
    }
}

impl NoiseReductionConfig {
    /// Check that every field is within its documented range.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] naming the first invalid field.
    pub fn validate(&self) -> Result<()> {
        if !(8000..=48_000).contains(&self.sample_rate_hz) {
            return Err(Error::Configuration(format!(
                "sample_rate_hz {} must be in 8000-48000",
                self.sample_rate_hz
            )));
        }

        if !(10.0..=50.0).contains(&self.window_ms) {
            return Err(Error::Configuration(format!(
                "window_ms {:.1} must be in 10-50",
                self.window_ms
            )));
        }

        if self.hop_ms <= 0.0 || self.hop_ms >= self.window_ms {
            return Err(Error::Configuration(format!(
                "hop_ms {:.1} must be positive and shorter than window_ms {:.1}",
                self.hop_ms, self.window_ms
            )));
        }

        if !(1.0..=3.0).contains(&self.oversubtraction_factor) {
            return Err(Error::Configuration(format!(
                "oversubtraction_factor {:.2} must be in 1.0-3.0",
                self.oversubtraction_factor
            )));
        }

        if !(0.001..=0.1).contains(&self.spectral_floor) {
            return Err(Error::Configuration(format!(
                "spectral_floor {:.3} must be in 0.001-0.1",
                self.spectral_floor
            )));
        }

        if !(0.9..1.0).contains(&self.noise_smoothing) {
            return Err(Error::Configuration(format!(
                "noise_smoothing {:.3} must be in [0.9, 1.0)",
                self.noise_smoothing
            )));
        }

        Ok(())
    }

    /// STFT window length in samples.
    #[must_use]
    pub fn frame_length(&self) -> usize {
        ((self.window_ms / 1000.0) * self.sample_rate_hz as f32).round() as usize
    }

    /// STFT hop length in samples.
    #[must_use]
    pub fn hop_length(&self) -> usize {
        ((self.hop_ms / 1000.0) * self.sample_rate_hz as f32).round() as usize
    }

    /// FFT size: the next power of two at or above the frame length.
    #[must_use]
    pub fn fft_size(&self) -> usize {
        self.frame_length().next_power_of_two()
    }
}

/// Spectral-subtraction noise reducer with a VAD-gated noise profile.
///
/// Each call to [`reduce`](Self::reduce) returns exactly as many samples as it
/// was given. The noise profile persists across calls; call
/// [`reset`](Self::reset) when switching to an unrelated stream.
///
/// ```
/// use speech_prep::preprocessing::{NoiseReducer, NoiseReductionConfig, VadContext};
///
/// # fn main() -> speech_prep::Result<()> {
/// let mut reducer = NoiseReducer::new(NoiseReductionConfig::default())?;
///
/// // Learn the noise floor from a silent stretch...
/// let room_tone = vec![0.001; 8000];
/// reducer.reduce(&room_tone, Some(VadContext { is_silence: true }))?;
///
/// // ...then clean speech.
/// let speech = vec![0.1; 8000];
/// let cleaned = reducer.reduce(&speech, Some(VadContext { is_silence: false }))?;
/// assert_eq!(cleaned.len(), speech.len());
/// # Ok(())
/// # }
/// ```
pub struct NoiseReducer {
    config: NoiseReductionConfig,
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    noise_profile: Vec<f32>,
    noise_initialized: bool,
}

impl std::fmt::Debug for NoiseReducer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NoiseReducer")
            .field("config", &self.config)
            .field("window_length", &self.window.len())
            .field("noise_profile_bins", &self.noise_profile.len())
            .field("noise_initialized", &self.noise_initialized)
            .finish_non_exhaustive()
    }
}

/// Initial noise magnitude per bin, before any silence has been observed.
const INITIAL_NOISE_MAGNITUDE: f32 = 1e-6;

/// Below this window-energy sum a sample is considered uncovered by the STFT
/// and is passed through instead of divided by a near-zero weight.
const MIN_WINDOW_WEIGHT: f32 = 1e-3;

impl NoiseReducer {
    /// Create a reducer from a validated configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] if `config` is invalid.
    pub fn new(config: NoiseReductionConfig) -> Result<Self> {
        config.validate()?;

        let fft_size = config.fft_size();
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(fft_size);
        let fft_inverse = planner.plan_fft_inverse(fft_size);

        Ok(Self {
            window: generate_hann_window(config.frame_length()),
            noise_profile: vec![INITIAL_NOISE_MAGNITUDE; fft_size / 2 + 1],
            noise_initialized: false,
            config,
            fft_forward,
            fft_inverse,
        })
    }

    /// Denoise one chunk of samples.
    ///
    /// If `vad_context` marks the chunk as silence, its full frames also
    /// update the noise profile before subtraction.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Processing`] if an FFT call fails.
    pub fn reduce(&mut self, samples: &[f32], vad_context: Option<VadContext>) -> Result<Vec<f32>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        if !self.config.enable {
            return Ok(samples.to_vec());
        }

        let frame_length = self.window.len();
        let hop_length = self.config.hop_length();
        let is_silence = vad_context.is_some_and(|ctx| ctx.is_silence);

        // Pad by one frame on each side so every input sample is covered by
        // the same number of overlapping windows. Without this the first and
        // last frame of each chunk fade towards zero.
        let pad = frame_length;
        let padded_len = samples.len() + 2 * pad;
        let mut output = vec![0.0f32; padded_len];
        let mut weight = vec![0.0f32; padded_len];

        let mut frame = vec![0.0f32; frame_length];
        let mut pos = 0;
        while pos + frame_length <= padded_len {
            for (i, slot) in frame.iter_mut().enumerate() {
                let idx = pos + i;
                *slot = if (pad..pad + samples.len()).contains(&idx) {
                    samples.get(idx - pad).copied().unwrap_or(0.0)
                } else {
                    0.0
                };
            }

            let fully_inside = pos >= pad && pos + frame_length <= pad + samples.len();
            let processed = self.process_frame(&frame, is_silence && fully_inside)?;

            for (i, (&s, &w)) in processed.iter().zip(&self.window).enumerate() {
                if let (Some(out), Some(wsum)) = (output.get_mut(pos + i), weight.get_mut(pos + i))
                {
                    *out += s;
                    *wsum = w.mul_add(w, *wsum);
                }
            }

            pos += hop_length;
        }

        let result = samples
            .iter()
            .enumerate()
            .map(|(i, &original)| {
                let w = weight.get(pad + i).copied().unwrap_or(0.0);
                let y = output.get(pad + i).copied().unwrap_or(0.0);
                if w > MIN_WINDOW_WEIGHT {
                    y / w
                } else {
                    original
                }
            })
            .collect();

        Ok(result)
    }

    /// Forget the learned noise profile.
    pub fn reset(&mut self) {
        self.noise_profile.fill(INITIAL_NOISE_MAGNITUDE);
        self.noise_initialized = false;
    }

    /// Mean of the noise magnitude spectrum, for diagnostics.
    #[must_use]
    pub fn noise_floor(&self) -> f32 {
        if self.noise_profile.is_empty() {
            return 0.0;
        }
        self.noise_profile.iter().sum::<f32>() / self.noise_profile.len() as f32
    }

    /// The configuration this reducer was built with.
    #[must_use]
    pub fn config(&self) -> &NoiseReductionConfig {
        &self.config
    }

    /// Analysis window, FFT, subtraction, inverse FFT, synthesis window.
    fn process_frame(&mut self, frame: &[f32], update_noise: bool) -> Result<Vec<f32>> {
        let mut input = self.fft_forward.make_input_vec();
        for ((dst, &s), &w) in input.iter_mut().zip(frame).zip(&self.window) {
            *dst = s * w;
        }

        let mut spectrum = self.fft_forward.make_output_vec();
        self.fft_forward
            .process(&mut input, &mut spectrum)
            .map_err(|e| Error::Processing(format!("forward FFT failed: {e}")))?;

        let magnitudes: Vec<f32> = spectrum.iter().map(|c| c.norm()).collect();
        if update_noise {
            self.update_noise_profile(&magnitudes);
        }

        let alpha = self.config.oversubtraction_factor;
        let beta = self.config.spectral_floor;
        let last_bin = spectrum.len() - 1;
        for (k, ((bin, &mag), &noise)) in spectrum
            .iter_mut()
            .zip(&magnitudes)
            .zip(&self.noise_profile)
            .enumerate()
        {
            let cleaned = alpha.mul_add(-noise, mag).max(beta * noise);
            *bin = if k == 0 || k == last_bin {
                // DC and Nyquist bins of a real signal's spectrum are real.
                Complex::new(cleaned.copysign(bin.re), 0.0)
            } else {
                Complex::from_polar(cleaned, bin.arg())
            };
        }

        let mut time = self.fft_inverse.make_output_vec();
        self.fft_inverse
            .process(&mut spectrum, &mut time)
            .map_err(|e| Error::Processing(format!("inverse FFT failed: {e}")))?;

        let scale = 1.0 / self.config.fft_size() as f32;
        Ok(time
            .iter()
            .zip(&self.window)
            .map(|(&s, &w)| s * scale * w)
            .collect())
    }

    fn update_noise_profile(&mut self, magnitudes: &[f32]) {
        if self.noise_initialized {
            let alpha = self.config.noise_smoothing;
            for (noise, &current) in self.noise_profile.iter_mut().zip(magnitudes) {
                *noise = alpha.mul_add(*noise, (1.0 - alpha) * current);
            }
        } else {
            self.noise_profile.copy_from_slice(magnitudes);
            self.noise_initialized = true;
        }
    }
}

/// Symmetric Hann window: `w[n] = 0.5 - 0.5 * cos(2πn / (N - 1))`.
fn generate_hann_window(length: usize) -> Vec<f32> {
    if length == 0 {
        return Vec::new();
    }
    if length == 1 {
        return vec![1.0];
    }

    let denom = (length - 1) as f32;
    (0..length)
        .map(|n| {
            let angle = 2.0 * PI * n as f32 / denom;
            0.5f32.mul_add(-angle.cos(), 0.5)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult<T> = std::result::Result<T, String>;

    #[test]
    fn test_configuration_validation() -> TestResult<()> {
        // Valid configuration
        let valid = NoiseReductionConfig::default();
        assert!(valid.validate().is_ok());

        // Invalid sample rate
        let invalid_sr = NoiseReductionConfig {
            sample_rate_hz: 5000,
            ..Default::default()
        };
        assert!(invalid_sr.validate().is_err());

        // Invalid window size
        let invalid_window = NoiseReductionConfig {
            window_ms: 100.0,
            ..Default::default()
        };
        assert!(invalid_window.validate().is_err());

        // Hop >= window
        let invalid_hop = NoiseReductionConfig {
            hop_ms: 30.0,
            window_ms: 25.0,
            ..Default::default()
        };
        assert!(invalid_hop.validate().is_err());

        // Invalid oversubtraction
        let invalid_alpha = NoiseReductionConfig {
            oversubtraction_factor: 5.0,
            ..Default::default()
        };
        assert!(invalid_alpha.validate().is_err());

        // Invalid spectral floor
        let invalid_beta = NoiseReductionConfig {
            spectral_floor: 0.5,
            ..Default::default()
        };
        assert!(invalid_beta.validate().is_err());

        // Invalid noise smoothing
        let invalid_smoothing = NoiseReductionConfig {
            noise_smoothing: 1.0,
            ..Default::default()
        };
        assert!(invalid_smoothing.validate().is_err());

        Ok(())
    }

    #[test]
    fn test_hann_window_properties() {
        // Empty window
        let window_0 = generate_hann_window(0);
        assert!(window_0.is_empty());

        // Single element
        let window_1 = generate_hann_window(1);
        assert_eq!(window_1.len(), 1);
        assert!((window_1[0] - 1.0).abs() < 1e-6);

        // Check Hann window properties
        let window = generate_hann_window(100);
        assert_eq!(window.len(), 100);

        // First and last samples should be near zero
        assert!(window[0].abs() < 1e-6);
        assert!(window[99].abs() < 1e-6);

        // Middle sample should be near 1.0
        assert!((window[50] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_noise_reducer_creation() -> TestResult<()> {
        let config = NoiseReductionConfig::default();
        let reducer = NoiseReducer::new(config).map_err(|e| e.to_string())?;

        // Check initialization
        assert_eq!(reducer.config().sample_rate_hz, 16000);
        assert!(reducer.noise_floor() > 0.0); // Initial estimate

        Ok(())
    }

    #[test]
    fn test_empty_input() -> TestResult<()> {
        let config = NoiseReductionConfig::default();
        let mut reducer = NoiseReducer::new(config).map_err(|e| e.to_string())?;

        let output = reducer.reduce(&[], None).map_err(|e| e.to_string())?;
        assert!(output.is_empty());

        Ok(())
    }

    #[test]
    fn test_bypass_mode() -> TestResult<()> {
        let config = NoiseReductionConfig {
            enable: false,
            ..Default::default()
        };
        let mut reducer = NoiseReducer::new(config).map_err(|e| e.to_string())?;

        let input = vec![0.1, 0.2, 0.3, 0.4];
        let output = reducer.reduce(&input, None).map_err(|e| e.to_string())?;

        // Bypass mode should return input unchanged
        assert_eq!(output, input);

        Ok(())
    }

    #[test]
    fn test_noise_profile_update() -> TestResult<()> {
        let config = NoiseReductionConfig::default();
        let mut reducer = NoiseReducer::new(config).map_err(|e| e.to_string())?;

        // Process silence to build noise profile
        let silence = vec![0.01; 8000]; // Low-level noise
        let vad_silence = VadContext { is_silence: true };

        let initial_noise = reducer.noise_floor();

        // Process multiple chunks to converge
        for _ in 0..5 {
            let _ = reducer
                .reduce(&silence, Some(vad_silence))
                .map_err(|e| e.to_string())?;
        }

        let converged_noise = reducer.noise_floor();

        // Noise floor should increase from initial estimate
        assert!(
            converged_noise > initial_noise,
            "Noise floor should adapt: initial={:.6}, converged={:.6}",
            initial_noise,
            converged_noise
        );

        Ok(())
    }

    #[test]
    fn test_vad_informed_noise_update() -> TestResult<()> {
        let config = NoiseReductionConfig::default();
        let mut reducer = NoiseReducer::new(config).map_err(|e| e.to_string())?;

        // Initialize with silence
        let silence = vec![0.01; 8000];
        let vad_silence = VadContext { is_silence: true };
        for _ in 0..5 {
            let _ = reducer
                .reduce(&silence, Some(vad_silence))
                .map_err(|e| e.to_string())?;
        }

        let noise_after_silence = reducer.noise_floor();

        // Process "speech" (should NOT update noise profile)
        let speech = vec![0.5; 8000];
        let vad_speech = VadContext { is_silence: false };
        let _ = reducer
            .reduce(&speech, Some(vad_speech))
            .map_err(|e| e.to_string())?;

        let noise_after_speech = reducer.noise_floor();

        // Noise profile should remain stable during speech
        let diff = (noise_after_speech - noise_after_silence).abs();
        assert!(
            diff < noise_after_silence * 0.01,
            "Noise profile changed during speech: {:.6} -> {:.6}",
            noise_after_silence,
            noise_after_speech
        );

        Ok(())
    }

    #[test]
    fn test_reset_clears_state() -> TestResult<()> {
        let config = NoiseReductionConfig::default();
        let mut reducer = NoiseReducer::new(config).map_err(|e| e.to_string())?;

        // Process some audio
        let samples = vec![0.1; 8000];
        let vad = VadContext { is_silence: true };
        let _ = reducer
            .reduce(&samples, Some(vad))
            .map_err(|e| e.to_string())?;

        let noise_before = reducer.noise_floor();
        assert!(noise_before > 1e-5, "Noise profile should be updated");

        // Reset
        reducer.reset();

        let noise_after = reducer.noise_floor();
        assert!(
            noise_after < 1e-5,
            "Noise profile should be reset to initial value"
        );

        Ok(())
    }

    // Helper: Generate sine wave
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

    /// Deterministic uniform noise in `[-1, 1)` (xorshift32), so the SNR
    /// tests give the same result on every run.
    fn uniform_noise(seed: u32) -> impl FnMut() -> f32 {
        let mut state = seed.max(1);
        move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32).mul_add(2.0, -1.0)
        }
    }

    // Helper: Add white noise to signal
    fn add_white_noise(signal: &[f32], noise_amplitude: f32) -> Vec<f32> {
        let mut noise = uniform_noise(0x9E37_79B9);
        signal
            .iter()
            .map(|&s| noise_amplitude.mul_add(noise(), s))
            .collect()
    }
    fn add_low_freq_hum(
        signal: &[f32],
        sample_rate: u32,
        frequency: f32,
        amplitude: f32,
    ) -> Vec<f32> {
        signal
            .iter()
            .enumerate()
            .map(|(i, &sample)| {
                let t = i as f32 / sample_rate as f32;
                let hum = (2.0 * PI * frequency * t).sin() * amplitude;
                sample + hum
            })
            .collect()
    }

    // Helper: Add café-like ambient noise (stationary broadband noise)
    // Simulates background café noise: HVAC, dishes, distant ambient chatter.
    // Uses white noise as a proxy for band-limited stationary noise (100-3000 Hz
    // typical). NOTE: Spectral subtraction works for STATIONARY noise, not
    // speech-like babble.
    fn add_cafe_noise(signal: &[f32], _sample_rate: u32, amplitude: f32) -> Vec<f32> {
        let mut noise = uniform_noise(0x85EB_CA6B);
        signal
            .iter()
            .map(|&s| amplitude.mul_add(noise(), s))
            .collect()
    }

    // Helper: Calculate SNR
    fn calculate_snr(clean: &[f32], noisy: &[f32]) -> f32 {
        if clean.len() != noisy.len() {
            return 0.0;
        }

        let signal_power: f32 = clean.iter().map(|&x| x * x).sum();
        let noise: Vec<f32> = clean
            .iter()
            .zip(noisy.iter())
            .map(|(&c, &n)| n - c)
            .collect();
        let noise_power: f32 = noise.iter().map(|&x| x * x).sum();

        if noise_power < 1e-10 {
            return 100.0; // Very high SNR
        }

        10.0 * (signal_power / noise_power).log10()
    }

    #[test]
    fn test_snr_improvement_white_noise() -> TestResult<()> {
        let config = NoiseReductionConfig::default();
        let mut reducer = NoiseReducer::new(config).map_err(|e| e.to_string())?;

        // Generate clean speech signal (440 Hz sine wave)
        let clean_speech = generate_sine_wave(440.0, 16000, 1.0, 0.5);

        // Add white noise (creating ~5 dB input SNR)
        let noisy_speech = add_white_noise(&clean_speech, 0.3);

        let snr_before = calculate_snr(&clean_speech, &noisy_speech);

        // Initialize noise profile with pure noise
        // Training: 10 iterations ensures EMA convergence (α=0.98 requires ~50 samples
        // for 95% convergence)
        let pure_noise = add_white_noise(&vec![0.0; 8000], 0.3);
        let vad_silence = VadContext { is_silence: true };
        for _ in 0..10 {
            let _ = reducer
                .reduce(&pure_noise, Some(vad_silence))
                .map_err(|e| e.to_string())?;
        }

        // Apply noise reduction to noisy speech
        let vad_speech = VadContext { is_silence: false };
        let denoised = reducer
            .reduce(&noisy_speech, Some(vad_speech))
            .map_err(|e| e.to_string())?;

        let snr_after = calculate_snr(&clean_speech, &denoised);
        let improvement = snr_after - snr_before;

        // Success criterion: ≥6 dB improvement
        assert!(
            improvement >= 6.0,
            "SNR improvement {:.1} dB < 6 dB target",
            improvement
        );

        Ok(())
    }

    #[test]
    fn test_snr_improvement_low_freq_hum() -> TestResult<()> {
        let config = NoiseReductionConfig::default();
        let mut reducer = NoiseReducer::new(config).map_err(|e| e.to_string())?;

        // Generate 440 Hz speech with 60 Hz HVAC hum (common electrical interference)
        let clean = generate_sine_wave(440.0, 16000, 1.0, 0.4);
        let noisy = add_low_freq_hum(&clean, 16000, 60.0, 0.3);
        let snr_before = calculate_snr(&clean, &noisy);

        // Train on pure 60 Hz hum
        // Training: 6 iterations sufficient for tonal noise (faster convergence than
        // broadband)
        let hum_only = add_low_freq_hum(&vec![0.0; 8000], 16000, 60.0, 0.3);
        let vad = VadContext { is_silence: true };
        for _ in 0..6 {
            let _ = reducer
                .reduce(&hum_only, Some(vad))
                .map_err(|e| e.to_string())?;
        }

        let vad_speech = VadContext { is_silence: false };
        let denoised = reducer
            .reduce(&noisy, Some(vad_speech))
            .map_err(|e| e.to_string())?;
        let snr_after = calculate_snr(&clean, &denoised);
        let improvement = snr_after - snr_before;
        assert!(
            improvement >= 6.0,
            "Hum SNR improvement {:.1} dB < 6 dB target",
            improvement
        );

        Ok(())
    }

    #[test]
    fn test_snr_improvement_cafe_ambient() -> TestResult<()> {
        // Use default config - stationary noise doesn't need aggressive oversubtraction
        let config = NoiseReductionConfig::default();
        let mut reducer = NoiseReducer::new(config).map_err(|e| e.to_string())?;

        // Generate clean speech signal
        let clean = generate_sine_wave(220.0, 16000, 1.0, 0.4);

        // Add stationary café ambient noise (HVAC, dishes, background chatter)
        let noisy = add_cafe_noise(&clean, 16000, 0.25);
        let snr_before = calculate_snr(&clean, &noisy);

        // Train noise profile on café ambient noise during "silence"
        // Training: 10 iterations for broadband stationary noise (white noise requires
        // more samples than tonal) Noise amplitude 0.25 creates ~5-6 dB input
        // SNR, realistic for café environment
        let cafe_only = add_cafe_noise(&vec![0.0; 8000], 16000, 0.25);
        let vad = VadContext { is_silence: true };
        for _ in 0..10 {
            let _ = reducer
                .reduce(&cafe_only, Some(vad))
                .map_err(|e| e.to_string())?;
        }

        // Apply noise reduction to noisy speech
        let vad_speech = VadContext { is_silence: false };
        let denoised = reducer
            .reduce(&noisy, Some(vad_speech))
            .map_err(|e| e.to_string())?;

        let snr_after = calculate_snr(&clean, &denoised);
        let improvement = snr_after - snr_before;

        assert!(
            improvement >= 6.0,
            "Café ambient SNR improvement {:.1} dB < 6 dB target",
            improvement
        );

        Ok(())
    }

    #[test]
    fn test_trailing_partial_frame_preserved() -> TestResult<()> {
        let config = NoiseReductionConfig::default();
        let mut reducer = NoiseReducer::new(config).map_err(|e| e.to_string())?;

        // Prime noise profile with a silence chunk so spectral subtraction behaves
        // normally.
        let silence = vec![0.0; 8000];
        let vad_silence = VadContext { is_silence: true };
        let _ = reducer
            .reduce(&silence, Some(vad_silence))
            .map_err(|e| e.to_string())?;

        // Speech chunk length intentionally not divisible by hop size (adds 80-sample
        // tail).
        let speech_len = 8080;
        let speech: Vec<f32> = (0..speech_len)
            .map(|i| {
                let phase = (i as f32 / speech_len as f32) * 20.0;
                phase.sin()
            })
            .collect();

        let vad_speech = VadContext { is_silence: false };
        let output = reducer
            .reduce(&speech, Some(vad_speech))
            .map_err(|e| e.to_string())?;

        assert_eq!(
            output.len(),
            speech_len,
            "Output length should match input length"
        );

        let tail = &output[speech_len - 80..];
        let tail_energy: f32 = tail.iter().map(|sample| sample.abs()).sum();
        assert!(
            tail_energy > 1e-3,
            "Trailing samples should retain energy, got tail_energy={tail_energy}"
        );

        Ok(())
    }

    #[test]
    fn test_missing_vad_context_does_not_update_noise_profile() -> TestResult<()> {
        let config = NoiseReductionConfig::default();
        let mut reducer = NoiseReducer::new(config).map_err(|e| e.to_string())?;

        // Prime noise profile with explicit silence (non-zero noise so baseline is
        // measurable).
        let ambient_noise = vec![0.05f32; 8000];
        let vad_silence = VadContext { is_silence: true };
        reducer
            .reduce(&ambient_noise, Some(vad_silence))
            .map_err(|e| e.to_string())?;
        let baseline_floor = reducer.noise_floor();

        // Process speech without VAD context; noise profile should remain unchanged.
        let speech = vec![0.2f32; 8000];
        let output = reducer.reduce(&speech, None).map_err(|e| e.to_string())?;
        let updated_floor = reducer.noise_floor();

        let floor_delta = (updated_floor - baseline_floor).abs();
        assert!(
            floor_delta < baseline_floor.max(1e-6) * 0.01,
            "Noise floor changed when VAD context missing: baseline={baseline_floor}, \
             updated={updated_floor}"
        );

        let output_rms =
            (output.iter().map(|sample| sample * sample).sum::<f32>() / output.len() as f32).sqrt();
        assert!(
            output_rms > 0.08,
            "Speech energy collapsed without VAD context (rms={output_rms})"
        );

        Ok(())
    }

    #[test]
    fn test_passthrough_without_noise_profile() -> TestResult<()> {
        // With no silence observed the noise estimate is ~0, so the STFT
        // round trip should reconstruct the input, including chunk edges.
        let mut reducer =
            NoiseReducer::new(NoiseReductionConfig::default()).map_err(|e| e.to_string())?;
        let input = generate_sine_wave(300.0, 16000, 1.0, 0.5);

        let mut output = Vec::new();
        for chunk in input.chunks(8000) {
            let speech = Some(VadContext { is_silence: false });
            output.extend(reducer.reduce(chunk, speech).map_err(|e| e.to_string())?);
        }

        assert_eq!(output.len(), input.len());
        let max_err = input
            .iter()
            .zip(&output)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "max reconstruction error {max_err}");
        Ok(())
    }
}
