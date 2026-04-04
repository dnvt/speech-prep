//! VAD configuration and validation.

use crate::error::{Error, Result};
use crate::time::{AudioDuration, AudioTimestamp};

/// Number of nanoseconds in one second, used for time conversion.
const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// Configuration for the voice activity detector.
///
/// # Performance Characteristics
///
/// - **Latency**: Typically <2ms per 20ms frame (10% overhead)
/// - **Memory**: ~10KB per detector instance (FFT buffers + state)
/// - **Accuracy**: >95% speech detection on clean audio
///
/// # Configuration Guidelines
///
/// ## Quick Start (Use Defaults)
///
/// ```rust,no_run
/// use speech_prep::VadConfig;
///
/// let config = VadConfig::default(); // Optimized for 16kHz mono speech
/// ```
///
/// ## Advanced Tuning
///
/// **For Noisy Environments**: Increase `activation_margin` to 1.3-1.5
///
/// ```rust,no_run
/// # use speech_prep::VadConfig;
/// let config = VadConfig {
///     activation_margin: 1.4, // Require stronger signal
///     hangover_frames: 5,     // Longer trailing silence tolerance
///     ..VadConfig::default()
/// };
/// ```
///
/// **For Low-Latency Applications**: Reduce `frame_duration`
///
/// ```rust,no_run
/// # use speech_prep::VadConfig;
/// # use speech_prep::time::AudioDuration;
/// let config = VadConfig {
///     frame_duration: AudioDuration::from_millis(10), // 10ms frames
///     ..VadConfig::default()
/// };
/// ```
///
/// **For Soft/Quiet Speech**: Lower `activation_margin`
///
/// ```rust,no_run
/// # use speech_prep::VadConfig;
/// let config = VadConfig {
///     activation_margin: 1.05, // More sensitive
///     min_speech_frames: 2,    // Faster activation
///     ..VadConfig::default()
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    /// Expected audio sample rate in Hz.
    ///
    /// **Default**: 16000 (16kHz - optimal for speech)
    ///
    /// **Valid Range**: 8000-48000 Hz
    ///
    /// **Performance Impact**: Higher rates increase FFT computation cost.
    /// At 48kHz, expect ~3x slower processing vs 16kHz.
    ///
    /// **Recommendation**: Use 16kHz unless your audio pipeline requires
    /// otherwise.
    pub sample_rate: u32,

    /// Frame duration used for analysis.
    ///
    /// **Default**: 20ms (320 samples at 16kHz)
    ///
    /// **Valid Range**: 10-50ms
    ///
    /// **Trade-offs**:
    /// - Shorter (10ms): Lower latency, less robust to noise
    /// - Longer (50ms): Higher latency, more stable detection
    ///
    /// **Performance Impact**: 20ms frame = ~1.5ms processing time.
    /// Linear scaling: 10ms → ~0.75ms, 50ms → ~3.75ms.
    pub frame_duration: AudioDuration,

    /// Fractional overlap between adjacent frames.
    ///
    /// **Default**: 0.5 (50% overlap)
    ///
    /// **Valid Range**: [0.0, 1.0)
    ///
    /// **Effect**: Higher overlap increases temporal resolution but adds
    /// computation cost. 50% overlap means processing 2x frames for same audio
    /// duration.
    ///
    /// **Recommendation**: 0.5 for balanced accuracy/performance, 0.75 for
    /// critical applications requiring precise boundary detection.
    pub frame_overlap: f32,

    /// Smoothing factor for rolling energy baseline (exponential moving
    /// average).
    ///
    /// **Default**: 0.85 (85% history, 15% new observation)
    ///
    /// **Valid Range**: [0.0, 1.0)
    ///
    /// **Effect**: Controls adaptation speed to background noise changes.
    /// - Higher (0.9-0.95): Slower adaptation, stable in constant noise
    /// - Lower (0.7-0.8): Faster adaptation, handles dynamic noise
    ///
    /// **Half-Life**: At 0.85, baseline half-life ≈ 4.3 frames (86ms at
    /// 20ms/frame).
    pub energy_smoothing: f32,

    /// Smoothing factor for rolling spectral flux baseline.
    ///
    /// **Default**: 0.8 (80% history, 20% new observation)
    ///
    /// **Valid Range**: [0.0, 1.0)
    ///
    /// **Effect**: Controls adaptation to spectral change patterns.
    /// Flux typically more variable than energy, so slightly lower smoothing.
    ///
    /// **Half-Life**: At 0.8, baseline half-life ≈ 3.1 frames (62ms at
    /// 20ms/frame).
    pub flux_smoothing: f32,

    /// Minimum energy floor to prevent division by zero in normalization.
    ///
    /// **Default**: 1e-4 (0.0001)
    ///
    /// **Valid Range**: >0.0 (typically 1e-6 to 1e-3)
    ///
    /// **Effect**: Prevents numerical instability when audio is completely
    /// silent. Value is small enough to not affect real audio.
    pub energy_floor: f32,

    /// Minimum spectral flux floor to prevent division by zero.
    ///
    /// **Default**: 1e-4 (0.0001)
    ///
    /// **Valid Range**: >0.0 (typically 1e-6 to 1e-3)
    ///
    /// **Effect**: Prevents numerical instability in flux calculations.
    pub flux_floor: f32,

    /// Smoothing factor for the dynamic decision threshold.
    ///
    /// **Default**: 0.9 (90% history, 10% new)
    ///
    /// **Valid Range**: [0.0, 1.0)
    ///
    /// **Effect**: Controls how quickly the detector adapts its sensitivity.
    /// Higher values make the threshold more stable, preventing rapid
    /// oscillations in marginal cases.
    pub threshold_smoothing: f32,

    /// Multiplier applied to dynamic threshold to activate speech detection.
    ///
    /// **Default**: 1.1 (110% of baseline threshold)
    ///
    /// **Valid Range**: ≥1.0
    ///
    /// **Effect**: Creates hysteresis to prevent chattering at boundaries.
    /// - 1.05-1.1: High sensitivity (detects soft speech, more false positives)
    /// - 1.2-1.5: Low sensitivity (robust to noise, may miss quiet speech)
    ///
    /// **Recommendation**: Start with 1.1, increase if too many false
    /// activations.
    pub activation_margin: f32,

    /// Multiplier applied to dynamic threshold when releasing to silence.
    ///
    /// **Default**: 0.9 (90% of baseline threshold)
    ///
    /// **Valid Range**: >0.0, must be ≤ `activation_margin`
    ///
    /// **Effect**: Creates hysteresis to maintain speech state during brief
    /// pauses. Difference between activation and release margins prevents
    /// rapid toggling.
    ///
    /// **Typical Gap**: 0.1-0.3 between margins (e.g., activate=1.2,
    /// release=0.9).
    pub release_margin: f32,

    /// Initial baseline threshold before dynamic adaptation kicks in.
    ///
    /// **Default**: 0.4 (40% of normalized scale)
    ///
    /// **Valid Range**: 0.0-1.0
    ///
    /// **Effect**: Starting point for adaptive threshold. After 10-20 frames,
    /// adaptive algorithm takes over and this value becomes less relevant.
    ///
    /// **Recommendation**: Leave at default unless you know audio
    /// characteristics.
    pub base_threshold: f32,

    /// Weight applied to normalized energy when combining dual metrics.
    ///
    /// **Default**: 0.6 (60% energy, 40% flux)
    ///
    /// **Valid Range**: 0.0-1.0 (combined with `flux_weight` should sum to 1.0)
    ///
    /// **Effect**: Energy detects signal presence, flux detects spectral
    /// changes. Higher energy weight emphasizes volume-based detection.
    ///
    /// **Use Cases**:
    /// - 0.7-0.8: Emphasize loudness (good for clean recordings)
    /// - 0.5-0.6: Balanced (default, works well generally)
    /// - 0.3-0.4: Emphasize spectral change (noisy environments)
    pub energy_weight: f32,

    /// Weight applied to normalized spectral flux when combining metrics.
    ///
    /// **Default**: 0.4 (40% flux, 60% energy)
    ///
    /// **Valid Range**: 0.0-1.0 (combined with `energy_weight` should sum to
    /// 1.0)
    ///
    /// **Effect**: Flux is more robust to constant background noise but can
    /// be fooled by music or non-speech sounds with spectral variation.
    pub flux_weight: f32,

    /// Number of trailing silent frames retained at the end of a speech
    /// segment.
    ///
    /// **Default**: 3 frames (60ms at 20ms/frame)
    ///
    /// **Valid Range**: 0-10 frames (typically)
    ///
    /// **Effect**: Prevents premature cutoff of speech segments during brief
    /// pauses (e.g., between words). Too high causes long trailing silence.
    ///
    /// **Recommendation**:
    /// - 2-3: Normal speech (default)
    /// - 5-8: Slow/hesitant speech
    /// - 0-1: Real-time applications requiring minimal latency
    pub hangover_frames: usize,

    /// Minimum number of speech frames required to emit a segment.
    ///
    /// **Default**: 3 frames (60ms at 20ms/frame)
    ///
    /// **Valid Range**: 1-10 frames (typically)
    ///
    /// **Effect**: Filters out brief noise spikes mistaken for speech.
    /// Too high causes missed short utterances (e.g., "yes", "no").
    ///
    /// **Recommendation**:
    /// - 2-3: Balanced (default)
    /// - 1: Detect very short sounds
    /// - 5+: Only long speech segments
    pub min_speech_frames: usize,

    /// Absolute start time for the first sample processed by this detector.
    ///
    /// **Default**: `AudioTimestamp::EPOCH` (zero-based stream time)
    ///
    /// **Effect**: Used for timestamping detected speech segments. Set this
    /// to the origin you want segment timestamps to use, or leave it as
    /// `EPOCH` for timestamps relative to the start of processing.
    ///
    /// **Use Cases**:
    /// - Live streams: Set to a shared stream origin
    /// - Batch processing: Keep `EPOCH` or provide a known offset
    /// - Testing: Leave as `EPOCH` for deterministic timestamps
    pub stream_start_time: AudioTimestamp,

    /// Optional pre-emphasis coefficient applied before analysis (high-pass
    /// filter).
    ///
    /// **Default**: `Some(0.97)` (standard speech pre-emphasis)
    ///
    /// **Valid Range**: `None` or `Some(0.9-0.99)`
    ///
    /// **Effect**: Applies first-order high-pass filter: `y[n] = x[n] -
    /// α*x[n-1]`
    /// - Boosts high frequencies relative to low frequencies
    /// - Compensates for typical speech spectral tilt (more energy in low
    ///   freqs)
    /// - Improves robustness to low-frequency rumble/hum
    ///
    /// **Recommendation**:
    /// - `Some(0.97)`: Standard for speech (default)
    /// - `Some(0.95)`: More aggressive high-pass (very noisy low-freq
    ///   environment)
    /// - `None`: Disable if audio already pre-emphasized or for
    ///   music/non-speech
    pub pre_emphasis: Option<f32>,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            frame_duration: AudioDuration::from_millis(20),
            frame_overlap: 0.5,
            energy_smoothing: 0.85,
            flux_smoothing: 0.8,
            energy_floor: 1e-4,
            flux_floor: 1e-4,
            threshold_smoothing: 0.9,
            activation_margin: 1.1,
            release_margin: 0.9,
            base_threshold: 0.4,
            energy_weight: 0.6,
            flux_weight: 0.4,
            hangover_frames: 3,
            min_speech_frames: 3,
            stream_start_time: AudioTimestamp::EPOCH,
            pre_emphasis: Some(0.97),
        }
    }
}

impl VadConfig {
    /// Validate configuration invariants.
    pub fn validate(&self) -> Result<()> {
        const EPSILON: f32 = 1e-6;

        if self.sample_rate == 0 {
            return Err(invalid_input("sample_rate must be greater than zero"));
        }

        if self.frame_duration.as_nanos() as u64 == 0 {
            return Err(invalid_input("frame_duration must be non-zero"));
        }

        if !(0.0..1.0).contains(&self.frame_overlap) {
            return Err(invalid_input("frame_overlap must be within [0.0, 1.0)"));
        }

        if !(0.0..1.0).contains(&self.energy_smoothing) {
            return Err(invalid_input("energy_smoothing must be within [0.0, 1.0)"));
        }

        if !(0.0..1.0).contains(&self.flux_smoothing) {
            return Err(invalid_input("flux_smoothing must be within [0.0, 1.0)"));
        }

        if !(0.0..1.0).contains(&self.threshold_smoothing) {
            return Err(invalid_input(
                "threshold_smoothing must be within [0.0, 1.0)",
            ));
        }

        if self.activation_margin < 1.0 {
            return Err(invalid_input("activation_margin must be >= 1.0"));
        }

        if self.release_margin <= 0.0 {
            return Err(invalid_input("release_margin must be positive"));
        }

        if self.release_margin > self.activation_margin {
            return Err(invalid_input("release_margin must be <= activation_margin"));
        }

        if self.base_threshold <= 0.0 {
            return Err(invalid_input("base_threshold must be positive"));
        }

        if self.energy_weight < 0.0 || self.flux_weight < 0.0 {
            return Err(invalid_input("metric weights must be non-negative"));
        }

        let weight_sum = self.energy_weight + self.flux_weight;
        if weight_sum.abs() < EPSILON {
            return Err(invalid_input("metric weights must not both be zero"));
        }

        if self.min_speech_frames == 0 {
            return Err(invalid_input("min_speech_frames must be greater than zero"));
        }

        if let Some(coeff) = self.pre_emphasis {
            if !(0.0..1.0).contains(&coeff) {
                return Err(invalid_input(
                    "pre_emphasis coefficient must be in [0.0, 1.0)",
                ));
            }
        }

        Ok(())
    }

    /// Frame length in samples derived from the configured duration and sample
    /// rate.
    /// Returns an error if the computed frame length exceeds platform limits.
    pub fn frame_length_samples(&self) -> Result<usize> {
        let sr = u128::from(self.sample_rate);
        let nanos = self.frame_duration.as_nanos();
        let numerator = nanos
            .saturating_mul(sr)
            .saturating_add(NANOS_PER_SECOND / 2);
        let samples = usize::try_from(numerator / NANOS_PER_SECOND)
            .map_err(|_| invalid_input("frame duration too large for platform"))?;
        // Ensure minimum of 1 sample to prevent division by zero downstream
        Ok(samples.max(1))
    }

    /// Hop size in samples considering the configured frame overlap.
    pub fn hop_length_samples(&self) -> Result<usize> {
        let frame_length = self.frame_length_samples()?;
        let hop = (frame_length as f32 * (1.0 - self.frame_overlap)).round() as usize;
        Ok(hop.max(1))
    }

    /// FFT size for spectral analysis (next power of two of the frame length).
    pub fn fft_size(&self) -> Result<usize> {
        Ok(self.frame_length_samples()?.next_power_of_two())
    }
}

fn invalid_input(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}
