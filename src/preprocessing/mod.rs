//! Audio preprocessing module for clean, analysis-ready audio.
//!
//! This module provides the preprocessing and quality foundation, delivering
//! clean, analysis-ready audio for downstream processing.
//!
//! # Components
//!
//! - **DC Offset Removal**: Removes DC bias using exponential moving average
//!   (EMA)
//! - **High-Pass Filtering**: Attenuates low-frequency rumble (<80 Hz) using
//!   Butterworth biquad filters
//! - **Noise Reduction**: Spectral subtraction with adaptive noise profiling
//!   (≥6 dB SNR improvement)
//! - **Chunk Continuity**: Maintains filter state across streaming boundaries
//!
//! # Performance
//!
//! - **DC/High-Pass**: <2ms per 500ms chunk (achieved: 0.014-0.127ms)
//! - **Noise Reduction**: <15ms per 500ms chunk (expected: ~7ms)
//! - **Optimization**: Precomputed coefficients, zero-allocation inner loops
//!
//! # Example Pipeline
//!
//! ```rust,no_run
//! use speech_prep::preprocessing::{
//!     DcHighPassFilter,
//!     NoiseReducer,
//!     NoiseReductionConfig,
//!     PreprocessingConfig,
//!     VadContext,
//! };
//!
//! # fn main() -> speech_prep::error::Result<()> {
//! // Step 1: DC removal + high-pass
//! let dc_config = PreprocessingConfig::default();
//! let mut dc_filter = DcHighPassFilter::new(dc_config)?;
//! let raw_samples = vec![0.0; 8000];
//! let dc_clean = dc_filter.process(&raw_samples, None)?;
//!
//! // Step 2: Noise reduction
//! let noise_config = NoiseReductionConfig::default();
//! let mut noise_reducer = NoiseReducer::new(noise_config)?;
//! let vad_ctx = VadContext { is_silence: false };
//! let denoised = noise_reducer.reduce(&dc_clean, Some(vad_ctx))?;
//! # Ok(())
//! # }
//! ```

pub mod artifacts;
pub mod dc_highpass;
pub mod noise_reduction;
pub mod normalization;
pub mod quality;

pub use artifacts::WavArtifactWriter;
pub use dc_highpass::{DcHighPassFilter, HighpassOrder, PreprocessingConfig, VadContext};
pub use noise_reduction::{NoiseReducer, NoiseReductionArtifacts, NoiseReductionConfig};
pub use normalization::Normalizer;
pub use quality::{QualityAssessor, QualityMetrics};
