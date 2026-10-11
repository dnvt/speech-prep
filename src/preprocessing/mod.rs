//! Signal clean-up stages.
//!
//! | Type | Does | Keeps state across calls |
//! |------|------|--------------------------|
//! | [`DcHighPassFilter`] | Removes DC offset and rumble below ~80 Hz | Yes |
//! | [`NoiseReducer`] | Spectral subtraction of stationary noise | Noise profile |
//! | [`Normalizer`] | Scales to a target RMS with a gain cap | No |
//! | [`QualityAssessor`] | Reports SNR, level, spectral centroid | No |
//!
//! Stateful stages expect one stream fed in order. Pass a [`VadContext`] so
//! the DC estimate and noise profile only learn from silence.
//!
//! ```
//! use speech_prep::preprocessing::{
//!     DcHighPassFilter, NoiseReducer, NoiseReductionConfig, PreprocessingConfig, VadContext,
//! };
//!
//! # fn main() -> speech_prep::Result<()> {
//! let mut filter = DcHighPassFilter::new(PreprocessingConfig::default())?;
//! let mut reducer = NoiseReducer::new(NoiseReductionConfig::default())?;
//!
//! let block = vec![0.01f32; 8000];
//! let ctx = Some(VadContext { is_silence: true });
//! let filtered = filter.process(&block, ctx)?;
//! let cleaned = reducer.reduce(&filtered, ctx)?;
//! assert_eq!(cleaned.len(), block.len());
//! # Ok(())
//! # }
//! ```

pub mod dc_highpass;
pub mod noise_reduction;
pub mod normalization;
pub mod quality;

pub use dc_highpass::{DcHighPassFilter, HighpassOrder, PreprocessingConfig, VadContext};
pub use noise_reduction::{NoiseReducer, NoiseReductionConfig};
pub use normalization::Normalizer;
pub use quality::{QualityAssessor, QualityMetrics};
