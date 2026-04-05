//! # speech-prep
//!
//! Speech-focused audio preprocessing for Rust.
//!
//! - Voice activity detection (dual-metric: energy + spectral flux)
//! - Audio format detection plus WAV decoding to 16kHz mono PCM
//! - Preprocessing (DC removal, high-pass filter, noise reduction, normalization)
//! - Speech-aligned chunking with overlap handling
//! - Quality assessment metrics
//!
//! ## Usage
//!
//! ```rust
//! use std::sync::Arc;
//! use speech_prep::{NoopVadMetricsCollector, VadConfig, VadDetector, VadMetricsCollector};
//!
//! # fn main() -> Result<(), speech_prep::Error> {
//! let config = VadConfig::default();
//! let metrics: Arc<dyn VadMetricsCollector> = Arc::new(NoopVadMetricsCollector);
//! let detector = VadDetector::new(config, metrics)?;
//!
//! let audio_samples = vec![0.0; 16_000];
//! let _segments = detector.detect(&audio_samples)?;
//! # Ok(())
//! # }
//! ```

#![cfg_attr(test, allow(clippy::unwrap_used))]
#![cfg_attr(test, allow(clippy::expect_used))]
#![cfg_attr(test, allow(clippy::panic))]
#![cfg_attr(test, allow(clippy::indexing_slicing))]
#![cfg_attr(test, allow(clippy::print_stdout))]
#![cfg_attr(test, allow(clippy::float_cmp))]

pub mod buffer;
pub mod chunker;
pub mod converter;
mod decoder;
pub mod error;
#[cfg(any(test, feature = "fixtures"))]
pub mod fixtures;
pub mod format;
mod monitoring;
pub mod pipeline;
pub mod preprocessing;
pub mod time;
pub mod types;
pub mod vad;

pub use buffer::{AudioBuffer, AudioBufferMetadata};
pub use chunker::{ChunkBoundary, Chunker, ChunkerConfig, ProcessedChunk};
pub use error::{Error, Result};
pub use monitoring::VADStats;
pub use pipeline::{AudioPipelineCoordinator, ProcessingResult, StageLatencies};
pub use preprocessing::{
    DcHighPassFilter, HighpassOrder, NoiseReducer, NoiseReductionConfig, PreprocessingConfig,
    VadContext,
};
pub use time::{AudioDuration, AudioTimestamp};
pub use types::AudioChunk;
pub use vad::{NoopVadMetricsCollector, SpeechChunk, VadConfig, VadDetector, VadMetricsCollector};
