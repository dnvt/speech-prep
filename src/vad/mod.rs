//! Voice Activity Detection (VAD) with dual-metric analysis.
//!
//! This module provides real-time voice activity detection using a combination
//! of energy and spectral flux metrics with adaptive baseline tracking.
//!
//! # Features
//!
//! - **Dual-metric analysis**: Energy + spectral flux for robust detection
//! - **Adaptive thresholding**: Dynamic baseline tracking for varying noise
//!   conditions
//! - **Streaming support**: Maintains state across detect() calls for
//!   continuous audio
//! - **Configurable sensitivity**: Extensive tuning options via `VadConfig`
//! - **Metrics instrumentation**: Performance and accuracy tracking via
//!   `VadMetricsCollector`
//!
//! # Example
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use speech_prep::vad::{NoopVadMetricsCollector, VadConfig, VadDetector};
//!
//! let config = VadConfig::default();
//! let metrics = Arc::new(NoopVadMetricsCollector);
//! let detector = VadDetector::new(config, metrics)?;
//!
//! // Process audio samples
//! let audio_samples: Vec<f32> = vec![0.0; 16000]; // 1 second at 16kHz
//! let speech_segments = detector.detect(&audio_samples)?;
//!
//! for segment in speech_segments {
//!     println!("Speech detected: {:?} to {:?}", segment.start_time, segment.end_time);
//! }
//! # Ok::<(), speech_prep::error::Error>(())
//! ```

pub mod config;
pub mod detector;
pub mod metrics;

// Re-export main types for convenient access
pub use config::VadConfig;
pub use detector::{SpeechChunk, VadDetector};
pub use metrics::{
    AdaptiveThresholdSnapshot, NoopVadMetricsCollector, VadMetricsCollector, VadMetricsSnapshot,
};
