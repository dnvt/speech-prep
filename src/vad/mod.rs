//! Voice activity detection.
//!
//! [`VadDetector`] combines frame energy and spectral flux against adaptive
//! baselines and returns speech segments as [`SpeechChunk`]s. It keeps state
//! between calls, so you can feed a stream block by block.
//!
//! ```
//! use speech_prep::{VadConfig, VadDetector};
//!
//! # fn main() -> speech_prep::Result<()> {
//! let detector = VadDetector::new(VadConfig::default())?;
//!
//! // 1 s of silence, 1 s of a 1 kHz tone, 1 s of silence, at 16 kHz.
//! let mut audio = vec![0.0f32; 48_000];
//! for (i, s) in audio[16_000..32_000].iter_mut().enumerate() {
//!     *s = 0.5 * (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / 16_000.0).sin();
//! }
//!
//! let segments = detector.detect(&audio)?;
//! assert_eq!(segments.len(), 1);
//! println!("speech from {} to {}", segments[0].start_time, segments[0].end_time);
//! # Ok(())
//! # }
//! ```
//!
//! This is a signal-level detector. It does not distinguish speech from other
//! sounds with similar energy and spectral change, such as music.

pub mod config;
pub mod detector;
pub mod metrics;

pub use config::VadConfig;
pub use detector::{SpeechChunk, VadDetector};
pub use metrics::{
    AdaptiveThresholdSnapshot, NoopVadMetricsCollector, VadMetricsCollector, VadMetricsSnapshot,
    VadStats,
};
