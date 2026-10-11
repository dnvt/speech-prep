//! Speech-focused audio preprocessing.
//!
//! `speech-prep` turns an audio file or stream into clean, 16 kHz mono
//! chunks that line up with speech, ready for a speech model or a scoring
//! step. Each stage is usable on its own:
//!
//! | Stage | Type |
//! |-------|------|
//! | Decode WAV, FLAC, MP3, AAC/M4A, Ogg Vorbis to 16 kHz mono | [`converter::AudioFormatConverter`] |
//! | Voice activity detection | [`VadDetector`] |
//! | Speech-aligned chunking with overlap | [`Chunker`] |
//! | DC removal and high-pass filter | [`preprocessing::DcHighPassFilter`] |
//! | Stationary noise reduction | [`preprocessing::NoiseReducer`] |
//! | Loudness normalization | [`preprocessing::Normalizer`] |
//! | Quality metrics | [`preprocessing::QualityAssessor`] |
//!
//! [`Pipeline`] wires decoding, VAD, chunking, and cleanup together.
//!
//! # Example
//!
//! ```no_run
//! use speech_prep::Pipeline;
//!
//! # fn main() -> speech_prep::Result<()> {
//! let bytes = std::fs::read("recording.wav")?;
//!
//! let mut pipeline = Pipeline::new()?;
//! let mut chunks = pipeline.process_bytes(&bytes)?.chunks;
//! chunks.extend(pipeline.flush()?.chunks);
//!
//! for chunk in chunks.iter().filter(|c| c.is_speech()) {
//!     println!("{} - {}: {} samples", chunk.start_time, chunk.end_time, chunk.samples.len());
//! }
//! # Ok(())
//! # }
//! ```

pub mod chunker;
pub mod converter;
mod decoder;
pub mod error;
pub mod format;
pub mod pipeline;
pub mod preprocessing;
pub mod time;
pub mod vad;

pub use chunker::{ChunkBoundary, Chunker, ChunkerConfig, ProcessedChunk};
pub use converter::{AudioFormatConverter, StandardAudio};
pub use error::{Error, Result};
pub use pipeline::{Pipeline, ProcessingResult, StageLatencies};
pub use preprocessing::{
    DcHighPassFilter, HighpassOrder, NoiseReducer, NoiseReductionConfig, PreprocessingConfig,
    VadContext,
};
pub use time::{AudioDuration, AudioTimestamp};
pub use vad::{SpeechChunk, VadConfig, VadDetector, VadMetricsCollector, VadStats};

/// Compiles the README examples as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
