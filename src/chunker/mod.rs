//! Audio chunking pipeline for multi-consumer processing.
//!
//! This module segments standardized PCM audio into speech-aligned chunks with
//! rich metadata, enabling parallel downstream processing by speech
//! recognition, scoring, and alignment pipelines.
//!
//! # Architecture
//!
//! The chunker follows a streaming-first design:
//! 1. Accept VAD boundaries (`SpeechChunk`) + raw PCM samples
//! 2. Generate fixed-duration chunks (default 500ms) aligned to speech
//!    boundaries
//! 3. Attach temporal metadata (`AudioTimestamp`) for deterministic testing
//! 4. Provide quality metrics (energy, speech ratio) for adaptive routing
//!
//! # Performance Contracts
//!
//! - **Latency**: <15ms total processing per chunk
//! - **Alignment**: ±20ms accuracy to VAD boundaries
//! - **Coverage**: Chunks cover 100% of input duration (no gaps)
//!
//! # Example
//!
//! ```rust
//! use speech_prep::{Chunker, ChunkerConfig, SpeechChunk};
//! use speech_prep::time::{AudioDuration, AudioTimestamp};
//!
//! let config = ChunkerConfig::default(); // 500ms chunks
//! let chunker = Chunker::new(config);
//!
//! let audio: Vec<f32> = vec![0.0; 16000]; // 1 second @ 16kHz
//! let vad_segments = vec![SpeechChunk {
//!     start_time:  AudioTimestamp::EPOCH,
//!     end_time:    AudioTimestamp::EPOCH
//!         .add_duration(AudioDuration::from_secs(1)),
//!     confidence:  0.9,
//!     avg_energy:  0.5,
//!     frame_count: 50,
//! }];
//!
//! let chunks = chunker.chunk(&audio, 16000, &vad_segments)?;
//! assert_eq!(chunks.len(), 2); // Two 500ms chunks from 1s speech
//!
//! // Overlaps are automatically added between chunks
//! assert!(chunks[0].overlap_next.is_some()); // First chunk has overlap for next
//! assert!(chunks[1].overlap_prev.is_some()); // Second chunk has overlap from prev
//! # Ok::<(), speech_prep::error::Error>(())
//! ```

use crate::error::{Error, Result};
use crate::time::AudioTimestamp;
use std::time::{Duration, Instant};

use crate::SpeechChunk;

mod analysis;
mod config;
mod overlap;
mod planner;
mod segments;
mod types;

pub use config::ChunkerConfig;
use overlap::apply_overlaps;
pub use types::{ChunkBoundary, ProcessedChunk};

/// Audio chunker for segmenting streams into processing units.
///
/// Combines VAD boundaries with duration heuristics to produce chunks optimized
/// for downstream processing by downstream consumers.
#[derive(Debug, Clone, Copy)]
pub struct Chunker {
    config: ChunkerConfig,
}

#[allow(clippy::multiple_inherent_impl)]
impl Chunker {
    /// Create a new chunker with the given configuration.
    #[must_use]
    pub fn new(config: ChunkerConfig) -> Self {
        Self { config }
    }

    /// Create a chunker with default configuration (500ms chunks).
    ///
    /// Alias for `Chunker::new(ChunkerConfig::default())`.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self::new(ChunkerConfig::default())
    }

    /// Segment audio into processing chunks aligned to VAD boundaries.
    ///
    /// This variant assumes that VAD timestamps are relative to the Unix epoch
    /// (e.g., tests that build times off `AudioTimestamp::EPOCH`). For streaming
    /// scenarios where VAD emits wall-clock timestamps (`AudioTimestamp::now()`),
    /// prefer [`Chunker::chunk_with_stream_start`] so the chunker can normalize
    /// against the actual stream start.
    ///
    /// # Arguments
    ///
    /// - `audio`: Raw PCM samples (f32, normalized to [-1.0, 1.0])
    /// - `sample_rate`: Audio sample rate in Hz (must be > 0)
    /// - `vad_segments`: Speech boundaries from VAD analysis
    ///
    /// # Returns
    ///
    /// Vector of `ProcessedChunk` covering the entire input duration with no
    /// gaps.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if:
    /// - `sample_rate` is zero
    /// - `audio` is empty
    /// - VAD segments have invalid timestamps (end < start)
    ///
    /// # Performance
    ///
    /// Target: <15ms total processing time per chunk generated.
    pub fn chunk(
        &self,
        audio: &[f32],
        sample_rate: u32,
        vad_segments: &[SpeechChunk],
    ) -> Result<Vec<ProcessedChunk>> {
        self.chunk_with_stream_start(audio, sample_rate, vad_segments, AudioTimestamp::EPOCH)
    }

    /// Segment audio into processing chunks with an explicit stream start time.
    ///
    /// Use this variant when the VAD timestamps are absolute (e.g., wall-clock)
    /// rather than relative to the Unix epoch.
    ///
    /// ```
    /// use speech_prep::{Chunker, ChunkerConfig, SpeechChunk};
    /// use speech_prep::time::{AudioDuration, AudioTimestamp};
    ///
    /// # fn main() -> speech_prep::error::Result<()> {
    /// let chunker = Chunker::new(ChunkerConfig::streaming());
    /// let stream_start = AudioTimestamp::EPOCH;
    ///
    /// // VAD emits wall-clock timestamps relative to the live stream
    /// let segments = vec![SpeechChunk {
    ///     start_time:  stream_start,
    ///     end_time:    stream_start.add_duration(AudioDuration::from_millis(240)),
    ///     confidence:  0.92,
    ///     avg_energy:  0.4,
    ///     frame_count: 48,
    /// }];
    ///
    /// let audio = vec![0.0; 3840]; // 240ms @ 16kHz
    /// let chunks = chunker.chunk_with_stream_start(&audio, 16_000, &segments, stream_start)?;
    /// assert_eq!(chunks.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn chunk_with_stream_start(
        &self,
        audio: &[f32],
        sample_rate: u32,
        vad_segments: &[SpeechChunk],
        stream_start_time: AudioTimestamp,
    ) -> Result<Vec<ProcessedChunk>> {
        if sample_rate == 0 {
            return Err(Error::InvalidInput("sample_rate must be > 0".into()));
        }
        if audio.is_empty() {
            return Err(Error::InvalidInput("audio buffer is empty".into()));
        }

        for segment in vad_segments {
            if segment.end_time < segment.start_time {
                return Err(Error::InvalidInput(
                    "VAD segment has end_time < start_time".into(),
                ));
            }
        }

        let processing_start = Instant::now();

        let total_samples = audio.len();
        let total_duration_secs = total_samples as f64 / f64::from(sample_rate);
        let total_duration = Duration::from_secs_f64(total_duration_secs);

        let earliest_segment_start = vad_segments.iter().map(|seg| seg.start_time).min();
        let audio_start = earliest_segment_start.map_or(stream_start_time, |start| {
            std::cmp::min(start, stream_start_time)
        });

        let noise_baseline =
            Self::compute_noise_baseline(audio, sample_rate, vad_segments, audio_start);

        let estimated_chunks =
            (total_duration.as_millis() / self.config.target_duration.as_millis()).max(1) as usize
                + 1;
        let mut chunks = Vec::with_capacity(estimated_chunks);

        if vad_segments.is_empty() {
            chunks.push(Self::create_silence_chunk(
                audio,
                sample_rate,
                audio_start,
                total_duration,
                audio_start,
            )?);
        } else {
            let mut current_time = audio_start;

            for segment in vad_segments {
                if segment.start_time > current_time {
                    let silence_end = segment.start_time;
                    let silence_duration =
                        silence_end.duration_since(current_time).ok_or_else(|| {
                            Error::Processing("VAD segment start_time < current_time".into())
                        })?;

                    chunks.push(Self::create_silence_chunk(
                        audio,
                        sample_rate,
                        current_time,
                        silence_duration,
                        audio_start,
                    )?);
                }

                let segment_chunks = self.process_speech_segment(
                    audio,
                    sample_rate,
                    segment,
                    noise_baseline,
                    audio_start,
                )?;
                chunks.extend(segment_chunks);

                current_time = segment.end_time;
            }

            let total_end_time = audio_start.add_duration(total_duration);
            if total_end_time > current_time {
                let trailing_duration = total_end_time
                    .duration_since(current_time)
                    .ok_or_else(|| Error::Processing("total_end_time < current_time".into()))?;
                chunks.push(Self::create_silence_chunk(
                    audio,
                    sample_rate,
                    current_time,
                    trailing_duration,
                    audio_start,
                )?);
            }
        }

        let overlap_samples = Self::duration_to_samples(self.config.overlap_duration, sample_rate);
        apply_overlaps(&mut chunks, overlap_samples, sample_rate);

        let latency = processing_start.elapsed();
        let chunk_count = chunks.len().max(1);
        let _per_chunk = Duration::from_secs_f64(latency.as_secs_f64() / chunk_count as f64);
        for _ in 0..chunk_count {}

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests;
