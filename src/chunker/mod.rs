//! Split audio into chunks aligned with speech segments.
//!
//! Given samples and the [`SpeechChunk`]s from a VAD, [`Chunker`] covers the
//! whole input with contiguous chunks:
//!
//! ```text
//! audio   |------ silence ------|========== speech ==========|--- silence ---|
//! chunks  [  Silence           ][ Start ][ Cont. ][   End    ][  Silence     ]
//!                                 ~target_duration each
//! ```
//!
//! Silence between segments becomes one chunk. Speech segments are split
//! into pieces close to `target_duration` (never longer than
//! `max_duration`), and short tails are merged into the previous piece.
//! Every chunk gets energy, speech ratio, SNR (when a noise baseline is
//! available), and a copy of the neighbouring samples in `overlap_prev` /
//! `overlap_next` for models that need context across boundaries.
//!
//! ```
//! use speech_prep::time::{AudioDuration, AudioTimestamp};
//! use speech_prep::{Chunker, SpeechChunk};
//!
//! # fn main() -> speech_prep::Result<()> {
//! let chunker = Chunker::default(); // 500 ms target
//!
//! let audio = vec![0.0f32; 16_000]; // 1 s at 16 kHz
//! let speech = SpeechChunk {
//!     start_time: AudioTimestamp::EPOCH,
//!     end_time: AudioTimestamp::EPOCH + AudioDuration::from_secs(1),
//!     confidence: 0.9,
//!     avg_energy: 0.5,
//!     frame_count: 50,
//! };
//!
//! let chunks = chunker.chunk(&audio, 16_000, &[speech])?;
//! assert_eq!(chunks.len(), 2);
//! assert!(chunks[0].overlap_next.is_some());
//! assert!(chunks[1].overlap_prev.is_some());
//! # Ok(())
//! # }
//! ```

use crate::error::{Error, Result};
use crate::time::AudioTimestamp;
use std::time::Duration;

use crate::SpeechChunk;

mod analysis;
mod config;
mod overlap;
mod planner;
mod segments;
mod types;

pub use config::ChunkerConfig;
pub(crate) use overlap::apply_overlaps;
pub use types::{ChunkBoundary, ProcessedChunk};

/// Splits audio into speech-aligned chunks. See the [module docs](self).
#[derive(Debug, Clone, Copy, Default)]
pub struct Chunker {
    config: ChunkerConfig,
}

impl Chunker {
    /// Create a chunker with the given configuration.
    #[must_use]
    pub fn new(config: ChunkerConfig) -> Self {
        Self { config }
    }

    /// The configuration this chunker was built with.
    #[must_use]
    pub fn config(&self) -> &ChunkerConfig {
        &self.config
    }

    /// Chunk `audio`, interpreting VAD timestamps as relative to
    /// [`AudioTimestamp::EPOCH`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if `sample_rate` is zero, `audio` is
    /// empty, or a segment ends before it starts.
    pub fn chunk(
        &self,
        audio: &[f32],
        sample_rate: u32,
        vad_segments: &[SpeechChunk],
    ) -> Result<Vec<ProcessedChunk>> {
        self.chunk_with_stream_start(audio, sample_rate, vad_segments, AudioTimestamp::EPOCH)
    }

    /// Chunk `audio` whose first sample is at `stream_start_time`.
    ///
    /// Use this when the VAD timestamps include a stream offset, for example
    /// when processing the second minute of a live stream.
    ///
    /// # Errors
    ///
    /// Same as [`chunk`](Self::chunk).
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

        let segments = merge_segments(vad_segments);

        let total_samples = audio.len();
        let total_duration_secs = total_samples as f64 / f64::from(sample_rate);
        let total_duration = Duration::from_secs_f64(total_duration_secs);

        let earliest_segment_start = segments.iter().map(|seg| seg.start_time).min();
        let audio_start = earliest_segment_start.map_or(stream_start_time, |start| {
            std::cmp::min(start, stream_start_time)
        });

        let noise_baseline =
            Self::compute_noise_baseline(audio, sample_rate, &segments, audio_start);

        let estimated_chunks =
            (total_duration.as_millis() / self.config.target_duration.as_millis()).max(1) as usize
                + 1;
        let mut chunks = Vec::with_capacity(estimated_chunks);

        if segments.is_empty() {
            chunks.push(Self::create_silence_chunk(
                audio,
                sample_rate,
                audio_start,
                total_duration,
                audio_start,
            )?);
        } else {
            let mut current_time = audio_start;

            for segment in &segments {
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

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests;

/// Sort segments by start time and merge any that overlap, so the chunker
/// never emits the same audio twice.
fn merge_segments(segments: &[SpeechChunk]) -> Vec<SpeechChunk> {
    let mut sorted = segments.to_vec();
    sorted.sort_by_key(|seg| seg.start_time);

    let mut merged: Vec<SpeechChunk> = Vec::with_capacity(sorted.len());
    for seg in sorted {
        match merged.last_mut() {
            Some(prev) if seg.start_time < prev.end_time => {
                let frames = prev.frame_count + seg.frame_count;
                let weight = |s: &SpeechChunk| s.frame_count as f32 / frames.max(1) as f32;
                prev.confidence = prev.confidence * weight(prev) + seg.confidence * weight(&seg);
                prev.avg_energy = prev.avg_energy * weight(prev) + seg.avg_energy * weight(&seg);
                prev.end_time = prev.end_time.max(seg.end_time);
                prev.frame_count = frames;
            }
            _ => merged.push(seg),
        }
    }
    merged
}
