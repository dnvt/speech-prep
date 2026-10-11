use crate::error::{Error, Result};
use crate::time::{AudioDuration, AudioTimestamp};

/// What a chunk edge lines up with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkBoundary {
    /// Chunk starts at beginning of speech segment.
    SpeechStart,

    /// Chunk ends at end of speech segment.
    SpeechEnd,

    /// Chunk is mid-speech (continuation of longer segment).
    Continuation,

    /// Chunk contains only silence (no speech detected by VAD).
    Silence,
}

/// A chunk of audio with timing and quality information.
#[derive(Debug, Clone)]
pub struct ProcessedChunk {
    /// Audio samples in the chunk (normalized f32, range [-1.0, 1.0]).
    pub samples: Vec<f32>,

    /// Type of boundary at chunk start.
    pub start_boundary: ChunkBoundary,

    /// Type of boundary at chunk end.
    pub end_boundary: ChunkBoundary,

    /// Absolute start time of chunk in audio stream.
    pub start_time: AudioTimestamp,

    /// Absolute end time of chunk in audio stream.
    pub end_time: AudioTimestamp,

    /// Fraction of the chunk covered by VAD speech, from 0.0 to 1.0.
    pub speech_ratio: f32,

    /// RMS level of `samples`.
    pub energy: f32,

    /// Estimated SNR in dB: `20 * log10(chunk_rms / noise_rms)`, with the
    /// noise level taken from the silent parts of the input. `None` when no
    /// silence was available to estimate it. Measured on the input, before
    /// any cleanup.
    pub snr_db: Option<f32>,

    /// `true` if any sample reaches `|x| >= 0.999`, which usually means the
    /// recording clipped.
    pub has_clipping: bool,

    /// Last `overlap_duration` of the previous chunk, for models that need
    /// left context. `None` for the first chunk of a stream.
    pub overlap_prev: Option<Vec<f32>>,

    /// Last `overlap_duration` of this chunk, i.e. the next chunk's
    /// `overlap_prev`. `None` for the last chunk of a call.
    pub overlap_next: Option<Vec<f32>>,

    /// Length of the overlap vectors in milliseconds. Can be shorter than
    /// configured next to very short chunks.
    pub overlap_ms: u32,
}

impl ProcessedChunk {
    /// Get the duration of this chunk.
    ///
    /// # Errors
    ///
    /// Returns `Error::Processing` if `end_time` < `start_time` (indicates
    /// invalid chunk).
    pub fn duration(&self) -> Result<AudioDuration> {
        self.end_time
            .duration_since(self.start_time)
            .ok_or_else(|| {
                Error::Processing("invalid chunk times: end_time precedes start_time".into())
            })
    }

    /// Check if this chunk contains primarily speech.
    #[must_use]
    pub fn is_speech(&self) -> bool {
        self.speech_ratio > 0.5
    }

    /// Check if this chunk is silence.
    #[must_use]
    pub fn is_silence(&self) -> bool {
        self.start_boundary == ChunkBoundary::Silence && self.end_boundary == ChunkBoundary::Silence
    }

    /// Returns total sample count including overlap regions.
    ///
    /// Useful for buffer allocation when reconstructing the full audio data
    /// with prepended/appended overlaps.
    #[must_use]
    pub fn total_samples_with_overlap(&self) -> usize {
        let prev_overlap = self.overlap_prev.as_ref().map_or(0, Vec::len);
        let next_overlap = self.overlap_next.as_ref().map_or(0, Vec::len);

        self.samples.len() + prev_overlap + next_overlap
    }
}
