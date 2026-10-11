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

    /// The last `overlap_duration` of the previous chunk, for models that
    /// need left context across chunk boundaries. `None` for the first chunk
    /// of a stream. See [`samples_with_context`](Self::samples_with_context).
    pub overlap_prev: Option<Vec<f32>>,

    /// Length of `overlap_prev` in milliseconds. Can be shorter than
    /// configured when the previous chunk is very short.
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

    /// `overlap_prev` followed by `samples`: the chunk with its left
    /// context, ready to feed a model that needs it.
    #[must_use]
    pub fn samples_with_context(&self) -> Vec<f32> {
        let context = self.overlap_prev.as_deref().unwrap_or_default();
        [context, &self.samples].concat()
    }

    /// Set `overlap_prev` and keep `overlap_ms` consistent with it.
    pub(crate) fn set_overlap_prev(&mut self, overlap: Option<Vec<f32>>, sample_rate: u32) {
        let len = overlap.as_ref().map_or(0, Vec::len) as u64;
        let rate = u64::from(sample_rate.max(1));
        self.overlap_ms = ((len * 1000 + rate / 2) / rate) as u32;
        self.overlap_prev = overlap;
    }
}
