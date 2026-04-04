use crate::error::{Error, Result};
use crate::time::{AudioDuration, AudioTimestamp};

/// Type of boundary at chunk edges.
///
/// Indicates the speech context at the start and end of a chunk.
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

/// A processed audio chunk with temporal and quality metadata.
///
/// Represents a segment of audio aligned to speech boundaries.
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

    /// Ratio of speech frames in chunk (0.0 = all silence, 1.0 = all speech).
    ///
    /// Derived from VAD analysis.
    pub speech_ratio: f32,

    /// RMS energy of the chunk (computed during generation).
    ///
    /// Useful for quality assessment and thresholding.
    pub energy: f32,

    /// Signal-to-noise ratio in decibels (dB).
    ///
    /// Computed as `20 * log10(signal_rms / noise_rms)`, where `noise_rms` is
    /// estimated from silence regions. `None` if no noise baseline is available
    /// (e.g., first chunk with no silence).
    ///
    /// Higher SNR values indicate cleaner audio:
    /// - >30 dB: Excellent quality
    /// - 20-30 dB: Good quality
    /// - 10-20 dB: Acceptable quality
    /// - <10 dB: Poor quality (high noise)
    pub snr_db: Option<f32>,

    /// Indicates whether the chunk contains clipping artifacts.
    ///
    /// Clipping occurs when sample values exceed the normalized range
    /// [-1.0, 1.0], typically manifesting as |sample| >= 0.999.
    /// Clipped audio may cause audible distortion.
    ///
    /// `true` if any sample in the chunk is clipped, `false` otherwise.
    pub has_clipping: bool,

    /// Overlap samples from the previous chunk (for context).
    ///
    /// Contains the trailing `overlap_duration` samples from the previous
    /// chunk, preserving acoustic context across the boundary.
    /// `None` for the first chunk in the stream.
    pub overlap_prev: Option<Vec<f32>>,

    /// Overlap samples for the next chunk (for context).
    ///
    /// Contains the trailing `overlap_duration` samples from this chunk, to be
    /// prepended to the next chunk for context. `None` for the last chunk in
    /// the stream.
    pub overlap_next: Option<Vec<f32>>,

    /// Actual overlap duration in milliseconds.
    ///
    /// The duration of samples in `overlap_prev` and `overlap_next`. Typically
    /// matches `ChunkerConfig::overlap_duration` (default 50ms), but may be
    /// shorter for chunks at stream boundaries.
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

    /// Get samples without overlap (deduplicated core content).
    ///
    /// Returns the chunk's primary samples, excluding any overlap regions that
    /// would be duplicated when processing sequential chunks. Useful for
    /// callers that want to avoid processing overlap regions twice.
    pub fn samples_without_overlap(&self) -> &[f32] {
        &self.samples
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
