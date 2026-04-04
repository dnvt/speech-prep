//! Core types for audio processing.

/// A chunk of audio data with metadata.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Raw PCM samples (mono, f32).
    pub data: Vec<f32>,
    /// Chunk sequence number.
    pub chunk_id: u64,
    /// Timestamp in seconds from stream start.
    pub timestamp_secs: f64,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl AudioChunk {
    pub fn new(data: Vec<f32>, chunk_id: u64, timestamp_secs: f64, sample_rate: u32) -> Self {
        Self {
            data,
            chunk_id,
            timestamp_secs,
            sample_rate,
        }
    }

    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.data.len() as f64 / self.sample_rate as f64
    }
}
