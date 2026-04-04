use std::time::Duration;

use crate::error::{Error, Result};

/// Configuration for the audio chunker.
///
/// Controls how audio is segmented into processing chunks.
#[derive(Debug, Clone, Copy)]
pub struct ChunkerConfig {
    /// Target duration for each chunk (default: 500ms).
    ///
    /// Chunks will be approximately this duration, but may vary by up to
    /// `duration_tolerance` to align with speech boundaries.
    pub target_duration: Duration,

    /// Maximum allowed chunk duration before forced split (default: 600ms).
    ///
    /// Long speech segments exceeding this duration will be split into
    /// multiple chunks to maintain streaming latency guarantees.
    pub max_duration: Duration,

    /// Tolerance for chunk duration variance (default: 100ms).
    ///
    /// Chunks may be `target_duration ± duration_tolerance` to better
    /// align with natural speech boundaries from VAD.
    pub duration_tolerance: Duration,

    /// Minimum chunk duration to emit (default: 100ms).
    ///
    /// Segments shorter than this are buffered or merged with adjacent chunks
    /// to avoid inefficient processing of tiny fragments.
    pub min_duration: Duration,

    /// Duration of overlap between adjacent chunks (default: 50ms).
    ///
    /// Preserves acoustic context across chunk boundaries. Must be in range
    /// 20-80ms. Overlaps are stored in
    /// `ProcessedChunk::overlap_prev` and `ProcessedChunk::overlap_next`.
    pub overlap_duration: Duration,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            target_duration: Duration::from_millis(500),
            max_duration: Duration::from_millis(600),
            duration_tolerance: Duration::from_millis(100),
            min_duration: Duration::from_millis(100),
            overlap_duration: Duration::from_millis(50),
        }
    }
}

impl ChunkerConfig {
    /// Create a new chunker configuration with validation.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if:
    /// - `target_duration` is zero or exceeds 5 seconds
    /// - `max_duration` is less than `target_duration`
    /// - `min_duration` exceeds `target_duration`
    /// - `overlap_duration` is outside range 20-80ms
    pub fn new(
        target_duration: Duration,
        max_duration: Duration,
        duration_tolerance: Duration,
        min_duration: Duration,
        overlap_duration: Duration,
    ) -> Result<Self> {
        if target_duration.as_millis() == 0 {
            return Err(Error::InvalidInput(
                "target_duration must be greater than zero".into(),
            ));
        }
        if target_duration > Duration::from_secs(5) {
            return Err(Error::InvalidInput(
                "target_duration must not exceed 5 seconds".into(),
            ));
        }

        if max_duration < target_duration {
            return Err(Error::InvalidInput(
                "max_duration must be >= target_duration".into(),
            ));
        }

        if min_duration > target_duration {
            return Err(Error::InvalidInput(
                "min_duration must be <= target_duration".into(),
            ));
        }

        let overlap_ms = overlap_duration.as_millis();
        if !(20..=80).contains(&overlap_ms) {
            return Err(Error::InvalidInput(format!(
                "overlap_duration must be 20-80ms, got {overlap_ms}ms"
            )));
        }

        Ok(Self {
            target_duration,
            max_duration,
            duration_tolerance,
            min_duration,
            overlap_duration,
        })
    }

    /// Create a configuration optimized for real-time streaming (smaller
    /// chunks).
    #[must_use]
    pub fn streaming() -> Self {
        Self {
            target_duration: Duration::from_millis(250),
            max_duration: Duration::from_millis(300),
            duration_tolerance: Duration::from_millis(50),
            min_duration: Duration::from_millis(100),
            overlap_duration: Duration::from_millis(50),
        }
    }

    /// Create a configuration optimized for batch processing (larger chunks).
    #[must_use]
    pub fn batch() -> Self {
        Self {
            target_duration: Duration::from_secs(1),
            max_duration: Duration::from_millis(1200),
            duration_tolerance: Duration::from_millis(200),
            min_duration: Duration::from_millis(200),
            overlap_duration: Duration::from_millis(50),
        }
    }
}
