use std::time::Duration;

use crate::error::{Error, Result};

/// Configuration for [`Chunker`](super::Chunker).
///
/// Build it with a struct literal, a preset ([`streaming`](Self::streaming),
/// [`batch`](Self::batch)), or [`new`](Self::new), which also validates.
#[derive(Debug, Clone, Copy)]
pub struct ChunkerConfig {
    /// Target duration for each chunk (default: 500ms).
    ///
    /// Chunks will be approximately this duration, but may vary by up to
    /// `duration_tolerance` to align with speech boundaries.
    pub target_duration: Duration,

    /// Maximum allowed chunk duration before forced split (default: 600ms).
    ///
    /// Speech segments longer than this are always split.
    pub max_duration: Duration,

    /// Tolerance for chunk duration variance (default: 100ms).
    ///
    /// Chunks may be `target_duration ± duration_tolerance` to better
    /// align with natural speech boundaries from VAD.
    pub duration_tolerance: Duration,

    /// Minimum chunk duration to emit (default: 100ms).
    ///
    /// A trailing piece of a split segment shorter than this is merged into
    /// the previous piece.
    pub min_duration: Duration,

    /// Duration of overlap between adjacent chunks (default: 50ms).
    ///
    /// Copied into `ProcessedChunk::overlap_prev`.
    /// [`new`](Self::new) accepts 20-80 ms.
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
    /// Create a validated configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] if:
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
            return Err(Error::Configuration(
                "target_duration must be greater than zero".into(),
            ));
        }
        if target_duration > Duration::from_secs(5) {
            return Err(Error::Configuration(
                "target_duration must not exceed 5 seconds".into(),
            ));
        }

        if max_duration < target_duration {
            return Err(Error::Configuration(
                "max_duration must be >= target_duration".into(),
            ));
        }

        if min_duration > target_duration {
            return Err(Error::Configuration(
                "min_duration must be <= target_duration".into(),
            ));
        }

        let overlap_ms = overlap_duration.as_millis();
        if !(20..=80).contains(&overlap_ms) {
            return Err(Error::Configuration(format!(
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

    /// 250 ms chunks, for lower latency.
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

    /// 1 s chunks, for offline processing.
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
