//! Timestamps for positions in an audio stream.

use std::time::{Duration, Instant};

/// Alias for [`std::time::Duration`], used for lengths of audio.
pub type AudioDuration = Duration;

/// Monotonic clock used for the latency figures the crate reports.
pub(crate) type AudioInstant = Instant;

/// A position in an audio stream, stored as nanoseconds from the stream's
/// origin.
///
/// Integer nanoseconds keep sample-accurate positions exact over long
/// streams; use [`as_secs`](Self::as_secs) for display or `f64` math.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct AudioTimestamp(u64);

impl AudioTimestamp {
    /// The stream origin (0 ns).
    pub const ZERO: Self = Self(0);
    /// Same as [`ZERO`](Self::ZERO); reads better as a stream start.
    pub const EPOCH: Self = Self(0);

    /// From seconds. Negative or non-finite values map to zero.
    #[must_use]
    pub fn from_secs(secs: f64) -> Self {
        if !secs.is_finite() || secs <= 0.0 {
            return Self::ZERO;
        }
        let nanos = (secs * 1_000_000_000.0).min(u64::MAX as f64);
        Self(nanos as u64)
    }

    /// From nanoseconds.
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Position of sample `samples` at `sample_rate` Hz. Returns zero if the
    /// rate is zero.
    #[must_use]
    pub fn from_samples(samples: u64, sample_rate: u32) -> Self {
        if sample_rate == 0 {
            return Self::ZERO;
        }
        Self(samples.saturating_mul(1_000_000_000) / u64::from(sample_rate))
    }

    /// Seconds since the origin.
    #[must_use]
    pub fn as_secs(&self) -> f64 {
        self.0 as f64 / 1_000_000_000.0
    }

    /// Milliseconds since the origin.
    #[must_use]
    pub fn as_millis(&self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }

    /// Nanoseconds since the origin.
    #[must_use]
    pub const fn nanos(&self) -> u64 {
        self.0
    }

    /// This timestamp moved forward by `d`, saturating at `u64::MAX` ns.
    #[must_use]
    pub fn add_duration(&self, d: Duration) -> Self {
        let nanos = d.as_nanos().min(u128::from(u64::MAX)) as u64;
        Self(self.0.saturating_add(nanos))
    }

    /// Time elapsed since `earlier`, or `None` if `earlier` is later.
    #[must_use]
    pub fn duration_since(&self, earlier: Self) -> Option<Duration> {
        self.0.checked_sub(earlier.0).map(Duration::from_nanos)
    }
}

impl std::ops::Add<Duration> for AudioTimestamp {
    type Output = Self;
    fn add(self, rhs: Duration) -> Self {
        self.add_duration(rhs)
    }
}

impl std::ops::Sub for AudioTimestamp {
    type Output = Duration;
    fn sub(self, rhs: Self) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(rhs.0))
    }
}

impl std::fmt::Display for AudioTimestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.3}s", self.as_secs())
    }
}
