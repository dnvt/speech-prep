//! Time types for audio processing.

use std::time::{Duration, Instant};

/// Duration in the audio pipeline. Alias for `std::time::Duration`.
pub type AudioDuration = Duration;

/// Monotonic instant for measuring elapsed time.
pub type AudioInstant = Instant;

/// Timestamp as nanoseconds from the start of an audio stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct AudioTimestamp(u64);

impl AudioTimestamp {
    pub const ZERO: Self = Self(0);
    pub const EPOCH: Self = Self(0);

    pub fn from_secs(secs: f64) -> Self {
        Self((secs * 1_000_000_000.0) as u64)
    }

    pub fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    pub fn from_samples(samples: u64, sample_rate: u32) -> Self {
        if sample_rate == 0 {
            return Self::ZERO;
        }
        Self(samples * 1_000_000_000 / sample_rate as u64)
    }

    pub fn as_secs(&self) -> f64 {
        self.0 as f64 / 1_000_000_000.0
    }

    pub fn as_millis(&self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }

    pub fn nanos(&self) -> u64 {
        self.0
    }

    pub fn add_duration(&self, d: Duration) -> Self {
        Self(self.0.saturating_add(d.as_nanos() as u64))
    }

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

/// Range validation helper.
pub fn validate_in_range<T: PartialOrd + std::fmt::Display>(
    value: T,
    min: T,
    max: T,
    name: &str,
) -> crate::error::Result<()> {
    if value < min || value > max {
        return Err(crate::error::Error::config(format!(
            "{name} = {value} is outside valid range [{min}, {max}]"
        )));
    }
    Ok(())
}
