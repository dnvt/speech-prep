//! Lightweight counters and VAD statistics.

use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe counter for tracking processing metrics.
#[derive(Debug, Default)]
pub(crate) struct AtomicCounter(AtomicU64);

impl AtomicCounter {
    pub(crate) fn new(initial: u64) -> Self {
        Self(AtomicU64::new(initial))
    }

    pub(crate) fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    pub(crate) fn reset(&self) {
        self.0.store(0, Ordering::Relaxed);
    }

    pub(crate) fn fetch_add(&self, val: u64) -> u64 {
        self.0.fetch_add(val, Ordering::Relaxed)
    }

    pub(crate) fn fetch_sub(&self, val: u64) -> u64 {
        self.0.fetch_sub(val, Ordering::Relaxed)
    }
}

/// VAD statistics snapshot.
#[derive(Debug, Clone, Default)]
pub struct VADStats {
    pub frames_processed: u64,
    pub speech_frames: u64,
    pub silence_frames: u64,
}

impl VADStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn speech_ratio(&self) -> f64 {
        if self.frames_processed == 0 {
            return 0.0;
        }
        self.speech_frames as f64 / self.frames_processed as f64
    }
}
