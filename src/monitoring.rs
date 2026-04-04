//! Lightweight monitoring primitives.

use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe counter for tracking processing metrics.
#[derive(Debug, Default)]
pub struct AtomicCounter(AtomicU64);

impl AtomicCounter {
    pub fn new(initial: u64) -> Self {
        Self(AtomicU64::new(initial))
    }

    pub fn increment(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }

    pub fn add(&self, n: u64) -> u64 {
        self.0.fetch_add(n, Ordering::Relaxed)
    }

    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.0.store(0, Ordering::Relaxed);
    }

    pub fn fetch_add(&self, val: u64) -> u64 {
        self.0.fetch_add(val, Ordering::Relaxed)
    }

    pub fn fetch_sub(&self, val: u64) -> u64 {
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

/// Trait for collecting audio processing metrics.
pub trait AudioMetrics: Send + Sync {
    fn record_latency(&self, _stage: &str, _duration_ms: f64) {}
    fn increment_counter(&self, _name: &str) {}
}

/// No-op metrics implementation.
#[derive(Debug, Default, Clone)]
pub struct NoopMetrics;

impl AudioMetrics for NoopMetrics {}
