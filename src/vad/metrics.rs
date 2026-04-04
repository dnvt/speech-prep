//! VAD metrics collection and monitoring.

use crate::monitoring::VADStats;
use crate::time::AudioDuration;

/// Metrics sink for VAD instrumentation.
pub trait VadMetricsCollector: Send + Sync {
    /// Record a snapshot of VAD metrics.
    fn record_vad_metrics(&self, snapshot: &VadMetricsSnapshot);
}

/// Structured metrics emitted after each detection call.
#[derive(Debug, Clone)]
pub struct VadMetricsSnapshot {
    /// Frame-level statistics collected during detection.
    pub stats: VADStats,
    /// End-to-end latency for the detection call.
    pub latency: AudioDuration,
    /// Adaptive baseline state captured alongside the stats.
    pub adaptive: AdaptiveThresholdSnapshot,
}

/// Snapshot of the adaptive baseline and threshold state.
#[derive(Debug, Clone, Copy, Default)]
pub struct AdaptiveThresholdSnapshot {
    /// Current exponentially weighted baseline for energy.
    pub energy_baseline: f32,
    /// Current exponentially weighted baseline for spectral flux.
    pub flux_baseline: f32,
    /// Current dynamic decision threshold used by the detector.
    pub dynamic_threshold: f32,
}

impl VadMetricsSnapshot {
    /// Create a new snapshot from aggregated stats and observed latency.
    #[must_use]
    pub fn new(
        stats: VADStats,
        latency: AudioDuration,
        adaptive: AdaptiveThresholdSnapshot,
    ) -> Self {
        Self {
            stats,
            latency,
            adaptive,
        }
    }

    /// Convenience accessor for the speech ratio.
    #[must_use]
    pub fn speech_ratio(&self) -> f64 {
        self.stats.speech_ratio()
    }
}

/// No-op metrics collector for components that do not require instrumentation.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopVadMetricsCollector;

impl VadMetricsCollector for NoopVadMetricsCollector {
    fn record_vad_metrics(&self, _snapshot: &VadMetricsSnapshot) {}
}
