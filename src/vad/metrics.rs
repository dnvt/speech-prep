//! Optional instrumentation hooks for the VAD.

use crate::time::AudioDuration;

/// Receives a [`VadMetricsSnapshot`] after every
/// [`VadDetector::detect`](super::VadDetector::detect) call.
///
/// Implement this to export VAD behaviour to your own metrics system. Pass
/// it with [`VadDetector::with_metrics`](super::VadDetector::with_metrics).
pub trait VadMetricsCollector: Send + Sync {
    /// Record one snapshot.
    fn record_vad_metrics(&self, snapshot: &VadMetricsSnapshot);
}

/// Frame counts from one `detect` call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VadStats {
    /// Analysis frames processed.
    pub frames_processed: u64,
    /// Frames whose level marked them as speech. Hangover frames, which only
    /// keep a segment open, count as silence.
    pub speech_frames: u64,
    /// Frames classified as silence.
    pub silence_frames: u64,
}

impl VadStats {
    /// Fraction of frames classified as speech, or 0.0 if none were processed.
    #[must_use]
    pub fn speech_ratio(&self) -> f64 {
        if self.frames_processed == 0 {
            return 0.0;
        }
        self.speech_frames as f64 / self.frames_processed as f64
    }
}

/// Metrics emitted after each detection call.
#[derive(Debug, Clone, Copy)]
pub struct VadMetricsSnapshot {
    /// Frame counts for the call.
    pub stats: VadStats,
    /// Wall-clock time spent in the call.
    pub latency: AudioDuration,
    /// Noise floor at the end of the call: the running estimate of the
    /// background frame energy that speech is measured against.
    pub noise_floor: f32,
}

/// Collector that discards every snapshot. Used by [`VadDetector::new`](super::VadDetector::new).
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopVadMetricsCollector;

impl VadMetricsCollector for NoopVadMetricsCollector {
    fn record_vad_metrics(&self, _snapshot: &VadMetricsSnapshot) {}
}
