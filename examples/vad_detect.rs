//! Detect speech segments in synthetic audio using VAD.
//!
//! ```bash
//! cargo run --example vad_detect
//! ```

use std::sync::Arc;

use speech_prep::vad::{NoopVadMetricsCollector, VadConfig, VadDetector, VadMetricsCollector};

fn main() {
    let config = VadConfig::default();
    let metrics: Arc<dyn VadMetricsCollector> = Arc::new(NoopVadMetricsCollector);
    let detector = VadDetector::new(config, metrics).expect("VAD init");

    // Generate 2 seconds of synthetic audio: silence + speech + silence
    let sample_rate = 16_000;
    let mut audio = vec![0.0f32; sample_rate * 2];

    // Insert a 440Hz tone from 0.3s to 1.5s (simulates speech)
    for i in (0.3 * sample_rate as f64) as usize..(1.5 * sample_rate as f64) as usize {
        let t = i as f64 / sample_rate as f64;
        audio[i] = (2.0 * std::f64::consts::PI * 440.0 * t).sin() as f32 * 0.5;
    }

    let segments = detector.detect(&audio).expect("VAD detection");

    println!("speech-prep: VAD detection example\n");
    println!("Audio: 2.0s at {sample_rate}Hz ({} samples)", audio.len());
    println!("Speech inserted: 0.3s — 1.5s (1.2s duration)\n");
    println!("Detected {} speech segment(s):", segments.len());

    for (i, seg) in segments.iter().enumerate() {
        println!(
            "  Segment {}: {:.3}s — {:.3}s  (confidence: {:.2}, energy: {:.4})",
            i + 1,
            seg.start_time.as_secs(),
            seg.end_time.as_secs(),
            seg.confidence,
            seg.avg_energy,
        );
    }
}
