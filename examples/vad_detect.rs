//! Detect speech in a synthetic signal.
//!
//! ```bash
//! cargo run --example vad_detect
//! ```

use speech_prep::{VadConfig, VadDetector};

fn main() -> speech_prep::Result<()> {
    let sample_rate = 16_000;

    // 2 s of silence with a 1 kHz tone from 0.3 s to 1.5 s.
    let mut audio = vec![0.0f32; sample_rate * 2];
    for (i, sample) in audio
        .iter_mut()
        .enumerate()
        .take(sample_rate * 3 / 2)
        .skip(sample_rate * 3 / 10)
    {
        let t = i as f32 / sample_rate as f32;
        *sample = 0.5 * (2.0 * std::f32::consts::PI * 1_000.0 * t).sin();
    }

    let detector = VadDetector::new(VadConfig::default())?;
    let segments = detector.detect(&audio)?;

    println!("Tone inserted at 0.300s - 1.500s");
    println!("Detected {} segment(s):", segments.len());
    for seg in &segments {
        println!(
            "  {} - {}  confidence {:.2}",
            seg.start_time, seg.end_time, seg.confidence
        );
    }
    Ok(())
}
