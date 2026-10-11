//! VAD regression test on a real recording.
//!
//! `tests/fixtures/jfk.ogg` is the opening of John F. Kennedy's 1961
//! inaugural address ("And so, my fellow Americans: ask not what your country
//! can do for you, ask what you can do for your country."), a US government
//! work in the public domain, as 16 kHz mono Vorbis.
//!
//! The reference segments come from Silero VAD v5 with its standard settings
//! (threshold 0.5, min speech 250 ms, min silence 100 ms, 30 ms padding).

use speech_prep::{AudioFormatConverter, VadConfig, VadDetector};

const REFERENCE: [(f64, f64); 4] = [
    (0.322, 2.270),
    (3.266, 4.446),
    (5.378, 7.678),
    (8.162, 10.622),
];

fn inside(segments: &[(f64, f64)], t: f64) -> bool {
    segments.iter().any(|&(a, b)| t >= a && t < b)
}

#[test]
fn jfk_matches_reference_segmentation() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/jfk.ogg");
    let audio = AudioFormatConverter::convert_file(path).unwrap();

    let vad = VadDetector::new(VadConfig::default()).unwrap();
    let mut segments = vad.detect(&audio.samples).unwrap();
    segments.extend(vad.flush().unwrap());
    let ours: Vec<(f64, f64)> = segments
        .iter()
        .map(|s| (s.start_time.as_secs(), s.end_time.as_secs()))
        .collect();

    // Score 10 ms frames against the reference.
    let (mut tp, mut fp, mut fn_) = (0u32, 0u32, 0u32);
    for i in 0..(audio.duration_sec() * 100.0) as usize {
        let t = (i as f64 + 0.5) / 100.0;
        match (inside(&ours, t), inside(&REFERENCE, t)) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => {}
        }
    }
    let precision = f64::from(tp) / f64::from(tp + fp);
    let recall = f64::from(tp) / f64::from(tp + fn_);

    assert!(precision >= 0.85, "precision {precision:.2}: {ours:?}");
    assert!(recall >= 0.85, "recall {recall:.2}: {ours:?}");
    assert!(
        (3..=7).contains(&ours.len()),
        "{} segments for 4 phrases: {ours:?}",
        ours.len()
    );
}
