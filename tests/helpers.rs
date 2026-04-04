//! Test helpers for generating synthetic audio data.

/// Generate synthetic speech-like audio (sine wave at 440Hz with amplitude envelope).
pub fn generate_synthetic_speech(sample_rate: u32, duration_secs: f64) -> Vec<f32> {
    let num_samples = (sample_rate as f64 * duration_secs) as usize;
    let freq = 440.0;
    (0..num_samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            let envelope = (t * std::f64::consts::PI / duration_secs).sin();
            (envelope * (2.0 * std::f64::consts::PI * freq * t).sin()) as f32
        })
        .collect()
}

/// Generate white noise at the given sample rate and duration.
pub fn create_white_noise(duration_secs: f64, sample_rate: u32) -> Vec<f32> {
    use rand::Rng;
    let num_samples = (sample_rate as f64 * duration_secs) as usize;
    let mut rng = rand::rng();
    (0..num_samples)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect()
}

/// Mix speech and noise at a given SNR (in dB).
pub fn mix_speech_noise(speech: &[f32], noise: &[f32], snr_db: f32) -> Vec<f32> {
    let speech_power: f32 = speech.iter().map(|s| s * s).sum::<f32>() / speech.len() as f32;
    let noise_power: f32 = noise.iter().map(|n| n * n).sum::<f32>() / noise.len() as f32;
    let target_noise_power = speech_power / (10.0f32).powf(snr_db / 10.0);
    let scale = if noise_power > 0.0 {
        (target_noise_power / noise_power).sqrt()
    } else {
        0.0
    };
    speech
        .iter()
        .zip(noise.iter().cycle())
        .map(|(s, n)| s + n * scale)
        .collect()
}
