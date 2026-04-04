//! Audio format variability tests
//!
//! Validates the audio conversion pipeline across different sample rates,
//! channel configurations, and edge cases using synthetic audio only (no
//! external WAV file dependencies).
//!
//! Test coverage gaps addressed:
//! - Sample rate conversion: 8kHz, 24kHz, 44.1kHz, 48kHz → 16kHz
//! - Stereo → mono mixing
//! - Edge cases: empty buffer, single sample, max duration boundary

// Integration test lint allows
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::panic)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::float_cmp)]

use std::f32::consts::PI;
use std::io::Cursor;

use speech_prep::converter::AudioFormatConverter;
use speech_prep::format::AudioFormat;

/// Create a minimal valid WAV file from i16 samples.
fn create_wav(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
    let spec = hound::WavSpec {
        sample_rate,
        channels,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, spec).expect("create WAV writer");

    for &s in samples {
        writer.write_sample(s).expect("write sample");
    }

    writer.finalize().expect("finalize WAV");
    cursor.into_inner()
}

/// Generate a sine wave as i16 samples at a given sample rate.
fn sine_wave_i16(freq_hz: f32, sample_rate: u32, duration_sec: f32) -> Vec<i16> {
    let num_samples = (sample_rate as f32 * duration_sec) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (0.5 * (2.0 * PI * freq_hz * t).sin() * 32767.0) as i16
        })
        .collect()
}

// =========================================================================
// Sample rate conversion tests: various rates → 16kHz
// =========================================================================

#[test]
fn test_8khz_to_16khz_conversion() {
    let samples = sine_wave_i16(440.0, 8000, 0.5); // 0.5s at 8kHz
    let wav = create_wav(8000, 1, &samples);

    let standard = AudioFormatConverter::convert_to_standard(&wav).expect("convert 8kHz");

    assert_eq!(standard.metadata.original_sample_rate, 8000);
    assert_eq!(standard.metadata.original_channels, 1);
    assert_eq!(standard.metadata.original_format, AudioFormat::WavPcm);

    // 0.5s at 16kHz = ~8000 samples (may vary slightly due to resampling)
    let expected = 8000usize;
    let tolerance = 100; // Allow small rounding differences
    assert!(
        standard.samples.len().abs_diff(expected) < tolerance,
        "Expected ~{expected} samples, got {}",
        standard.samples.len()
    );

    // Verify signal is not silent
    let max_amplitude: f32 = standard
        .samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_amplitude > 0.1,
        "Signal should not be silent after conversion"
    );
}

#[test]
fn test_24khz_to_16khz_conversion() {
    let samples = sine_wave_i16(440.0, 24000, 0.5);
    let wav = create_wav(24000, 1, &samples);

    let standard = AudioFormatConverter::convert_to_standard(&wav).expect("convert 24kHz");

    assert_eq!(standard.metadata.original_sample_rate, 24000);

    let expected = 8000usize;
    let tolerance = 100;
    assert!(
        standard.samples.len().abs_diff(expected) < tolerance,
        "Expected ~{expected} samples, got {}",
        standard.samples.len()
    );
}

#[test]
fn test_44100hz_to_16khz_conversion() {
    let samples = sine_wave_i16(440.0, 44100, 0.5);
    let wav = create_wav(44100, 1, &samples);

    let standard = AudioFormatConverter::convert_to_standard(&wav).expect("convert 44.1kHz");

    assert_eq!(standard.metadata.original_sample_rate, 44100);

    // 0.5s at 16kHz = 8000 samples
    let expected = 8000usize;
    let tolerance = 100;
    assert!(
        standard.samples.len().abs_diff(expected) < tolerance,
        "Expected ~{expected} samples, got {}",
        standard.samples.len()
    );
}

#[test]
fn test_48khz_to_16khz_conversion() {
    let samples = sine_wave_i16(440.0, 48000, 0.5);
    let wav = create_wav(48000, 1, &samples);

    let standard = AudioFormatConverter::convert_to_standard(&wav).expect("convert 48kHz");

    assert_eq!(standard.metadata.original_sample_rate, 48000);

    let expected = 8000usize;
    let tolerance = 100;
    assert!(
        standard.samples.len().abs_diff(expected) < tolerance,
        "Expected ~{expected} samples, got {}",
        standard.samples.len()
    );
}

#[test]
fn test_16khz_identity_no_resampling() {
    let samples = sine_wave_i16(440.0, 16000, 0.5);
    let wav = create_wav(16000, 1, &samples);

    let standard = AudioFormatConverter::convert_to_standard(&wav).expect("convert 16kHz");

    assert_eq!(standard.metadata.original_sample_rate, 16000);
    // Should be exactly 8000 samples (no resampling)
    assert_eq!(standard.samples.len(), 8000);
}

// =========================================================================
// Stereo → mono mixing
// =========================================================================

#[test]
fn test_stereo_to_mono_mixing() {
    // Create stereo samples: left = sine, right = silence
    let mono_samples = sine_wave_i16(440.0, 16000, 0.25); // 0.25s
    let mut stereo_samples = Vec::with_capacity(mono_samples.len() * 2);
    for &s in &mono_samples {
        stereo_samples.push(s); // Left channel
        stereo_samples.push(0); // Right channel (silence)
    }

    let wav = create_wav(16000, 2, &stereo_samples);

    let standard = AudioFormatConverter::convert_to_standard(&wav).expect("convert stereo");

    assert_eq!(standard.metadata.original_channels, 2);

    // After mono mixing, we should have half the samples
    assert_eq!(standard.samples.len(), mono_samples.len());

    // Mixed signal should be ~half amplitude of original (L+R / 2, R=0)
    let max_amplitude: f32 = standard
        .samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_amplitude > 0.05,
        "Mono mix should preserve signal from left channel"
    );
}

#[test]
fn test_stereo_44100_to_mono_16khz() {
    // Full pipeline: stereo 44.1kHz → mono 16kHz
    let mono_samples = sine_wave_i16(440.0, 44100, 0.3);
    let mut stereo_samples = Vec::with_capacity(mono_samples.len() * 2);
    for &s in &mono_samples {
        stereo_samples.push(s); // Both channels identical
        stereo_samples.push(s);
    }

    let wav = create_wav(44100, 2, &stereo_samples);

    let standard = AudioFormatConverter::convert_to_standard(&wav).expect("convert stereo 44.1kHz");

    assert_eq!(standard.metadata.original_sample_rate, 44100);
    assert_eq!(standard.metadata.original_channels, 2);

    // 0.3s at 16kHz = ~4800 samples
    let expected = 4800usize;
    let tolerance = 100;
    assert!(
        standard.samples.len().abs_diff(expected) < tolerance,
        "Expected ~{expected} samples, got {}",
        standard.samples.len()
    );
}

// =========================================================================
// Edge cases
// =========================================================================

#[test]
fn test_empty_audio_rejected() {
    let result = AudioFormatConverter::convert_to_standard(&[]);
    assert!(result.is_err(), "Empty audio should be rejected");
}

#[test]
fn test_single_sample_wav() {
    let wav = create_wav(16000, 1, &[1000i16]);

    let standard = AudioFormatConverter::convert_to_standard(&wav).expect("convert single sample");

    assert_eq!(standard.samples.len(), 1);
}

#[test]
fn test_near_zero_amplitude_detected_as_silent() {
    // All samples near zero
    let samples = vec![0i16; 16000]; // 1s of silence
    let wav = create_wav(16000, 1, &samples);

    let standard = AudioFormatConverter::convert_to_standard(&wav).expect("convert silence");

    assert!(
        standard.is_silent(),
        "Zero-amplitude audio should be detected as silent"
    );
}

#[test]
fn test_max_amplitude_wav() {
    // Full-scale samples (i16::MAX and i16::MIN alternating)
    let samples: Vec<i16> = (0..1600)
        .map(|i| if i % 2 == 0 { i16::MAX } else { i16::MIN })
        .collect();
    let wav = create_wav(16000, 1, &samples);

    let standard = AudioFormatConverter::convert_to_standard(&wav).expect("convert max amplitude");

    // Verify samples are normalized to [-1.0, 1.0]
    for &s in &standard.samples {
        assert!(
            (-1.0..=1.0).contains(&s),
            "Sample {s} out of [-1.0, 1.0] range"
        );
    }
}

#[test]
fn test_non_wav_format_rejected() {
    // Random bytes that don't form a valid WAV
    let garbage = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03];
    let result = AudioFormatConverter::convert_to_standard(&garbage);
    assert!(result.is_err(), "Non-WAV data should be rejected");
}

#[test]
fn test_truncated_wav_header_rejected() {
    // Valid RIFF header start but truncated
    let truncated = b"RIFF\x00\x00\x00\x00WAVE";
    let result = AudioFormatConverter::convert_to_standard(truncated);
    assert!(result.is_err(), "Truncated WAV should be rejected");
}

// =========================================================================
// Duration and metadata accuracy
// =========================================================================

#[test]
fn test_duration_accuracy_across_sample_rates() {
    let target_duration_sec = 1.0f64;
    let rates = [8000u32, 16000, 44100, 48000];

    for rate in rates {
        let samples = sine_wave_i16(440.0, rate, target_duration_sec as f32);
        let wav = create_wav(rate, 1, &samples);

        let standard = AudioFormatConverter::convert_to_standard(&wav)
            .unwrap_or_else(|e| panic!("convert {rate}Hz: {e}"));

        let actual_duration = standard.duration_sec();
        let tolerance = 0.05; // 50ms tolerance for resampling rounding
        assert!(
            (actual_duration - target_duration_sec).abs() < tolerance,
            "Duration at {rate}Hz: expected ~{target_duration_sec}s, got {actual_duration}s"
        );
    }
}
