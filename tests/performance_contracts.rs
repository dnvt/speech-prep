//! Performance contract tests for audio processing.
//!
//! Enforces latency limits per component. Run in release mode:
//!
//! ```bash
//! cargo test --release --test performance_contracts
//! ```
//!
//! CI runners get a tolerance multiplier (3x base, plus SIMD/memory penalties
//! for specific stages) since shared infrastructure is slower than local dev.

#![allow(clippy::expect_used)]
#![allow(clippy::similar_names)]
#![allow(clippy::suboptimal_flops)]

use std::f32::consts::PI;
use std::io::Cursor;
use std::sync::Arc;
use std::time::{Duration, Instant};

use hound::{SampleFormat, WavSpec, WavWriter};
use speech_prep::converter::AudioFormatConverter;
use speech_prep::preprocessing::*;
use speech_prep::time::{AudioDuration, AudioTimestamp};
use speech_prep::{Chunker, NoopVadMetricsCollector, SpeechChunk, VadConfig, VadDetector};
mod helpers;
use helpers::{create_white_noise, generate_synthetic_speech, mix_speech_noise};

// Latency contracts (ms per 1s audio unless noted)
const VAD_LATENCY_MS: u64 = 5;
const FORMAT_CONVERSION_LATENCY_MS: u64 = 10; // 3s stereo clip
const CHUNKING_LATENCY_MS: u64 = 15;
const PIPELINE_TOTAL_LATENCY_MS: u64 = 30;
const DC_HIGHPASS_LATENCY_MS: u64 = 10;
const NOISE_REDUCTION_LATENCY_MS: u64 = 15;
const NORMALIZATION_LATENCY_MS: u64 = 5;
const QUALITY_ASSESSMENT_LATENCY_MS: u64 = 10;

// CI tolerance: shared runners are slower
const CI_TOLERANCE_MULTIPLIER: u64 = 3;
const DC_HIGHPASS_CI_SIMD_PENALTY: u64 = 16; // biquad filter without AVX
const NORMALIZATION_CI_MEMORY_PENALTY: u64 = 8; // RMS with limited cache
const MAX_TIMEOUT_RATE: f32 = 0.05;

fn is_ci_environment() -> bool {
    std::env::var("CI").is_ok()
}

fn get_tolerance_multiplier() -> u64 {
    if is_ci_environment() {
        CI_TOLERANCE_MULTIPLIER
    } else {
        1
    }
}

fn create_test_wav_bytes(sample_rate: u32, channels: u16, duration_secs: f32) -> Vec<u8> {
    const I16_MAX_F32: f32 = i16::MAX as f32;
    const I16_MIN_F32: f32 = i16::MIN as f32;

    let total_frames = (sample_rate as f32 * duration_secs).round() as usize;
    let spec = WavSpec {
        sample_rate,
        channels,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut buffer = Vec::new();
    {
        let cursor = Cursor::new(&mut buffer);
        let mut writer = WavWriter::new(cursor, spec).expect("WAV writer");

        for i in 0..total_frames {
            let t = i as f32 / sample_rate as f32;
            let left = (2.0 * PI * 440.0 * t).sin() * 0.5;
            let right = (2.0 * PI * 660.0 * t).sin() * 0.4;
            let left_sample = (left * I16_MAX_F32).clamp(I16_MIN_F32, I16_MAX_F32) as i16;
            let right_sample = (right * I16_MAX_F32).clamp(I16_MIN_F32, I16_MAX_F32) as i16;

            match channels {
                1 => writer.write_sample(left_sample).expect("write sample"),
                2 => {
                    writer.write_sample(left_sample).expect("write left");
                    writer.write_sample(right_sample).expect("write right");
                }
                #[allow(clippy::panic)]
                other => panic!("unsupported channel count: {other}"),
            }
        }

        writer.finalize().expect("finalize WAV");
    }

    buffer
}

fn assert_latency(elapsed: Duration, limit_ms: u64, label: &str, tolerance: u64) {
    assert!(
        elapsed < Duration::from_millis(limit_ms),
        "{label}: {}ms > {}ms ({tolerance}x tolerance)",
        elapsed.as_millis(),
        limit_ms,
    );
}

// ── VAD / Format / Chunking ─────────────────────────────────────────

#[test]
#[cfg_attr(debug_assertions, ignore = "Release mode only")]
fn test_vad_latency_contract() {
    let tolerance = get_tolerance_multiplier();
    let limit_ms = VAD_LATENCY_MS * tolerance;

    let metrics: Arc<dyn speech_prep::VadMetricsCollector> = Arc::new(NoopVadMetricsCollector);
    let detector = VadDetector::new(VadConfig::default(), metrics).expect("VAD init");

    let speech = generate_synthetic_speech(16_000, 1.0);
    detector.detect(&speech).expect("warm-up");

    let start = Instant::now();
    let _chunks = detector.detect(&speech).expect("VAD detect");
    assert_latency(start.elapsed(), limit_ms, "VAD", tolerance);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Release mode only")]
fn test_format_conversion_latency_contract() {
    let tolerance = get_tolerance_multiplier();
    let limit_ms = FORMAT_CONVERSION_LATENCY_MS * tolerance;

    let wav_bytes = create_test_wav_bytes(44_100, 2, 3.0);
    let _ = AudioFormatConverter::convert_to_standard(&wav_bytes).expect("warm-up");

    let start = Instant::now();
    let result = AudioFormatConverter::convert_to_standard(&wav_bytes).expect("convert");
    assert!(!result.samples.is_empty());
    assert_latency(start.elapsed(), limit_ms, "Format conversion", tolerance);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Release mode only")]
fn test_chunking_latency_contract() {
    let tolerance = get_tolerance_multiplier();
    let limit_ms = CHUNKING_LATENCY_MS * tolerance;

    let chunker = Chunker::default();
    let audio = generate_synthetic_speech(16_000, 1.0);
    let vad_segments = vec![SpeechChunk {
        start_time: AudioTimestamp::ZERO,
        end_time: AudioTimestamp::ZERO.add_duration(AudioDuration::from_secs(1)),
        confidence: 0.95,
        avg_energy: 0.4,
        frame_count: 50,
    }];

    chunker
        .chunk(&audio, 16_000, &vad_segments)
        .expect("warm-up");

    let start = Instant::now();
    let chunks = chunker.chunk(&audio, 16_000, &vad_segments).expect("chunk");
    assert!(!chunks.is_empty());

    let scaled_limit = limit_ms.saturating_mul(chunks.len() as u64);
    assert_latency(start.elapsed(), scaled_limit, "Chunking", tolerance);
}

// ── Preprocessing ───────────────────────────────────────────────────

#[test]
#[cfg_attr(debug_assertions, ignore = "Release mode only")]
fn test_preprocessing_pipeline_total_contract() {
    let tolerance = get_tolerance_multiplier();
    let limit_ms = PIPELINE_TOTAL_LATENCY_MS * tolerance;

    let speech = generate_synthetic_speech(16000, 1.0);
    let noise = create_white_noise(1.0, 16000);
    let noisy_audio = mix_speech_noise(&speech, &noise, 10.0);

    let start = Instant::now();

    let mut dc_filter = DcHighPassFilter::new(PreprocessingConfig::default()).expect("DC init");
    let dc_clean = dc_filter.process(&noisy_audio, None).expect("DC filter");

    let mut reducer = NoiseReducer::new(NoiseReductionConfig::default()).expect("NR init");
    let denoised = reducer.reduce(&dc_clean, None).expect("noise reduce");

    let normalizer = Normalizer::new(0.5, 10.0).expect("normalizer init");
    let normalized = normalizer.normalize(&denoised).expect("normalize");

    let assessor = QualityAssessor::new(16000);
    let _quality = assessor.assess(&normalized).expect("assess");

    assert_latency(start.elapsed(), limit_ms, "Pipeline total", tolerance);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Release mode only")]
fn test_dc_highpass_contract() {
    let tolerance = get_tolerance_multiplier();
    let mut limit_ms = DC_HIGHPASS_LATENCY_MS * tolerance;
    if is_ci_environment() {
        limit_ms *= DC_HIGHPASS_CI_SIMD_PENALTY;
    }

    let duration_secs = 0.25_f32;
    let audio = generate_synthetic_speech(16000, duration_secs as f64);
    let mut filter = DcHighPassFilter::new(PreprocessingConfig::default()).expect("DC init");

    let start = Instant::now();
    let _result = filter.process(&audio, None).expect("DC filter");

    let normalized_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(duration_secs);
    assert!(
        normalized_ms < limit_ms as f64,
        "DC/high-pass: {normalized_ms:.2}ms > {limit_ms}ms (per 1s, {tolerance}x tolerance)"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Release mode only")]
fn test_noise_reduction_contract() {
    let tolerance = get_tolerance_multiplier();
    let limit_ms = NOISE_REDUCTION_LATENCY_MS * tolerance;

    let speech = generate_synthetic_speech(16000, 1.0);
    let noise = create_white_noise(1.0, 16000);
    let noisy_audio = mix_speech_noise(&speech, &noise, 10.0);

    let mut reducer = NoiseReducer::new(NoiseReductionConfig::default()).expect("NR init");

    let start = Instant::now();
    let _result = reducer.reduce(&noisy_audio, None).expect("noise reduce");
    assert_latency(start.elapsed(), limit_ms, "Noise reduction", tolerance);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Release mode only")]
fn test_normalization_contract() {
    let tolerance = get_tolerance_multiplier();
    let mut limit_ms = NORMALIZATION_LATENCY_MS * tolerance;
    if is_ci_environment() {
        limit_ms *= NORMALIZATION_CI_MEMORY_PENALTY;
    }

    let duration_secs = 0.25_f32;
    let audio = generate_synthetic_speech(16000, duration_secs as f64);
    let normalizer = Normalizer::new(0.5, 10.0).expect("normalizer init");

    let start = Instant::now();
    let _result = normalizer.normalize(&audio).expect("normalize");

    let normalized_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(duration_secs);
    assert!(
        normalized_ms < limit_ms as f64,
        "Normalization: {normalized_ms:.2}ms > {limit_ms}ms (per 1s, {tolerance}x tolerance)"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Release mode only")]
fn test_quality_assessment_contract() {
    let tolerance = get_tolerance_multiplier();
    let limit_ms = QUALITY_ASSESSMENT_LATENCY_MS * tolerance;

    let audio = generate_synthetic_speech(16000, 1.0);
    let assessor = QualityAssessor::new(16000);

    let start = Instant::now();
    let _result = assessor.assess(&audio).expect("assess");
    assert_latency(start.elapsed(), limit_ms, "Quality assessment", tolerance);
}

// ── Timeout rate ────────────────────────────────────────────────────

#[test]
#[ignore = "Slow — run with --ignored"]
fn test_preprocessing_pipeline_timeout_rate() {
    const ITERATIONS: usize = 100;
    let limit_ms = PIPELINE_TOTAL_LATENCY_MS * get_tolerance_multiplier();
    let mut timeout_count = 0;

    for _ in 0..ITERATIONS {
        let speech = generate_synthetic_speech(16000, 1.0);
        let noise = create_white_noise(1.0, 16000);
        let noisy_audio = mix_speech_noise(&speech, &noise, 10.0);

        let start = Instant::now();

        let mut dc = DcHighPassFilter::new(PreprocessingConfig::default()).expect("DC");
        let clean = dc.process(&noisy_audio, None).expect("DC");
        let mut nr = NoiseReducer::new(NoiseReductionConfig::default()).expect("NR");
        let denoised = nr.reduce(&clean, None).expect("NR");
        let norm = Normalizer::new(0.5, 10.0).expect("norm");
        let normalized = norm.normalize(&denoised).expect("norm");
        let qa = QualityAssessor::new(16000);
        let _q = qa.assess(&normalized).expect("QA");

        if start.elapsed() >= Duration::from_millis(limit_ms) {
            timeout_count += 1;
        }
    }

    let rate = timeout_count as f32 / ITERATIONS as f32;
    assert!(
        rate <= MAX_TIMEOUT_RATE,
        "Timeout rate: {:.1}% > {:.1}% ({timeout_count}/{ITERATIONS} exceeded {limit_ms}ms)",
        rate * 100.0,
        MAX_TIMEOUT_RATE * 100.0,
    );
}
