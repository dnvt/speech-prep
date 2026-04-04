//! Integration tests for complete preprocessing pipeline.
//!
//! This module validates the end-to-end integration of all preprocessing
//! components:
//! - DC Offset Removal & High-Pass Filtering
//! - Noise Reduction
//! - Normalization
//! - Quality Assessment
//!
//! # Performance Target
//!
//! - Total latency: <30ms for 1 second of 16 kHz mono audio
//! - Stage breakdown:
//!   - DC/High-pass: <5ms
//!   - Noise reduction: <15ms
//!   - Normalization: <5ms
//!   - Quality assessment: <10ms

// Test-specific lint allows: integration tests require unwrap/expect for ergonomic assertions
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::print_stdout)]
#![allow(clippy::print_stderr)]

use std::time::Instant;

use speech_prep::error::Result;
use speech_prep::preprocessing::{
    DcHighPassFilter, NoiseReducer, NoiseReductionConfig, Normalizer, PreprocessingConfig,
    QualityAssessor, VadContext,
};

const SAMPLE_RATE: u32 = 16000;
const ONE_SECOND_SAMPLES: usize = 16000;

/// Complete preprocessing pipeline result.
#[derive(Debug)]
struct PreprocessedAudio {
    samples: Vec<f32>,
    snr_db: f32,
    energy: f32,
    quality_score: f32,
}

/// Executes the complete preprocessing pipeline.
///
/// Pipeline stages:
/// 1. DC Offset Removal & High-Pass Filtering
/// 2. Noise Reduction
/// 3. Normalization
/// 4. Quality Assessment
fn preprocess_audio(input: &[f32]) -> Result<PreprocessedAudio> {
    // Stage 1: DC offset removal + high-pass filtering
    let dc_config = PreprocessingConfig::default();
    let mut dc_filter = DcHighPassFilter::new(dc_config)?;
    let filtered = dc_filter.process(input, None)?;

    // Stage 2: Noise reduction
    let noise_config = NoiseReductionConfig::default();
    let mut noise_reducer = NoiseReducer::new(noise_config)?;
    let vad_ctx = VadContext { is_silence: false };
    let denoised = noise_reducer.reduce(&filtered, Some(vad_ctx))?;

    // Stage 3: Normalization
    let normalizer = Normalizer::new(0.5, 10.0)?;
    let normalized_audio = normalizer.normalize(&denoised)?;

    // Stage 4: Quality assessment
    let assessor = QualityAssessor::new(SAMPLE_RATE);
    let metrics = assessor.assess(&normalized_audio)?;

    Ok(PreprocessedAudio {
        samples: normalized_audio,
        snr_db: metrics.snr_db,
        energy: metrics.energy,
        quality_score: metrics.quality_score,
    })
}

#[test]
fn test_pipeline_integration_basic() {
    // Basic integration test: pipeline executes without errors
    let input = vec![0.3f32; ONE_SECOND_SAMPLES];

    let result = preprocess_audio(&input).expect("Pipeline should execute without errors");

    // Verify output exists and has expected length
    assert_eq!(result.samples.len(), ONE_SECOND_SAMPLES);

    // Verify quality metrics are valid
    assert!((0.0..=60.0).contains(&result.snr_db));
    assert!((0.0..=1.0).contains(&result.energy));
    assert!((0.0..=1.0).contains(&result.quality_score));
}

#[test]
fn test_pipeline_stages_executed() {
    // Verify all stages actually process the audio
    let mut input = vec![0.0f32; ONE_SECOND_SAMPLES];

    // Add DC offset (will be removed by stage 1)
    for (i, sample) in input.iter_mut().enumerate() {
        let sine_wave = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin();
        *sample = 0.1f32.mul_add(sine_wave, 0.5);
    }

    let result = preprocess_audio(&input).expect("Pipeline should process audio with DC offset");

    // Output should be different from input (processing occurred)
    assert_ne!(result.samples, input);

    // Quality should be assessed (non-zero metrics)
    assert!(result.snr_db > 0.0 || result.quality_score > 0.0);
}

#[test]
fn test_dc_offset_removal_integration() {
    // Create audio with known DC offset
    let dc_offset = 0.3f32;
    let mut input = vec![0.0f32; ONE_SECOND_SAMPLES];
    for (i, sample) in input.iter_mut().enumerate() {
        let sine_component =
            0.1 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin();
        *sample = sine_component.mul_add(1.0, dc_offset);
    }

    let result = preprocess_audio(&input).expect("Pipeline should remove DC offset");

    // Verify DC offset was removed (mean should be near zero)
    let mean: f32 = result.samples.iter().sum::<f32>() / result.samples.len() as f32;
    assert!(
        mean.abs() < 0.1,
        "Expected DC offset removal, mean = {mean:.3}"
    );
}

#[test]
fn test_normalization_integration() {
    // Create very quiet audio signal (will be amplified by normalization)
    // Use a sine wave so there's actual signal content (not just DC)
    let mut input = vec![0.0f32; ONE_SECOND_SAMPLES];
    for (i, sample) in input.iter_mut().enumerate() {
        *sample = 0.05 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin();
    }

    let result = preprocess_audio(&input).expect("Pipeline should normalize quiet audio");

    // After normalization, output should have non-trivial energy
    // (Pipeline may reduce energy due to filtering, but final should still be
    // measurable)
    assert!(
        result.energy > 0.001,
        "Expected measurable energy after normalization, got {:.6}",
        result.energy
    );

    // Quality assessment should complete successfully
    assert!((0.0..=1.0).contains(&result.quality_score));
}

#[test]
fn test_quality_assessment_integration() {
    // Create high-quality audio
    let mut input = vec![0.0f32; ONE_SECOND_SAMPLES];
    // Clean sine wave in middle 50% (clear signal/noise separation)
    let start = ONE_SECOND_SAMPLES / 4;
    let end = 3 * ONE_SECOND_SAMPLES / 4;
    for (offset, sample) in input.iter_mut().skip(start).take(end - start).enumerate() {
        let i = start + offset;
        *sample = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.5;
    }

    let result = preprocess_audio(&input).expect("Pipeline should process high-quality audio");

    // High-quality audio should have decent SNR and quality score
    assert!(
        result.snr_db > 10.0,
        "Expected SNR > 10 dB for clean audio, got {:.1} dB",
        result.snr_db
    );
    assert!(
        result.quality_score > 0.3,
        "Expected quality > 0.3 for clean audio, got {:.2}",
        result.quality_score
    );
}

#[test]
fn test_noise_reduction_integration() {
    // Create noisy audio
    let mut input = vec![0.0f32; ONE_SECOND_SAMPLES];
    for (i, sample) in input.iter_mut().enumerate() {
        let signal =
            0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin();
        let noise =
            0.05 * (2.0 * std::f32::consts::PI * 1200.0 * i as f32 / SAMPLE_RATE as f32).cos();
        *sample = signal + noise;
    }

    // Assess input quality
    let input_assessor = QualityAssessor::new(SAMPLE_RATE);
    let input_metrics = input_assessor
        .assess(&input)
        .expect("Should assess input quality");

    // Process through pipeline
    let result = preprocess_audio(&input).expect("Pipeline should reduce noise");

    // Noise reduction should improve or maintain quality
    // (Note: May not always improve due to signal processing trade-offs)
    assert!(
        result.quality_score >= input_metrics.quality_score * 0.8,
        "Quality degraded significantly: {:.2} -> {:.2}",
        input_metrics.quality_score,
        result.quality_score
    );
}

#[test]
fn test_performance_contract_total_latency() {
    // Validate total pipeline latency is <30ms for 1 second of audio
    let input = vec![0.2f32; ONE_SECOND_SAMPLES];

    let start = Instant::now();
    let _ = preprocess_audio(&input).expect("Pipeline should execute for performance test");
    let elapsed = start.elapsed();

    if cfg!(debug_assertions) {
        println!(
            "Pipeline latency (debug build): {} ms — informational only",
            elapsed.as_millis()
        );
    } else {
        assert!(
            elapsed.as_millis() < 30,
            "Pipeline latency {} ms exceeds 30 ms target",
            elapsed.as_millis()
        );
    }
}

#[test]
fn test_performance_per_stage() {
    // Measure individual stage latencies
    let input = vec![0.2f32; ONE_SECOND_SAMPLES];

    // Stage 1: DC offset + high-pass
    let dc_config = PreprocessingConfig::default();
    let mut dc_filter = DcHighPassFilter::new(dc_config).expect("Should create DC filter");
    let start = Instant::now();
    let filtered = dc_filter
        .process(&input, None)
        .expect("Should process DC/high-pass");
    let dc_elapsed = start.elapsed();

    // Stage 2: Noise reduction
    let noise_config = NoiseReductionConfig::default();
    let mut noise_reducer = NoiseReducer::new(noise_config).expect("Should create noise reducer");
    let vad_ctx = VadContext { is_silence: false };
    let start = Instant::now();
    let denoised = noise_reducer
        .reduce(&filtered, Some(vad_ctx))
        .expect("Should reduce noise");
    let noise_elapsed = start.elapsed();

    // Stage 3: Normalization
    let normalizer = Normalizer::new(0.5, 10.0).expect("Should create normalizer");
    let start = Instant::now();
    let normalized_audio = normalizer
        .normalize(&denoised)
        .expect("Should normalize audio");
    let norm_elapsed = start.elapsed();

    // Stage 4: Quality assessment
    let assessor = QualityAssessor::new(SAMPLE_RATE);
    let start = Instant::now();
    let _ = assessor
        .assess(&normalized_audio)
        .expect("Should assess quality");
    let qual_elapsed = start.elapsed();

    // Log stage timings (visible with --nocapture)
    println!("Stage latencies (1 second of audio @ 16 kHz):");
    println!(
        "  DC/High-pass:     {:>4} ms (target: <5 ms)",
        dc_elapsed.as_millis()
    );
    println!(
        "  Noise reduction:  {:>4} ms (target: <15 ms)",
        noise_elapsed.as_millis()
    );
    println!(
        "  Normalization:    {:>4} ms (target: <5 ms)",
        norm_elapsed.as_millis()
    );
    println!(
        "  Quality assess:   {:>4} ms (target: <10 ms)",
        qual_elapsed.as_millis()
    );

    if cfg!(debug_assertions) {
        println!("Per-stage latency checks skipped for debug builds");
        return;
    }

    // Validate individual stage contracts (release/profiled builds only)
    assert!(
        dc_elapsed.as_millis() < 5,
        "DC/High-pass {} ms exceeds 5 ms target",
        dc_elapsed.as_millis()
    );
    assert!(
        noise_elapsed.as_millis() < 15,
        "Noise reduction {} ms exceeds 15 ms target",
        noise_elapsed.as_millis()
    );
    assert!(
        norm_elapsed.as_millis() < 5,
        "Normalization {} ms exceeds 5 ms target",
        norm_elapsed.as_millis()
    );
    assert!(
        qual_elapsed.as_millis() < 10,
        "Quality assessment {} ms exceeds 10 ms target",
        qual_elapsed.as_millis()
    );
}

#[test]
fn test_error_propagation() {
    // Empty input should fail with clear error
    let empty_input: Vec<f32> = vec![];

    let result = preprocess_audio(&empty_input);

    assert!(result.is_err(), "Pipeline should reject empty input");
}

#[test]
fn test_silence_handling() {
    // Pure silence should be processed without panic
    let input = vec![0.0f32; ONE_SECOND_SAMPLES];

    let result = preprocess_audio(&input).expect("Pipeline should handle silence");

    // Silence should have low energy and quality
    assert!(
        result.energy < 0.01,
        "Expected near-zero energy for silence, got {:.3}",
        result.energy
    );
    assert!(
        result.quality_score < 0.2,
        "Expected low quality for silence, got {:.2}",
        result.quality_score
    );
}

#[test]
fn test_extreme_amplitude_handling() {
    // Near-clipping audio should be handled gracefully
    let input = vec![0.95f32; ONE_SECOND_SAMPLES];

    let result = preprocess_audio(&input).expect("Pipeline should handle extreme amplitudes");

    // Output should be normalized to prevent clipping
    let max_sample = result
        .samples
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_sample <= 1.0,
        "Output amplitude {max_sample:.2} exceeds 1.0 (clipping)"
    );
}

#[test]
fn test_output_bounds_validation() {
    // Verify all output samples are in valid range [-1.0, 1.0]
    let input = vec![0.5f32; ONE_SECOND_SAMPLES];

    let result = preprocess_audio(&input).expect("Pipeline should validate output bounds");

    for (i, &sample) in result.samples.iter().enumerate() {
        assert!(
            (-1.0..=1.0).contains(&sample),
            "Sample {i} = {sample:.3} out of bounds [-1.0, 1.0]"
        );
    }
}
