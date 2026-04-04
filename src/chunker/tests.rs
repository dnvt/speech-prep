use std::time::Duration;

use super::*;
use crate::time::{AudioDuration, AudioTimestamp};

const EPSILON: f32 = 1e-6;

#[test]
fn test_chunker_config_default() {
    let config = ChunkerConfig::default();
    assert_eq!(config.target_duration, Duration::from_millis(500));
    assert_eq!(config.max_duration, Duration::from_millis(600));
}

#[test]
fn test_chunker_config_validation() {
    // Zero target duration should fail
    let result = ChunkerConfig::new(
        Duration::from_millis(0),
        Duration::from_millis(600),
        Duration::from_millis(100),
        Duration::from_millis(100),
        Duration::from_millis(50),
    );
    assert!(result.is_err());

    // max_duration < target_duration should fail
    let result = ChunkerConfig::new(
        Duration::from_millis(500),
        Duration::from_millis(400),
        Duration::from_millis(100),
        Duration::from_millis(100),
        Duration::from_millis(50),
    );
    assert!(result.is_err());

    // overlap_duration < 20ms should fail
    let result = ChunkerConfig::new(
        Duration::from_millis(500),
        Duration::from_millis(600),
        Duration::from_millis(100),
        Duration::from_millis(100),
        Duration::from_millis(10),
    );
    assert!(result.is_err());

    // overlap_duration > 80ms should fail
    let result = ChunkerConfig::new(
        Duration::from_millis(500),
        Duration::from_millis(600),
        Duration::from_millis(100),
        Duration::from_millis(100),
        Duration::from_millis(100),
    );
    assert!(result.is_err());

    // Valid configuration should succeed
    let result = ChunkerConfig::new(
        Duration::from_millis(500),
        Duration::from_millis(600),
        Duration::from_millis(100),
        Duration::from_millis(100),
        Duration::from_millis(50),
    );
    assert!(result.is_ok());
}

#[test]
fn test_empty_audio_returns_error() {
    let chunker = Chunker::default();
    let result = chunker.chunk(&[], 16000, &[]);
    assert!(result.is_err());
}

#[test]
fn test_zero_sample_rate_returns_error() {
    let chunker = Chunker::default();
    let audio = vec![0.0; 1000];
    let result = chunker.chunk(&audio, 0, &[]);
    assert!(result.is_err());
}

#[test]
fn test_silence_chunk_creation() {
    let chunker = Chunker::default();
    let audio = vec![0.0; 16000]; // 1 second of silence @ 16kHz

    let chunks = chunker.chunk(&audio, 16000, &[]).expect("chunking failed");

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].start_boundary, ChunkBoundary::Silence);
    assert_eq!(chunks[0].end_boundary, ChunkBoundary::Silence);
    assert!(
        (chunks[0].speech_ratio - 0.0).abs() < EPSILON,
        "silence speech_ratio"
    );
    assert!(chunks[0].energy < EPSILON);
}

#[test]
fn test_single_short_speech_segment() {
    let chunker = Chunker::default();
    let audio = vec![0.5; 8000]; // 500ms of audio @ 16kHz

    let vad_segments = vec![SpeechChunk {
        start_time: AudioTimestamp::EPOCH,
        end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(500)),
        confidence: 0.9,
        avg_energy: 0.5,
        frame_count: 25,
    }];

    let chunks = chunker
        .chunk(&audio, 16000, &vad_segments)
        .expect("chunking failed");

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].start_boundary, ChunkBoundary::SpeechStart);
    assert_eq!(chunks[0].end_boundary, ChunkBoundary::SpeechEnd);
    assert!(
        (chunks[0].speech_ratio - 1.0).abs() < EPSILON,
        "speech ratio for single segment"
    );
    assert!(chunks[0].energy > 0.0);
}

#[test]
fn test_long_speech_segment_splits_into_chunks() {
    let chunker = Chunker::default();
    let audio = vec![0.5; 16000]; // 1 second @ 16kHz

    let vad_segments = vec![SpeechChunk {
        start_time: AudioTimestamp::EPOCH,
        end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_secs(1)),
        confidence: 0.9,
        avg_energy: 0.5,
        frame_count: 50,
    }];

    let chunks = chunker
        .chunk(&audio, 16000, &vad_segments)
        .expect("chunking failed");

    assert_eq!(chunks.len(), 2); // 1s split into 2x 500ms chunks
    assert_eq!(chunks[0].start_boundary, ChunkBoundary::SpeechStart);
    assert_eq!(chunks[0].end_boundary, ChunkBoundary::Continuation);
    assert_eq!(chunks[1].start_boundary, ChunkBoundary::Continuation);
    assert_eq!(chunks[1].end_boundary, ChunkBoundary::SpeechEnd);
}

#[test]
fn test_long_speech_segment_merges_small_tail() {
    let chunker = Chunker::default();
    let audio = vec![0.5; 16800]; // 1.05 seconds @ 16kHz

    let vad_segments = vec![SpeechChunk {
        start_time: AudioTimestamp::EPOCH,
        end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(1050)),
        confidence: 0.9,
        avg_energy: 0.5,
        frame_count: 52,
    }];

    let chunks = chunker
        .chunk(&audio, 16000, &vad_segments)
        .expect("chunking failed");

    assert_eq!(
        chunks.len(),
        2,
        "Trailing fragment should merge into previous chunk"
    );
    for (i, chunk) in chunks.iter().enumerate() {
        let duration_ms = chunk.duration().unwrap_or_default().as_millis() as u64;
        assert!(
            duration_ms >= 100,
            "Chunk {} duration {}ms should respect min_duration",
            i,
            duration_ms
        );
    }
}

#[test]
fn test_chunk_duration_respects_tolerance_upper_bound() {
    let config = ChunkerConfig::new(
        Duration::from_millis(500),
        Duration::from_millis(600),
        Duration::from_millis(10),
        Duration::from_millis(100),
        Duration::from_millis(50),
    )
    .expect("config should be valid");
    let chunker = Chunker::new(config);
    let sample_rate = 16000;
    let audio = vec![0.5; (sample_rate as usize * 6) / 10]; // 600ms @ 16kHz

    let vad_segments = vec![SpeechChunk {
        start_time: AudioTimestamp::EPOCH,
        end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(600)),
        confidence: 0.9,
        avg_energy: 0.5,
        frame_count: 60,
    }];

    let chunks = chunker
        .chunk(&audio, sample_rate, &vad_segments)
        .expect("chunking failed");
    assert_eq!(
        chunks.len(),
        2,
        "Should split 600ms segment when tolerance is ±10ms"
    );

    let max_duration_ms = (config.target_duration + config.duration_tolerance).as_millis() as u64;

    for (i, chunk) in chunks.iter().enumerate() {
        let duration_ms = chunk.duration().unwrap_or_default().as_millis() as u64;
        assert!(
            duration_ms <= max_duration_ms,
            "Chunk {} duration {}ms exceeds tolerance cap {}ms",
            i,
            duration_ms,
            max_duration_ms
        );
    }
}

#[test]
fn test_speech_with_silence_gaps() {
    let chunker = Chunker::default();
    let audio = vec![0.5; 24000]; // 1.5 seconds @ 16kHz

    // Two 400ms speech segments with 700ms gap
    let vad_segments = vec![
        SpeechChunk {
            start_time: AudioTimestamp::EPOCH,
            end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(400)),
            confidence: 0.9,
            avg_energy: 0.5,
            frame_count: 20,
        },
        SpeechChunk {
            start_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(1100)),
            end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(1500)),
            confidence: 0.9,
            avg_energy: 0.5,
            frame_count: 20,
        },
    ];

    let chunks = chunker
        .chunk(&audio, 16000, &vad_segments)
        .expect("chunking failed");

    assert_eq!(chunks.len(), 3); // speech + silence + speech
    assert_eq!(chunks[0].start_boundary, ChunkBoundary::SpeechStart);
    assert_eq!(chunks[1].start_boundary, ChunkBoundary::Silence);
    assert_eq!(chunks[2].start_boundary, ChunkBoundary::SpeechStart);
}

#[test]
fn test_overlap_metadata_matches_actual_samples() {
    let config = ChunkerConfig::new(
        Duration::from_millis(20),
        Duration::from_millis(30),
        Duration::from_millis(10),
        Duration::from_millis(10),
        Duration::from_millis(40),
    )
    .expect("config");
    let chunker = Chunker::new(config);
    let sample_rate = 16000;
    let audio = vec![0.5; (sample_rate / 50) as usize * 2]; // 40ms total @ 16kHz

    let vad_segments = vec![SpeechChunk {
        start_time: AudioTimestamp::EPOCH,
        end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(40)),
        confidence: 0.9,
        avg_energy: 0.5,
        frame_count: 4,
    }];

    let chunks = chunker
        .chunk(&audio, sample_rate, &vad_segments)
        .expect("chunking failed");
    assert_eq!(chunks.len(), 2, "Should produce two short chunks");

    let first_overlap = chunks[0]
        .overlap_next
        .as_ref()
        .expect("first chunk should expose overlap");
    assert_eq!(
        first_overlap.len(),
        320,
        "First chunk overlap should match chunk length"
    );
    assert_eq!(
        chunks[0].overlap_ms, 20,
        "Overlap metadata should reflect actual samples"
    );

    let second_overlap = chunks[1]
        .overlap_prev
        .as_ref()
        .expect("second chunk should receive overlap");
    assert_eq!(
        second_overlap.len(),
        320,
        "Second chunk overlap should match chunk length"
    );
    assert_eq!(
        chunks[1].overlap_ms, 20,
        "Overlap metadata should reflect actual samples"
    );
}

#[test]
fn test_chunk_duration_calculation() {
    let chunk = ProcessedChunk {
        samples: vec![0.0; 8000],
        start_boundary: ChunkBoundary::SpeechStart,
        end_boundary: ChunkBoundary::SpeechEnd,
        start_time: AudioTimestamp::EPOCH,
        end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(500)),
        speech_ratio: 1.0,
        energy: 0.5,
        snr_db: None,
        has_clipping: false,
        overlap_prev: None,
        overlap_next: None,
        overlap_ms: 0,
    };

    let duration = chunk.duration().unwrap_or_default();
    assert_eq!(duration, Duration::from_millis(500));
}

#[test]
fn test_is_speech_threshold() {
    let mut chunk = ProcessedChunk {
        samples: vec![],
        start_boundary: ChunkBoundary::SpeechStart,
        end_boundary: ChunkBoundary::SpeechEnd,
        start_time: AudioTimestamp::EPOCH,
        end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(500)),
        speech_ratio: 0.6,
        energy: 0.5,
        snr_db: None,
        has_clipping: false,
        overlap_prev: None,
        overlap_next: None,
        overlap_ms: 0,
    };

    assert!(chunk.is_speech());

    chunk.speech_ratio = 0.4;
    assert!(!chunk.is_speech());
}

#[test]
fn test_rms_energy_computation() {
    // Silence should have near-zero energy
    let silence = vec![0.0; 1000];
    let energy = Chunker::compute_rms_energy(&silence);
    assert!(energy < EPSILON);

    // Constant signal should have known RMS
    let signal = vec![0.5; 1000];
    let energy = Chunker::compute_rms_energy(&signal);
    assert!((energy - 0.5).abs() < EPSILON);
}

#[test]
fn test_overlap_generation_two_chunks() {
    let chunker = Chunker::default(); // 50ms overlap
    let audio = vec![0.5f32; 16000]; // 1s @ 16kHz
    let vad_segments = vec![SpeechChunk {
        start_time: AudioTimestamp::EPOCH,
        end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_secs(1)),
        confidence: 0.9,
        avg_energy: 0.5,
        frame_count: 50,
    }];

    let chunks = chunker
        .chunk(&audio, 16000, &vad_segments)
        .expect("chunking should succeed");

    // Should produce 2 chunks (500ms each from 1s audio)
    assert_eq!(chunks.len(), 2);

    // First chunk: no overlap_prev, has overlap_next
    assert!(chunks[0].overlap_prev.is_none());
    assert!(chunks[0].overlap_next.is_some());
    let overlap_next = chunks[0]
        .overlap_next
        .as_ref()
        .expect("overlap_next should exist");
    // 50ms @ 16kHz = 800 samples
    assert_eq!(overlap_next.len(), 800);
    assert_eq!(chunks[0].overlap_ms, 50);

    // Second chunk: has overlap_prev, no overlap_next (last chunk)
    assert!(chunks[1].overlap_prev.is_some());
    assert!(chunks[1].overlap_next.is_none());
    let overlap_prev = chunks[1]
        .overlap_prev
        .as_ref()
        .expect("overlap_prev should exist");
    assert_eq!(overlap_prev.len(), 800);
    assert_eq!(chunks[1].overlap_ms, 50);

    // Verify overlap_prev of chunk[1] matches overlap_next of chunk[0]
    assert_eq!(overlap_prev, overlap_next);
}

#[test]
fn test_overlap_generation_single_chunk() {
    let chunker = Chunker::default();
    let audio = vec![0.5f32; 8000]; // 500ms @ 16kHz (single chunk)
    let vad_segments = vec![SpeechChunk {
        start_time: AudioTimestamp::EPOCH,
        end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(500)),
        confidence: 0.9,
        avg_energy: 0.5,
        frame_count: 25,
    }];

    let chunks = chunker
        .chunk(&audio, 16000, &vad_segments)
        .expect("chunking should succeed");

    // Single chunk has no overlaps
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].overlap_prev.is_none());
    assert!(chunks[0].overlap_next.is_none());
    assert_eq!(chunks[0].overlap_ms, 0); // No overlap data recorded
}

#[test]
fn test_overlap_streaming_config() {
    let chunker = Chunker::new(ChunkerConfig::streaming()); // 250ms chunks, 50ms overlap
    let audio = vec![0.5f32; 16000]; // 1s @ 16kHz
    let vad_segments = vec![SpeechChunk {
        start_time: AudioTimestamp::EPOCH,
        end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_secs(1)),
        confidence: 0.9,
        avg_energy: 0.5,
        frame_count: 50,
    }];

    let chunks = chunker
        .chunk(&audio, 16000, &vad_segments)
        .expect("chunking should succeed");

    // Streaming config should produce ~4 chunks (250ms each from 1s)
    assert!(chunks.len() >= 3);

    // All chunks should have overlap metadata
    for chunk in &chunks {
        assert_eq!(chunk.overlap_ms, 50);
    }

    // Intermediate chunks should have both overlaps
    if chunks.len() > 2 {
        for chunk in chunks.iter().take(chunks.len() - 1).skip(1) {
            assert!(chunk.overlap_prev.is_some());
            assert!(chunk.overlap_next.is_some());
        }
    }
}

#[test]
fn test_overlap_with_silence_gaps() {
    let chunker = Chunker::default();
    let audio = vec![0.5f32; 24000]; // 1.5s @ 16kHz
    let vad_segments = vec![
        SpeechChunk {
            start_time: AudioTimestamp::EPOCH,
            end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(400)),
            confidence: 0.9,
            avg_energy: 0.5,
            frame_count: 20,
        },
        SpeechChunk {
            start_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(1100)),
            end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(1500)),
            confidence: 0.9,
            avg_energy: 0.5,
            frame_count: 20,
        },
    ];

    let chunks = chunker
        .chunk(&audio, 16000, &vad_segments)
        .expect("chunking should succeed");

    // Should have 3 chunks: speech1 + silence + speech2
    assert_eq!(chunks.len(), 3);

    // speech1 → silence: overlap exists
    assert!(chunks[0].overlap_next.is_some());
    assert!(chunks[1].overlap_prev.is_some());

    // silence → speech2: overlap exists
    assert!(chunks[1].overlap_next.is_some());
    assert!(chunks[2].overlap_prev.is_some());
}

#[test]
fn test_dedupe_utilities() {
    let chunk = ProcessedChunk {
        samples: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        start_boundary: ChunkBoundary::SpeechStart,
        end_boundary: ChunkBoundary::Continuation,
        start_time: AudioTimestamp::EPOCH,
        end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(300)),
        speech_ratio: 1.0,
        energy: 0.5,
        snr_db: None,
        has_clipping: false,
        overlap_prev: Some(vec![0.5]),
        overlap_next: Some(vec![5.0]),
        overlap_ms: 10,
    };

    // samples_without_overlap returns core samples
    assert_eq!(chunk.samples_without_overlap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);

    // total_samples_with_overlap calculates total including overlaps
    let total = chunk.total_samples_with_overlap();
    // Core: 5 samples, overlaps contain 1 sample each (explicit vectors)
    assert_eq!(total, 7);
}

#[test]
fn test_reconstruction_from_deduplicated_chunks() {
    // Test that concatenating deduplicated chunks reproduces original audio
    let chunker = Chunker::default();

    // Create distinctive test audio pattern (sine-like pattern for easy
    // verification)
    let mut original_audio = Vec::new();
    for i in 0..16000 {
        // 1s @ 16kHz
        let t = i as f32 / 16000.0;
        original_audio.push((t * 10.0).sin() * 0.5); // Distinctive pattern
    }

    let vad_segments = vec![SpeechChunk {
        start_time: AudioTimestamp::EPOCH,
        end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_secs(1)),
        confidence: 0.9,
        avg_energy: 0.5,
        frame_count: 50,
    }];

    let chunks = chunker
        .chunk(&original_audio, 16000, &vad_segments)
        .expect("chunking should succeed");

    // Reconstruct audio by concatenating deduplicated chunks
    let mut reconstructed = Vec::new();
    for chunk in &chunks {
        let core_samples = chunk.samples_without_overlap();
        reconstructed.extend_from_slice(core_samples);
    }

    // Verify reconstruction matches original
    assert_eq!(
        reconstructed.len(),
        original_audio.len(),
        "Reconstructed length should match original"
    );

    // Verify sample-by-sample equality (within floating point precision)
    for (i, (&original, &reconstructed)) in
        original_audio.iter().zip(reconstructed.iter()).enumerate()
    {
        assert!(
            (original - reconstructed).abs() < EPSILON,
            "Sample {} differs: original={}, reconstructed={}",
            i,
            original,
            reconstructed
        );
    }
}

#[test]
fn test_reconstruction_with_silence_gaps() {
    // Test reconstruction with speech + silence + speech pattern
    let chunker = Chunker::default();

    // Create test audio: 400ms speech + 700ms silence + 400ms speech = 1.5s
    let mut original_audio = Vec::new();

    // Speech 1: 400ms (6400 samples)
    for i in 0..6400 {
        let t = i as f32 / 16000.0;
        original_audio.push((t * 20.0).sin() * 0.5);
    }

    // Silence: 700ms (11200 samples)
    for _ in 0..11200 {
        original_audio.push(0.0);
    }

    // Speech 2: 400ms (6400 samples)
    for i in 0..6400 {
        let t = (i + 6400) as f32 / 16000.0;
        original_audio.push((t * 20.0).cos() * 0.5);
    }

    let vad_segments = vec![
        SpeechChunk {
            start_time: AudioTimestamp::EPOCH,
            end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(400)),
            confidence: 0.9,
            avg_energy: 0.5,
            frame_count: 20,
        },
        SpeechChunk {
            start_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(1100)),
            end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(1500)),
            confidence: 0.9,
            avg_energy: 0.5,
            frame_count: 20,
        },
    ];

    let chunks = chunker
        .chunk(&original_audio, 16000, &vad_segments)
        .expect("chunking should succeed");

    // Reconstruct
    let mut reconstructed = Vec::new();
    for chunk in &chunks {
        reconstructed.extend_from_slice(chunk.samples_without_overlap());
    }

    // Verify
    assert_eq!(reconstructed.len(), original_audio.len());
    for (i, (&orig, &recon)) in original_audio.iter().zip(reconstructed.iter()).enumerate() {
        assert!(
            (orig - recon).abs() < EPSILON,
            "Sample {} mismatch at {:.3}s: orig={:.6}, recon={:.6}",
            i,
            i as f32 / 16000.0,
            orig,
            recon
        );
    }
}

#[test]
fn test_quality_metrics_in_speech_chunks() {
    // Integration test: Verify quality metrics are present in speech chunks
    let chunker = Chunker::default();

    // Audio with known characteristics:
    // - Leading silence (noise baseline)
    // - Speech with moderate amplitude (good SNR)
    // - No clipping
    let sample_rate = 16000;
    let mut audio = Vec::new();

    // Leading silence: 200ms (3200 samples) with noise
    for _ in 0..3200 {
        audio.push(0.01); // Noise floor
    }

    // Speech: 800ms (12800 samples) with higher amplitude
    for i in 0..12800 {
        let t = i as f32 / sample_rate as f32;
        audio.push((t * 10.0).sin() * 0.5); // No clipping
    }

    let vad_segments = vec![SpeechChunk {
        start_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(200)),
        end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(1000)),
        confidence: 0.9,
        avg_energy: 0.5,
        frame_count: 40,
    }];

    let chunks = chunker
        .chunk(&audio, sample_rate, &vad_segments)
        .expect("chunking should succeed");

    // Verify all speech chunks have quality metrics
    let speech_chunks: Vec<_> = chunks.iter().filter(|c| c.is_speech()).collect();

    assert!(
        !speech_chunks.is_empty(),
        "Should have at least one speech chunk"
    );

    for (i, chunk) in speech_chunks.iter().enumerate() {
        // Verify energy is positive
        assert!(
            chunk.energy > 0.0,
            "Chunk {} should have positive energy, got {:.6}",
            i,
            chunk.energy
        );

        // Verify SNR is present (we have silence for noise baseline)
        assert!(
            chunk.snr_db.is_some(),
            "Chunk {} should have SNR (silence regions available)",
            i
        );

        let snr = chunk.snr_db.expect("SNR should be Some");
        assert!(
            snr > 0.0,
            "Chunk {} should have positive SNR, got {:.2} dB",
            i,
            snr
        );

        // Verify no clipping
        assert!(
            !chunk.has_clipping,
            "Chunk {} should not have clipping (max amplitude 0.5)",
            i
        );
    }
}

#[test]
fn test_quality_metrics_with_clipped_audio() {
    // Integration test: Verify clipping detection works in real chunks
    let chunker = Chunker::default();

    // Audio with clipping
    let sample_rate = 16000;
    let mut audio = Vec::new();

    // Leading silence: 200ms (3200 samples)
    for _ in 0..3200 {
        audio.push(0.01);
    }

    // Speech with clipping: 800ms (12800 samples)
    for i in 0..12800 {
        let t = i as f32 / sample_rate as f32;
        let sample = (t * 10.0).sin() * 1.5; // Intentionally exceed [-1.0, 1.0]
        audio.push(sample.clamp(-1.0, 1.0)); // Clamp to simulate clipping
    }

    let vad_segments = vec![SpeechChunk {
        start_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(200)),
        end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(1000)),
        confidence: 0.9,
        avg_energy: 0.5,
        frame_count: 40,
    }];

    let chunks = chunker
        .chunk(&audio, sample_rate, &vad_segments)
        .expect("chunking should succeed");

    // Verify at least one speech chunk detected clipping
    let speech_chunks: Vec<_> = chunks.iter().filter(|c| c.is_speech()).collect();

    let clipped_chunks: Vec<_> = speech_chunks.iter().filter(|c| c.has_clipping).collect();

    assert!(
        !clipped_chunks.is_empty(),
        "Should detect clipping in at least one chunk (>99% accuracy requirement)"
    );
}

#[test]
fn test_wall_clock_timestamps() {
    // P0 Bug Fix: Verify chunker works with wall-clock timestamps (production
    // streaming) Previously, the chunker assumed EPOCH-based timestamps,
    // causing sample index calculations to produce values far beyond audio
    // buffer bounds
    let chunker = Chunker::default();

    // Simulate real streaming: VAD starts at current wall-clock time
    let stream_start = AudioTimestamp::ZERO;

    // Audio: 1 second @ 16kHz
    let sample_rate = 16000;
    let mut audio = Vec::new();

    // Silence: 200ms (3200 samples)
    for _ in 0..3200 {
        audio.push(0.001);
    }

    // Speech: 600ms (9600 samples)
    for i in 0..9600 {
        let t = i as f32 / sample_rate as f32;
        audio.push((t * 10.0).sin() * 0.5);
    }

    // Silence: 200ms (3200 samples)
    for _ in 0..3200 {
        audio.push(0.001);
    }

    // VAD segments with wall-clock timestamps (absolute times)
    let vad_segments = vec![SpeechChunk {
        start_time: stream_start.add_duration(AudioDuration::from_millis(200)),
        end_time: stream_start.add_duration(AudioDuration::from_millis(800)),
        confidence: 0.9,
        avg_energy: 0.5,
        frame_count: 30,
    }];

    // This should work without error (previously would fail with InvalidInput)
    let chunks = chunker
        .chunk_with_stream_start(&audio, sample_rate, &vad_segments, stream_start)
        .expect("chunker should handle wall-clock timestamps");

    // Verify chunks were created correctly
    assert!(
        !chunks.is_empty(),
        "Should produce chunks with wall-clock timestamps"
    );

    // Verify we produce the expected speech chunk (~600ms duration)
    let speech_chunk = chunks
        .iter()
        .find(|chunk| chunk.is_speech())
        .expect("Expected speech chunk with wall-clock timestamps");
    let speech_duration = speech_chunk.duration().expect("speech duration");
    assert!(
        (550..=650).contains(&speech_duration.as_millis()),
        "Speech chunk duration {}ms outside expected range",
        speech_duration.as_millis()
    );

    // Verify chunk timestamps reflect wall-clock origin
    assert!(
        speech_chunk.start_time >= stream_start,
        "Chunk timestamps should be wall-clock based"
    );

    // Verify samples were extracted correctly from audio buffer
    let total_samples: usize = chunks.iter().map(|c| c.samples.len()).sum();
    assert_eq!(
        total_samples,
        audio.len(),
        "Total samples across chunks should match audio buffer"
    );
}
