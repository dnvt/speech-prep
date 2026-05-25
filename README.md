# speech-prep

Speech-focused audio preprocessing for Rust: voice activity detection, WAV
decoding, resampling, noise reduction, normalization, quality checks, and
speech-aligned chunking.

This crate is small on purpose. It is for preparing learner or voice audio
before model inference or scoring. It is not a full speech recognition system.

```bash
cargo add speech-prep
```

## Usage

```rust
use std::sync::Arc;
use speech_prep::{NoopVadMetricsCollector, VadConfig, VadDetector, VadMetricsCollector};

let config = VadConfig::default();
let metrics: Arc<dyn VadMetricsCollector> = Arc::new(NoopVadMetricsCollector);
let detector = VadDetector::new(config, metrics)?;

let segments = detector.detect(&audio_samples)?;
for seg in &segments {
    println!("{:.3}s — {:.3}s", seg.start_time.as_secs(), seg.end_time.as_secs());
}
```

Run the example:

```bash
cargo run --example vad_detect
```

Example output:

```text
speech-prep: VAD detection example

Audio: 2.0s at 16000Hz (32000 samples)
Speech inserted: 0.3s — 1.5s (1.2s duration)

Detected 1 speech segment(s):
  Segment 1: 0.290s — 1.540s  (confidence: 1.00, energy: 0.0362)
```

## What is included

- VAD with energy and spectral-flux signals.
- WAV/PCM decoding to mono 16kHz `f32` samples.
- Format detection for common containers.
- DC removal, high-pass filtering, normalization, and stationary noise
  reduction.
- Speech-aligned chunking with overlap metadata.
- Lightweight quality metrics for audio checks.

## Current limits

- WAV decoding is implemented. Other formats may be detected, but they are not
  decoded by the high-level converter yet.
- Resampling uses linear interpolation. It is simple and dependency-light, not
  studio-grade sample-rate conversion.
- The VAD example uses synthetic audio. It is a runnable smoke test, not a
  benchmark.

See [`docs/audio-support.md`](docs/audio-support.md) for the current support
matrix and format-support rules. See [`docs/benchmarks.md`](docs/benchmarks.md)
for the current benchmark policy.

## Trust And Scope

- [`SECURITY.md`](SECURITY.md): how to report sensitive issues without exposing
  private audio.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): contribution rules, test commands, and
  AI-assistance disclosure expectations.

## Development

```bash
cargo test
cargo test --features fixtures
cargo clippy -- -D warnings
cargo fmt --check
```

## License

MIT OR Apache-2.0
