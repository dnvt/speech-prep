# speech-prep

[![CI](https://github.com/dnvt/speech-prep/actions/workflows/ci.yml/badge.svg)](https://github.com/dnvt/speech-prep/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)

Speech-focused audio preprocessing for Rust.

## Features

- **Voice Activity Detection** — dual-metric (energy + spectral flux) with adaptive thresholds
- **Multi-format decoding** — WAV, MP3, FLAC, OGG, M4A, Opus → 16kHz mono PCM
- **Preprocessing** — DC removal, high-pass filter, spectral noise reduction, normalization
- **Chunking** — speech-aligned segmentation with configurable duration and overlap
- **Quality assessment** — signal metrics for downstream processing gates

## Quick start

```bash
cargo run --example vad_detect
```

```
Detected 1 speech segment(s):
  Segment 1: 0.290s — 1.540s  (confidence: 1.00, energy: 0.0362)
```

## Usage

```rust
use std::sync::Arc;
use speech_prep::vad::{NoopVadMetricsCollector, VadConfig, VadDetector, VadMetricsCollector};

let config = VadConfig::default();
let metrics: Arc<dyn VadMetricsCollector> = Arc::new(NoopVadMetricsCollector);
let detector = VadDetector::new(config, metrics)?;

let segments = detector.detect(&audio_samples)?;
for seg in &segments {
    println!("{:.3}s — {:.3}s", seg.start_time.as_secs(), seg.end_time.as_secs());
}
```

## Pipeline

```
Raw audio bytes
    │
    ▼
Format detection ─→ Decoding ─→ Resampling ─→ Channel mixing
  (format.rs)      (decoder/)    (16kHz)       (mono)
    │
    ▼
Preprocessing ─→ VAD ─→ Chunking
  (preprocessing/)  (vad/)  (chunker/)
    │
    ▼
Processed audio chunks with speech metadata
```

## Modules

| Module | What it does |
|--------|-------------|
| `vad` | Voice activity detection with energy + spectral flux |
| `decoder` | WAV/PCM decoding, sample rate conversion, channel mixing |
| `converter` | Unified format conversion pipeline |
| `format` | Audio format detection (6 formats) |
| `preprocessing` | DC removal, high-pass filter, noise reduction, normalization |
| `chunker` | Speech-aligned segmentation with overlap handling |
| `pipeline` | End-to-end processing coordinator |
| `buffer` | Audio buffer types with metadata |

## Configuration

```rust
use speech_prep::VadConfig;

let config = VadConfig {
    base_threshold: 0.02,      // energy threshold for speech detection
    energy_weight: 0.6,        // weight of energy vs spectral flux
    ..VadConfig::default()
};
```

```rust
use speech_prep::ChunkerConfig;

let config = ChunkerConfig::default(); // 500ms target chunks
```

## License

MIT OR Apache-2.0
