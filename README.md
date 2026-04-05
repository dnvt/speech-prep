# speech-prep

Speech-focused audio preprocessing for Rust — VAD, format detection, WAV decoding, noise reduction, chunking.

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

See `examples/vad_detect.rs` for a runnable demo.

## License

MIT OR Apache-2.0
