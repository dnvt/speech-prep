# Changelog

## 0.1.4 — 2026-04-04

- Reduced Symphonia to the targeted feature set needed for the crate's advertised format support
- Added `cargo test --features fixtures` to CI coverage
- Replaced the crate-level rustdoc snippet with a complete VAD example

## 0.1.3 — 2026-04-04

- Tightened the published crate contents with an explicit package allowlist
- Made `serde_json` fixture-only instead of a core library dependency
- Added docs verification to CI and contributor checks
- Completed the README usage example and trimmed small internal-only API exports

## 0.1.2 — 2026-04-04

- Aligned crate docs and package metadata with the current WAV-only decoding scope
- Renamed buffer metadata to `AudioBufferMetadata`
- Renamed `AudioChunk.timestamp` to `timestamp_secs`
- Hid internal decoder, monitoring, and preprocessing QA helpers from the public API
- Added package validation to CI and contributor checks

## 0.1.1 — 2026-04-04

- Cleaned doc comments and removed extraction artifacts
- Trimmed public API surface (removed unused `AudioMetrics` trait)
- Fixed `fixtures` module path resolution for standalone use
- Removed redundant error constructors

## 0.1.0 — 2026-04-04

Initial release.

- Voice activity detection (dual-metric: energy + spectral flux)
- Audio format detection for WAV/MP3/FLAC/OGG/M4A/Opus plus WAV decoding to 16kHz mono PCM
- Preprocessing: DC removal, high-pass filter, noise reduction, normalization
- Speech-aligned chunking with configurable duration and overlap
- Audio quality assessment metrics
- Performance contract test suite
