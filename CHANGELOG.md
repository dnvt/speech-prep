# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased] - 0.2.0

### Added

- Decoding of FLAC, MP3, AAC (MP4/M4A), and Ogg/WebM Vorbis through
  Symphonia, plus 8-bit, 32-bit, and float WAV. Any channel count is
  downmixed to mono.
- `AudioFormatConverter::convert_file` for reading from a path.
- `Pipeline::process_samples` for already-decoded 16 kHz audio, and
  `Pipeline::reset`.
- `VadConfig::min_speech_energy` to tune the absolute level below which a
  frame is never speech.
- `VadDetector::with_metrics` for attaching a metrics collector.
- `Error::Unsupported` for recognized but undecodable formats (Opus).
- `cargo bench --bench throughput` and a `process_file` example.

### Changed

- **Breaking:** `AudioPipelineCoordinator` is now `Pipeline`. Its methods
  take `&mut self` and return the cleaned `ProcessedChunk`s instead of only
  a count. `process_frame` is now `process_bytes`.
- **Breaking:** `VadDetector::new` takes only a `VadConfig`.
- **Breaking:** `VADStats` is now `vad::VadStats`.
- **Breaking:** `ConversionMetadata` reports a single `conversion_time` and
  `original_channels` is a `u16`.
- **Breaking:** `DcHighPassFilter::process` takes `Option<VadContext>` by
  value, like `NoiseReducer::reduce`.
- **Breaking:** configuration validation errors are `Error::Configuration`
  everywhere; `Error` is `#[non_exhaustive]`.
- The default VAD energy gate is 10x lower (0.002 instead of 0.02). The old
  value ignored voiced audio peaking at -20 dBFS.
- Channels are mixed to mono before resampling instead of after.
- Resampling uses a Kaiser-windowed sinc filter instead of linear
  interpolation. A 12 kHz tone at 48 kHz used to alias to 4 kHz at full
  level; it is now attenuated by about 100 dB.
- Edition 2024. Cargo now resolves dependencies against `rust-version`, so
  1.87 builds without pinning anything.
- Library code no longer logs at `info`/`warn` on every call or warns when a
  stage takes longer than a fixed wall-clock budget.

### Fixed

- The pipeline no longer emits the start of a speech segment that is still
  in progress as a silence chunk.
- `NoiseReducer` no longer fades each chunk in from near zero at its first
  samples (about -23 dB on a steady tone with 500 ms chunks).
- The fourth-order high-pass filter now has Butterworth stage Q values; the
  cutoff is at -3 dB instead of -6 dB.
- `VadStats::speech_frames` counted segments instead of frames, and
  `silence_frames` was always zero.
- `QualityMetrics::spectral_centroid` is now computed from the spectrum. It
  used to be a weighted time index, not a frequency.
- `VadDetector::reset` now also resets the adaptive baselines.
- `LICENSE-APACHE` now contains the full license text.

### Removed

- **Breaking:** `buffer::AudioBuffer`, `types::AudioChunk`, the `fixtures`
  feature and `fixtures` module, `Chunker::default()` as an inherent method
  (use `Default`), `ProcessedChunk::samples_without_overlap`,
  `ConversionMetadata::has_performance_issue`,
  `AudioFormat::is_container_format`, `ProcessingResult::backpressure_active`,
  and `StageLatencies::broadcasting_avg`.
- `glob`, `serde_json`, `parking_lot`, and `rand` dependencies; `hound` is
  now dev-only.

## [0.1.4] - 2026-04-05

- Voice activity detection with energy and spectral-flux signals.
- WAV/PCM conversion to mono 16 kHz `f32` samples.
- DC removal, high-pass filtering, normalization, stationary noise
  reduction, and quality checks.
- Speech-aligned chunking with overlap metadata.

[Unreleased]: https://github.com/dnvt/speech-prep/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/dnvt/speech-prep/releases/tag/v0.1.4
