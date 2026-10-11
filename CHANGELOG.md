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
- `Pipeline::with_max_speech_hold` (default 2 s): long uninterrupted speech
  is emitted in chunks while it is still going instead of after it ends.
- `VadDetector::flush` to analyze the last partial frame and close a segment
  that runs to the end of the stream.
- `Error::Unsupported` for recognized but undecodable formats (Opus).
- `cargo bench --bench throughput`, a `process_file` example, and
  `cargo fuzz` targets for decoding and the pipeline.

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
- `VadDetector::detect` keeps samples that don't fill a frame for the next
  call. Before, each call started a new frame grid, so blocks under about 64
  samples never detected speech.
- With `frame_overlap` above 0.5, consecutive speech segments could overlap
  and the chunker emitted the overlapping audio twice. Segments are now
  disjoint, and `Chunker` merges overlapping input segments.
- Decoding no longer returns empty or partial audio without an error for
  chained Ogg streams, or for files where many packets fail to decode. Same-layout chained
  streams decode fully; a sample rate or channel count change is
  `Error::Unsupported`.
- A panic inside Symphonia on malformed input (found by fuzzing: a
  Matroska element with a zero length byte) is returned as
  `Error::InvalidInput` instead of unwinding into the caller.
- `NoiseReductionConfig::validate` rejects a hop shorter than one sample,
  which made `reduce` loop forever.
- `FormatDetector::detect_with_metadata` picks the same track as the
  decoder and recognizes every PCM variant (mu-law, A-law, big-endian).
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
