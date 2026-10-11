# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- Resampler response documentation now distinguishes downsampling, lower-rate
  input filtering, and unfiltered 16 kHz passthrough; the codec support table
  no longer claims mu-law or A-law decoding in CAF.
- The noise floor's slow rise during speech now has the same time constant
  (5 s) at any frame hop. It was a fixed per-frame factor, so with
  `frame_overlap: 0.75` steady noise under speech was absorbed in half the
  time, and with no overlap in twice the time. Defaults are unchanged.
- `VadConfig::validate` rejects an infinite `min_speech_energy`, which passed
  the sign check and silently turned detection off.
- `PreprocessingConfig::validate` rejects a non-finite high-pass cutoff or
  `dc_bias_alpha`, which previously passed and turned the filter output into
  `NaN`.
- A `ChunkerConfig` built as a struct literal with a sub-millisecond
  `target_duration` is now `Error::Configuration` from `Chunker::chunk`
  instead of a divide-by-zero panic.
- `NoiseReducer` now carries the last frame of the stream as real left-hand
  context and keeps its STFT frames on one hop grid across calls, so a
  continuous signal split across calls (at any sample, in blocks of any size)
  no longer has a discontinuity at the leading edge of each chunk once a noise
  profile is learned (previously each call zero-padded its own left edge and
  restarted its frame grid). The trailing
  edge of a call still lacks future context and is re-covered on the next
  call; eliminating that residual entirely would require an output-lagging
  (buffered) API, which is a separate change.
- A truncated stream (cut mid-packet) is now rejected with `InvalidInput`
  instead of returning the partial audio decoded so far as `Ok`.
- Symphonia 0.6.1 or newer is required so a malformed metadata length cannot
  trigger an allocation for a payload that is absent from the input.
- The decoder caps how many samples one decode may buffer (64 Mi interleaved,
  256 MiB), and how many 16 kHz samples it will resample to, so a
  decompression bomb — a tiny compressed file declaring hours of audio — is
  refused instead of allocating gigabytes.

## [0.2.0] - 2026-10-11

### Upgrading from 0.1

| 0.1 | 0.2 |
| --- | --- |
| `AudioPipelineCoordinator::new_with_defaults()` | `Pipeline::new()?` |
| `AudioPipelineCoordinator::new(Arc::new(vad), chunker, dc, nr)` | `Pipeline::with_components(vad, chunker, dc, nr)?` |
| `coordinator.process_frame(&bytes)` | `pipeline.process_bytes(&bytes)` (`&mut self`; `.chunks` holds the cleaned chunks) |
| `VadDetector::new(config, metrics)` | `VadDetector::new(config)` or `VadDetector::with_metrics(config, metrics)` |
| `VADStats` | `vad::VadStats` |
| `VadConfig { activation_margin, release_margin, energy_weight, flux_weight, .. }` | `VadConfig { activation_ratio, release_ratio, .. }` |
| `chunk.overlap_next` | the next chunk's `overlap_prev`, or `chunk.samples_with_context()` |
| `AudioFormat::WebM` | `AudioFormat::Vorbis` or `AudioFormat::Opus` |
| `AudioBuffer`, `AudioChunk`, the `fixtures` feature | removed |

`Error` and `AudioFormat` are `#[non_exhaustive]`, so a `match` on them
needs a wildcard arm. Opus needs `features = ["opus"]` (Rust 1.95).

### Added

- Decoding of FLAC, ALAC, MP3, AAC (MP4/M4A), Ogg/WebM Vorbis, AIFF, CAF,
  and ADPCM through Symphonia 0.6, plus 8-bit, 32-bit, float, mu-law, and
  A-law WAV. Any channel count is downmixed to mono.
- `AudioFormatConverter::convert_file` for reading from a path.
- `Pipeline::process_samples` for already-decoded 16 kHz audio, and
  `Pipeline::reset`.
- `VadConfig::min_speech_energy` to tune the absolute level below which a
  frame is never speech.
- `VadDetector::with_metrics` for attaching a metrics collector.
- `Pipeline::with_max_speech_hold` (default 2 s): long uninterrupted speech
  is emitted in chunks while it is still going instead of after it ends.
- `ProcessedChunk::samples_with_context`: the previous chunk's tail followed
  by this chunk's samples. In the pipeline, the context also carries across
  `process_samples` calls.
- `VadDetector::flush` to analyze the last partial frame and close a segment
  that runs to the end of the stream.
- `Error::Unsupported` for recognized formats that can't be decoded, such
  as Opus without the `opus` feature (the message names the feature).
- Opus decoding (Ogg and WebM) behind the optional `opus` feature, using the
  pure-Rust `ruopus` decoder. Its output matches libopus to within float
  rounding (121 dB SNR on speech). It is off by default because it adds
  about 20 crates and needs Rust 1.95.
- `cargo bench --bench throughput`, a `process_file` example, and
  `cargo fuzz` targets for decoding and the pipeline.

### Changed

- **Breaking:** `AudioPipelineCoordinator` is now `Pipeline`. Its methods
  take `&mut self` and return the cleaned `ProcessedChunk`s instead of only
  a count. `process_frame` is now `process_bytes`.
- **Breaking:** `VadDetector::new` takes only a `VadConfig`.
- **Breaking:** `VADStats` is now `vad::VadStats`.
- **Breaking:** the VAD decides on frame energy relative to a tracked noise
  floor and nothing else. The spectral-flux score, combined threshold and
  their settings never changed a decision once the relative energy gate was
  in place, so they are gone: `VadConfig` loses `energy_smoothing`
  (now `noise_smoothing`), `flux_smoothing`, `flux_floor`,
  `threshold_smoothing`, `activation_margin`, `release_margin`,
  `base_threshold`, `energy_weight`, `flux_weight` and `fft_size()`, and
  gains `activation_ratio` (default 2.5) and `release_ratio` (default 1.3).
  The hysteresis between them raises F1 on real speech from 0.946 to 0.958
  and splits speech in heavy noise into fewer pieces.
  `VadMetricsSnapshot::adaptive` and `AdaptiveThresholdSnapshot` are
  replaced by `VadMetricsSnapshot::noise_floor`.
- `SpeechChunk::confidence` is now how far the segment's speech rises above
  the noise floor (0 at the floor, 1 at 20 dB above). It used to be 1.0 for
  almost every segment.
- **Breaking:** `ConversionMetadata` reports a single `conversion_time` and
  `original_channels` is a `u16`.
- **Breaking:** `AudioFormat` names the codec, not the container, and is
  `#[non_exhaustive]`: `WebM` is gone (a WebM file reports `Vorbis` or
  `Opus`), `Alac` and `Adpcm` are new. `FormatDetector::detect` always
  probes the stream and fills in all metadata;
  `FormatDetector::detect_with_metadata`, `AudioMetadata::format_only`, and
  `AudioMetadata::with_properties`
  are removed.
- Decoding no longer copies the input buffer.
- **Breaking:** `DcHighPassFilter::process` takes `Option<VadContext>` by
  value, like `NoiseReducer::reduce`.
- **Breaking:** configuration validation errors are `Error::Configuration`
  everywhere; `Error` is `#[non_exhaustive]`.
- The VAD noise floor is seeded from the quietest frame of the first
  300 ms and keeps adapting during speech. Before, a recording that started
  with noise above the energy gate was classified as one long speech
  segment.
- The default `hangover_frames` is 20 (200 ms) instead of 3 (30 ms; the
  docs said 60 ms), and hangover frames no longer extend a segment's end.
  Segments get 50 ms of padding on each side (`VadConfig::speech_padding`,
  new). On real speech in three languages, scored against Silero VAD and a
  human annotation, a sentence is no longer split at every gap between
  words (e.g. 6 segments instead of 56 on a 30 s meeting clip) and F1 goes
  from 0.89 to 0.95.
- `VadStats::speech_frames` counts frames that measured as speech; hangover
  frames now count as silence.
- Because the noise floor now adapts, the default VAD energy gate is 40x
  lower (0.0005 instead of 0.02). The old value missed 60% of speech
  peaking at -6 dBFS and all speech at -20 dBFS in a synthesized test set.
- Channels are mixed to mono before resampling instead of after.
- Resampling uses a Kaiser-windowed sinc filter instead of linear
  interpolation. A 12 kHz tone at 48 kHz used to alias to 4 kHz at full
  level; it is now attenuated by about 100 dB.
- **Licensing:** the crate is now MIT only (0.1.x was MIT OR Apache-2.0).
- Edition 2024. Cargo now resolves dependencies against `rust-version`, so
  1.87 builds without pinning anything.
- Library code no longer logs at `info`/`warn` on every call or warns when a
  stage takes longer than a fixed wall-clock budget.

### Fixed

- The pipeline no longer emits the start of a speech segment that is still
  in progress as a silence chunk.
- Pipeline speech chunks emitted from a window with no silence of its own
  (long speech held past `max_speech_hold`, or the end of a stream) had no
  `snr_db`. They now use the level of the last silence in the stream.
- `NoiseReducer` no longer fades each chunk in from near zero at its first
  samples (about -23 dB on a steady tone with 500 ms chunks).
- The fourth-order high-pass filter now has Butterworth stage Q values; the
  cutoff is at -3 dB instead of -6 dB.
- `VadStats::speech_frames` counted segments instead of frames, and
  `silence_frames` was always zero.
- `QualityMetrics::spectral_centroid` is now computed from the spectrum. It
  used to be a weighted time index, not a frequency.
- `VadDetector::reset` now also resets the noise floor.
- `VadDetector::detect` keeps samples that don't fill a frame for the next
  call. Before, each call started a new frame grid, so blocks under about 64
  samples never detected speech.
- With `frame_overlap` above 0.5, consecutive speech segments could overlap
  and the chunker emitted the overlapping audio twice. Segments are now
  disjoint, and `Chunker` merges overlapping input segments.
- Decoding no longer returns empty or partial audio without an error for
  chained Ogg streams, or for files where many packets fail to decode.
  Same-layout chained streams decode fully; a sample rate or channel count
  change is `Error::Unsupported`.
- A panic inside Symphonia on malformed input (found by fuzzing: a
  Matroska element with a zero length byte) is returned as
  `Error::InvalidInput` instead of unwinding into the caller.
- `StandardAudio` samples are now guaranteed finite and within `[-1, 1]`:
  NaN or infinite input is an error, and louder-than-full-scale float input
  or resampler overshoot is clamped.
- Inputs declaring a sample rate outside 4-768 kHz are rejected with
  `Error::Unsupported`. Fuzzing found a 10 KB file claiming 1.04 GHz that
  made the resampler build a 1.7 GB filter and run for 40 s.
- `NoiseReductionConfig::validate` rejects a hop shorter than one sample,
  which made `reduce` loop forever.
- Symphonia 0.6 fixes an untrusted allocation in the MP4 reader: a 5 KB
  M4A with a corrupt sample table could make 0.5 try to allocate 32 GB and
  abort the process (found by fuzzing).

### Removed

- **Breaking:** `buffer::AudioBuffer`, `types::AudioChunk`, the `fixtures`
  feature and `fixtures` module, `Chunker::default()` as an inherent method
  (use `Default`), `ProcessedChunk::samples_without_overlap`,
  `ProcessedChunk::overlap_next` (it duplicated the next chunk's
  `overlap_prev`), `ProcessedChunk::total_samples_with_overlap`,
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

[Unreleased]: https://github.com/dnvt/speech-prep/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dnvt/speech-prep/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/dnvt/speech-prep/releases/tag/v0.1.4
