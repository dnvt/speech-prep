# Changelog

## 0.1.0 — 2026-04-04

Initial release.

- Voice activity detection (dual-metric: energy + spectral flux)
- Multi-format audio decoding (WAV/MP3/FLAC/OGG/M4A/Opus → 16kHz mono PCM)
- Preprocessing: DC removal, high-pass filter, noise reduction, normalization
- Speech-aligned chunking with configurable duration and overlap
- Audio quality assessment metrics
- Performance contract test suite
