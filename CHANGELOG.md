# Changelog

## 0.1.4 - 2026-04-05

Current public release.

- Voice activity detection with energy and spectral-flux signals
- WAV/PCM conversion to mono 16kHz `f32` samples
- DC removal, high-pass filtering, normalization, stationary noise reduction,
  and quality checks
- Speech-aligned chunking with overlap metadata
- Synthetic fixture support for tests and examples

## Notes

The high-level converter decodes WAV today. Other container formats may be
detected, but they are not decoded by the high-level converter yet.
