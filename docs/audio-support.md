# Audio Support

This crate focuses on local speech preprocessing before model inference or
scoring. The goal is a small, inspectable Rust path, not a full media pipeline.

## Current Support Matrix

| Area | Status | Notes |
|------|--------|-------|
| WAV / PCM decoding | Supported | High-level converter path decodes WAV/PCM into mono `f32` samples. |
| Format detection | Partial | Common containers can be identified, but detection does not mean decoding is implemented. |
| MP3 / AAC / Ogg / FLAC | Detection only | Do not claim high-level conversion support until there is a tested decode path. |
| Resampling | Basic | Linear interpolation, intended for simple preprocessing, not mastering-quality SRC. |
| Mono conversion | Supported for decoded WAV | Multi-channel input is mixed down for speech processing. |
| VAD | Supported | Energy and spectral-flux based; useful for segmentation, not speaker diarization. |
| Noise reduction | Basic | Stationary noise reduction for preprocessing, not a restoration tool. |
| Chunking | Supported | Speech-aligned chunks with overlap metadata. |

## Assumptions

- Speech-oriented audio, not music mastering.
- Local files or in-memory samples.
- Small fixtures for tests.
- Public or synthetic audio for issues and examples.

## What This Is Not

- A full ASR frontend.
- A certified audio quality analyzer.
- A replacement for specialized media tools such as FFmpeg.
- A privacy boundary for user audio.

## Adding Format Support

When adding or documenting a format, include:

- a small public or generated fixture;
- a test that decodes it;
- sample rate and channel behavior;
- expected output type;
- limitations in the README and this file.
