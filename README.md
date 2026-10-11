# speech-prep

[![crates.io](https://img.shields.io/crates/v/speech-prep.svg)](https://crates.io/crates/speech-prep)
[![docs.rs](https://docs.rs/speech-prep/badge.svg)](https://docs.rs/speech-prep)
[![CI](https://github.com/dnvt/speech-prep/actions/workflows/ci.yml/badge.svg)](https://github.com/dnvt/speech-prep/actions/workflows/ci.yml)

Audio preprocessing for speech in pure Rust. It turns a recording or a live
stream into clean 16 kHz mono chunks that line up with speech, ready to feed
a speech model or a scoring step.

```text
file / stream ─► decode ─► 16 kHz mono ─► VAD ─► chunk ─► DC + high-pass ─► denoise ─► chunks
```

It doesn't do speech recognition, speaker diarization, or source
separation.

```bash
cargo add speech-prep
```

## Quick start

```rust,no_run
use speech_prep::Pipeline;

fn main() -> speech_prep::Result<()> {
    let bytes = std::fs::read("recording.mp3")?;

    let mut pipeline = Pipeline::new()?;
    let mut chunks = pipeline.process_bytes(&bytes)?.chunks;
    chunks.extend(pipeline.flush()?.chunks);

    for chunk in chunks.iter().filter(|c| c.is_speech()) {
        println!("{} - {}  {} samples", chunk.start_time, chunk.end_time, chunk.samples.len());
    }
    Ok(())
}
```

For live audio, call `pipeline.process_samples(&block)` with 16 kHz mono
`f32` blocks of any size, then `flush()` at the end of the stream. A speech
segment is held back until it ends, so a chunk is never split across calls.

Try it on a file:

```bash
cargo run --release --example process_file -- path/to/recording.wav
```

## Using the stages directly

Each stage works on its own:

```rust,no_run
use speech_prep::{AudioFormatConverter, Chunker, VadConfig, VadDetector};

fn main() -> speech_prep::Result<()> {
    let audio = AudioFormatConverter::convert_file("recording.flac")?; // 16 kHz mono
    let segments = VadDetector::new(VadConfig::default())?.detect(&audio.samples)?;
    let chunks = Chunker::default().chunk(&audio.samples, 16_000, &segments)?;
    println!("{} speech segments, {} chunks", segments.len(), chunks.len());
    Ok(())
}
```

| Module | What it does |
|--------|--------------|
| `converter` | Decode, downmix, and resample to 16 kHz mono `f32` |
| `vad` | Voice activity detection from frame energy and spectral flux, with adaptive thresholds |
| `chunker` | Split audio into ~500 ms chunks aligned to speech, with overlap context |
| `preprocessing` | DC removal, Butterworth high-pass, spectral-subtraction denoising, RMS normalization, quality metrics |
| `pipeline` | All of the above, streaming |

## Supported input

| Format | Decoded | Notes |
|--------|:-------:|-------|
| WAV | yes | 8/16/24/32-bit integer and 32/64-bit float PCM |
| FLAC | yes | |
| MP3 | yes | |
| AAC (MP4/M4A) | yes | AAC-LC |
| Ogg Vorbis | yes | |
| WebM / Matroska | partial | Vorbis audio only |
| Opus | no | Detected, then rejected with `Error::Unsupported` |

Decoding uses [Symphonia](https://github.com/pdeljanov/Symphonia). Any
channel count is averaged to mono.

## Limitations

- **Resampling** is linear interpolation with no anti-aliasing filter. That
  is fine for speech recorded at 44.1/48 kHz, but use a dedicated resampler
  such as [`rubato`](https://crates.io/crates/rubato) and feed the pipeline
  16 kHz samples if you need a clean high band.
- **VAD** is signal-based. It is fast and needs no model, but it can trigger
  on non-speech sounds like music or typing. Tune it with `VadConfig`
  (`activation_margin`, `min_speech_energy`, `hangover_frames`).
- **Noise reduction** handles steady noise (fans, hum, hiss). It won't
  remove other voices, music, or reverb.
- **The pipeline** runs at 16 kHz. `process_bytes` decodes the whole input
  in memory, and audio inside an open speech segment is buffered until the
  segment ends, so a very long uninterrupted segment means a long delay.

## Performance

Everything runs on the calling thread with no model inference, so all
stages are much faster than real time. `cargo bench --bench throughput`
prints per-stage timings for 10 s of synthetic audio on your machine.

## Minimum Rust version

1.87. The crate uses edition 2024, so Cargo picks dependency versions that
build on your compiler.

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
To report a security problem, see [SECURITY.md](SECURITY.md).

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or
[MIT](LICENSE-MIT), at your option.
