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
`f32` blocks of any size, then `flush()` at the end of the stream. Speech is
held back until the speaker pauses, or for at most 2 s
(`with_max_speech_hold`), so chunks line up with speech without unbounded
delay.

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
    let vad = VadDetector::new(VadConfig::default())?;
    let mut segments = vad.detect(&audio.samples)?;
    segments.extend(vad.flush()?); // close a segment that runs to the end
    let chunks = Chunker::default().chunk(&audio.samples, 16_000, &segments)?;
    println!("{} speech segments, {} chunks", segments.len(), chunks.len());
    Ok(())
}
```

| Module | What it does |
|--------|--------------|
| `converter` | Decode, downmix, and resample to 16 kHz mono `f32` |
| `vad` | Voice activity detection from frame energy against a tracked noise floor |
| `chunker` | Split audio into ~500 ms chunks aligned to speech, with overlap context |
| `preprocessing` | DC removal, Butterworth high-pass, spectral-subtraction denoising, RMS normalization, quality metrics |
| `pipeline` | All of the above, streaming |

## Supported input

| Codec | Containers | Decoded |
|-------|------------|:-------:|
| PCM (integer, float) | WAV, AIFF, CAF | yes |
| PCM (mu-law, A-law) | WAV, AIFF | yes |
| ADPCM (Microsoft, IMA) | WAV | yes |
| FLAC | FLAC, Ogg | yes |
| ALAC | MP4/M4A, CAF | yes |
| MP3 | MP3 | yes |
| AAC-LC | MP4/M4A | yes |
| Vorbis | Ogg, WebM | yes |
| Opus | Ogg, WebM | with the `opus` feature |

Opus support is opt-in because its pure-Rust decoder adds about 20
dependencies and needs Rust 1.95:

```toml
speech-prep = { version = "0.2", features = ["opus"] }
```

Decoding uses [Symphonia](https://github.com/pdeljanov/Symphonia). Any
channel count is averaged to mono, and sample rates from 4 to 768 kHz are
accepted.

## Limitations

- **Resampling** uses a windowed-sinc low-pass filter. For inputs above 16 kHz,
  the response is flat to about 6.6 kHz and rolls off from there
  (about -3 dB at 7.1 kHz, -6 dB at 7.2 kHz), with everything near and above
  the 8 kHz output Nyquist strongly attenuated. The 16 kHz output keeps the
  speech band but not the top of the spectrum. Inputs already at 16 kHz
  pass through without filtering. Lower-rate inputs are band-limited at
  90% of their own Nyquist frequency.
- **VAD** is signal-based. It is fast and needs no model, but it can trigger
  on non-speech sounds like music or typing. Tune it with `VadConfig`
  (`activation_ratio`, `hangover_frames`, `speech_padding`).
- **Noise reduction** handles steady noise (fans, hum, hiss). It won't
  remove other voices, music, or reverb.
- **The pipeline** runs at 16 kHz, and `process_bytes` decodes its whole
  input in memory.

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

[MIT](LICENSE)
