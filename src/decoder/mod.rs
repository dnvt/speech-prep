//! Audio decoding, resampling, and channel mixing utilities.
//!
//! This module exposes the decoding pipeline used by the audio processing pipeline:
//! WAV/PCM parsing (`WavDecoder`), linear resampling
//! (`SampleRateConverter`), and channel mixing (`ChannelMixer`). Samples are
//! normalized to the range [-1.0, 1.0] and converted to mono/16kHz.

mod mixer;
mod resampler;
mod types;
mod wav;

pub use mixer::ChannelMixer;
pub use resampler::SampleRateConverter;
pub use types::{DecodedAudio, MixedAudio};
pub use wav::WavDecoder;
