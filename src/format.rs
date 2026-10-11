//! Identify the codec and properties of encoded audio.
//!
//! [`FormatDetector::detect`] probes the container with Symphonia and reports
//! the codec of the first audio track, plus channels, sample rate, bit depth,
//! and duration where the container records them. The format is named after
//! the codec, not the container: AAC and ALAC both live in `.m4a` files, and
//! WAV can hold PCM or ADPCM.
//!
//! Detection does not guarantee decoding: Opus is always recognized, but
//! only decoded with the `opus` feature.

use std::io::Cursor;

use symphonia::core::codecs::audio::AudioCodecId;
use symphonia::core::codecs::audio::well_known::{
    CODEC_ID_AAC, CODEC_ID_ALAC, CODEC_ID_FLAC, CODEC_ID_MP3, CODEC_ID_OPUS, CODEC_ID_VORBIS,
};
use symphonia::core::formats::FormatOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

use crate::error::{Error, Result};

/// Audio codec of a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AudioFormat {
    /// Uncompressed PCM, including mu-law and A-law, in WAV, AIFF, or CAF.
    WavPcm,
    /// ADPCM (Microsoft or IMA), usually in WAV.
    Adpcm,
    /// FLAC.
    Flac,
    /// ALAC (Apple Lossless), usually in MP4/M4A or CAF.
    Alac,
    /// MPEG-1/2 Layer III.
    Mp3,
    /// AAC-LC, usually in MP4/M4A.
    Aac,
    /// Vorbis, in Ogg or WebM.
    Vorbis,
    /// Opus, in Ogg or WebM. Decoded with the `opus` feature.
    Opus,
}

impl AudioFormat {
    /// Short lowercase name, e.g. `"wav"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WavPcm => "wav",
            Self::Adpcm => "adpcm",
            Self::Flac => "flac",
            Self::Alac => "alac",
            Self::Mp3 => "mp3",
            Self::Aac => "aac",
            Self::Vorbis => "vorbis",
            Self::Opus => "opus",
        }
    }

    /// Whether the codec stores audio without loss.
    #[must_use]
    pub const fn is_lossless(self) -> bool {
        matches!(self, Self::WavPcm | Self::Flac | Self::Alac)
    }

    /// Map a Symphonia codec to a format, or `None` if this crate does not
    /// know it.
    pub(crate) fn from_codec(codec: AudioCodecId) -> Option<Self> {
        let short_name = crate::decoder::codecs()
            .get_audio_decoder(codec)
            .map(|decoder| decoder.codec.info.short_name)
            .unwrap_or_default();

        Some(match codec {
            _ if short_name.starts_with("pcm") => Self::WavPcm,
            _ if short_name.starts_with("adpcm") => Self::Adpcm,
            CODEC_ID_FLAC => Self::Flac,
            CODEC_ID_ALAC => Self::Alac,
            CODEC_ID_MP3 => Self::Mp3,
            CODEC_ID_AAC => Self::Aac,
            CODEC_ID_VORBIS => Self::Vorbis,
            CODEC_ID_OPUS => Self::Opus,
            _ => return None,
        })
    }
}

impl std::fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Codec plus whatever properties the container exposes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioMetadata {
    /// Codec of the first audio track.
    pub format: AudioFormat,
    /// Channel count, if known.
    pub channels: Option<u16>,
    /// Sample rate in Hz, if known.
    pub sample_rate: Option<u32>,
    /// Bits per sample, for PCM-like codecs.
    pub bit_depth: Option<u16>,
    /// Duration in seconds, if the container records it.
    pub duration_sec: Option<f64>,
}

/// Identifies the codec of encoded audio without decoding it.
#[derive(Debug, Default, Clone, Copy)]
pub struct FormatDetector;

impl FormatDetector {
    /// Probe `data` and describe its first audio track.
    ///
    /// ```
    /// use speech_prep::format::{AudioFormat, FormatDetector};
    ///
    /// # fn main() -> speech_prep::Result<()> {
    /// let wav = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tone.flac"))?;
    /// let meta = FormatDetector::detect(&wav)?;
    /// assert_eq!(meta.format, AudioFormat::Flac);
    /// assert_eq!(meta.sample_rate, Some(44_100));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if `data` is not recognizable audio or
    /// has no audio track, and [`Error::Unsupported`] for an unknown codec.
    pub fn detect(data: &[u8]) -> Result<AudioMetadata> {
        crate::decoder::catch_decoder_panic(|| Self::probe(data))
    }

    fn probe(data: &[u8]) -> Result<AudioMetadata> {
        let source = MediaSourceStream::new(
            Box::new(Cursor::new(data)),
            MediaSourceStreamOptions::default(),
        );

        let reader = symphonia::default::get_probe()
            .probe(
                &Hint::new(),
                source,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(|err| {
                Error::InvalidInput(format!("unrecognized or malformed audio: {err}"))
            })?;

        let (track_id, params) = crate::decoder::first_audio_track(reader.as_ref())
            .ok_or_else(|| Error::InvalidInput("no audio track found".into()))?;
        let format = AudioFormat::from_codec(params.codec)
            .ok_or_else(|| Error::Unsupported(format!("codec {}", params.codec)))?;

        let sample_rate = params.sample_rate;
        let num_frames = reader
            .tracks()
            .iter()
            .find(|t| t.id == track_id)
            .and_then(|t| t.num_frames);

        Ok(AudioMetadata {
            format,
            channels: params
                .channels
                .as_ref()
                .and_then(|ch| u16::try_from(ch.count()).ok()),
            sample_rate,
            bit_depth: params.bits_per_sample.and_then(|b| u16::try_from(b).ok()),
            duration_sec: num_frames
                .zip(sample_rate)
                .map(|(frames, rate)| frames as f64 / f64::from(rate)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_short_and_unknown_input() {
        assert!(FormatDetector::detect(&[]).is_err());
        assert!(FormatDetector::detect(&[0xFF, 0xFE]).is_err());
        assert!(FormatDetector::detect(&[0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE]).is_err());
        assert!(FormatDetector::detect(b"RIFF\x24\x00\x00\x00AVI ").is_err());
    }

    #[test]
    fn display_matches_as_str() {
        for format in [
            AudioFormat::WavPcm,
            AudioFormat::Adpcm,
            AudioFormat::Flac,
            AudioFormat::Alac,
            AudioFormat::Mp3,
            AudioFormat::Aac,
            AudioFormat::Vorbis,
            AudioFormat::Opus,
        ] {
            assert_eq!(format.to_string(), format.as_str());
        }
    }

    #[test]
    fn lossless_formats() {
        assert!(AudioFormat::WavPcm.is_lossless());
        assert!(AudioFormat::Flac.is_lossless());
        assert!(AudioFormat::Alac.is_lossless());
        assert!(!AudioFormat::Mp3.is_lossless());
        assert!(!AudioFormat::Adpcm.is_lossless());
    }
}
