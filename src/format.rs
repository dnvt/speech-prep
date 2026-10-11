//! Audio container and codec detection.
//!
//! [`FormatDetector::detect`] checks magic bytes first (WAV, FLAC, MP3,
//! WebM/Matroska, MP4/M4A) and falls back to a Symphonia probe for anything
//! else, such as Ogg, where the codec decides the format.
//!
//! Detection does not guarantee decoding: Opus is recognized but has no
//! decoder in this crate.

use std::io::Cursor;

use symphonia::core::codecs::{
    CODEC_TYPE_AAC, CODEC_TYPE_FLAC, CODEC_TYPE_MP3, CODEC_TYPE_NULL, CODEC_TYPE_OPUS,
    CODEC_TYPE_VORBIS,
};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::error::{Error, Result};

/// Detected audio format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AudioFormat {
    /// RIFF WAV with PCM or IEEE float samples.
    WavPcm,
    /// FLAC.
    Flac,
    /// MPEG-1/2 Layer III.
    Mp3,
    /// Opus in Ogg. Detected but not decodable.
    Opus,
    /// Vorbis in Ogg.
    Vorbis,
    /// WebM/Matroska. Decodable when the audio codec is supported.
    WebM,
    /// AAC, usually in an MP4/M4A container.
    Aac,
}

impl AudioFormat {
    /// Short lowercase name, e.g. `"wav"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WavPcm => "wav",
            Self::Flac => "flac",
            Self::Mp3 => "mp3",
            Self::Opus => "opus",
            Self::Vorbis => "vorbis",
            Self::WebM => "webm",
            Self::Aac => "aac",
        }
    }

    /// Whether the format stores audio without lossy compression.
    #[must_use]
    pub const fn is_lossless(self) -> bool {
        matches!(self, Self::WavPcm | Self::Flac)
    }

    /// File extension to hint the decoder with.
    pub(crate) fn extension(self) -> &'static str {
        match self {
            Self::WavPcm => "wav",
            Self::Flac => "flac",
            Self::Mp3 => "mp3",
            Self::Opus | Self::Vorbis => "ogg",
            Self::WebM => "webm",
            Self::Aac => "m4a",
        }
    }
}

impl std::fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Format plus whatever properties the header exposed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioMetadata {
    /// Detected format.
    pub format: AudioFormat,
    /// Channel count, if known.
    pub channels: Option<u16>,
    /// Sample rate in Hz, if known.
    pub sample_rate: Option<u32>,
    /// Bits per sample, for PCM-like formats.
    pub bit_depth: Option<u16>,
    /// Duration in seconds, if the container records it.
    pub duration_sec: Option<f64>,
}

impl AudioMetadata {
    /// Metadata with only the format filled in.
    #[must_use]
    pub const fn format_only(format: AudioFormat) -> Self {
        Self {
            format,
            channels: None,
            sample_rate: None,
            bit_depth: None,
            duration_sec: None,
        }
    }
}

/// Identifies audio formats from raw bytes.
#[derive(Debug, Default, Clone, Copy)]
pub struct FormatDetector;

impl FormatDetector {
    /// Identify the format, using magic bytes when possible.
    ///
    /// Only [`AudioMetadata::format`] is guaranteed to be set. Use
    /// [`detect_with_metadata`](Self::detect_with_metadata) for channels,
    /// sample rate, and duration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if `data` is shorter than 4 bytes or is
    /// not recognizable audio, and [`Error::Unsupported`] for a recognized
    /// container with an unknown codec.
    pub fn detect(data: &[u8]) -> Result<AudioMetadata> {
        if data.len() < 4 {
            return Err(Error::InvalidInput(
                "audio payload too short (minimum 4 bytes)".into(),
            ));
        }

        if let Some(format) = Self::detect_magic_bytes(data) {
            return Ok(AudioMetadata::format_only(format));
        }

        Self::detect_with_metadata(data)
    }

    /// Identify the format by probing the container with Symphonia, and fill
    /// in channels, sample rate, bit depth, and duration where available.
    ///
    /// # Errors
    ///
    /// Same as [`detect`](Self::detect).
    pub fn detect_with_metadata(data: &[u8]) -> Result<AudioMetadata> {
        let source = MediaSourceStream::new(
            Box::new(Cursor::new(data.to_vec())),
            MediaSourceStreamOptions::default(),
        );

        let probed = symphonia::default::get_probe()
            .format(
                &Hint::new(),
                source,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|err| {
                Error::InvalidInput(format!("unrecognized or malformed audio: {err}"))
            })?;

        // Same track choice as the decoder: the first track with a codec.
        let params = &probed
            .format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| Error::InvalidInput("no audio track found".into()))?
            .codec_params;

        let is_pcm = symphonia::default::get_codecs()
            .get_codec(params.codec)
            .is_some_and(|codec| codec.short_name.starts_with("pcm"));
        let format = match params.codec {
            _ if is_pcm => AudioFormat::WavPcm,
            CODEC_TYPE_FLAC => AudioFormat::Flac,
            CODEC_TYPE_MP3 => AudioFormat::Mp3,
            CODEC_TYPE_OPUS => AudioFormat::Opus,
            CODEC_TYPE_VORBIS => AudioFormat::Vorbis,
            CODEC_TYPE_AAC => AudioFormat::Aac,
            other => {
                return Err(Error::Unsupported(format!("codec {other:?}")));
            }
        };

        let sample_rate = params.sample_rate;
        Ok(AudioMetadata {
            format,
            channels: params
                .channels
                .and_then(|ch| u16::try_from(ch.count()).ok()),
            sample_rate,
            bit_depth: params.bits_per_sample.and_then(|b| u16::try_from(b).ok()),
            duration_sec: params
                .n_frames
                .zip(sample_rate)
                .map(|(frames, rate)| frames as f64 / f64::from(rate)),
        })
    }

    /// Recognize formats with fixed signatures. Returns `None` when the
    /// container must be probed (Ogg, raw AAC, unknown data).
    fn detect_magic_bytes(data: &[u8]) -> Option<AudioFormat> {
        let at = |range: std::ops::Range<usize>| data.get(range);

        if at(0..4) == Some(b"RIFF") && at(8..12) == Some(b"WAVE") {
            return Some(AudioFormat::WavPcm);
        }
        if at(0..4) == Some(b"fLaC") {
            return Some(AudioFormat::Flac);
        }
        if at(0..3) == Some(b"ID3") {
            return Some(AudioFormat::Mp3);
        }
        // MPEG audio frame sync with layer bits = Layer III.
        if let (Some(&b0), Some(&b1)) = (data.first(), data.get(1)) {
            if b0 == 0xFF && (b1 & 0xE0) == 0xE0 && ((b1 >> 1) & 0x03) == 0x01 {
                return Some(AudioFormat::Mp3);
            }
        }
        if at(0..4) == Some(&[0x1A, 0x45, 0xDF, 0xA3]) {
            return Some(AudioFormat::WebM);
        }
        if at(4..8) == Some(b"ftyp") {
            let brand = at(8..12);
            if [b"M4A ", b"M4B ", b"mp42", b"mp41", b"isom"]
                .iter()
                .any(|b| brand == Some(&b[..]))
            {
                return Some(AudioFormat::Aac);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult<T> = std::result::Result<T, String>;

    fn detect_format(data: &[u8]) -> TestResult<AudioMetadata> {
        FormatDetector::detect(data).map_err(|e| e.to_string())
    }

    // Magic-byte test fixtures
    fn wav_header() -> Vec<u8> {
        // Minimal valid WAV header: RIFF + size + WAVE + fmt chunk
        let mut header = Vec::new();
        header.extend_from_slice(b"RIFF");
        header.extend_from_slice(&36u32.to_le_bytes()); // File size - 8
        header.extend_from_slice(b"WAVE");
        header.extend_from_slice(b"fmt ");
        header.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1 size
        header.extend_from_slice(&1u16.to_le_bytes()); // Audio format (PCM)
        header.extend_from_slice(&2u16.to_le_bytes()); // Num channels (stereo)
        header.extend_from_slice(&44100u32.to_le_bytes()); // Sample rate
        header.extend_from_slice(&(44100u32 * 2 * 2).to_le_bytes()); // Byte rate
        header.extend_from_slice(&4u16.to_le_bytes()); // Block align
        header.extend_from_slice(&16u16.to_le_bytes()); // Bits per sample
        header
    }

    fn flac_header() -> Vec<u8> {
        // FLAC stream marker: "fLaC"
        b"fLaC".to_vec()
    }

    fn mp3_header() -> Vec<u8> {
        // MP3 frame sync: 0xFF 0xFB (MPEG-1 Layer 3, no CRC)
        // Frame header format: 11111111 111BBCCD EEEEFFGH IIJJKLMM
        // 0xFF 0xFB = 11111111 11111011
        // Bits: sync(11) + version(11=MPEG-1) + layer(01=Layer3) + CRC(1=no)
        vec![0xFF, 0xFB, 0x90, 0x00] // Minimal valid MP3 frame header
    }

    fn webm_header() -> Vec<u8> {
        // WebM/Matroska EBML header
        vec![0x1A, 0x45, 0xDF, 0xA3, 0x00, 0x00, 0x00, 0x20]
    }

    fn aac_header() -> Vec<u8> {
        // M4A/AAC ftyp box
        let mut header = Vec::new();
        header.extend_from_slice(&20u32.to_be_bytes()); // Box size
        header.extend_from_slice(b"ftyp"); // Box type
        header.extend_from_slice(b"M4A "); // Major brand
        header.extend_from_slice(&0u32.to_be_bytes()); // Minor version
        header.extend_from_slice(b"mp42"); // Compatible brand
        header
    }

    // Positive path tests
    #[test]
    fn test_detect_wav_format() -> TestResult<()> {
        let metadata = detect_format(&wav_header())?;
        assert_eq!(metadata.format, AudioFormat::WavPcm);
        assert_eq!(metadata.format.as_str(), "wav");
        assert!(metadata.format.is_lossless());
        Ok(())
    }

    #[test]
    fn test_detect_flac_format() -> TestResult<()> {
        let metadata = detect_format(&flac_header())?;
        assert_eq!(metadata.format, AudioFormat::Flac);
        assert_eq!(metadata.format.as_str(), "flac");
        assert!(metadata.format.is_lossless());
        Ok(())
    }

    #[test]
    fn test_detect_mp3_format() -> TestResult<()> {
        let metadata = detect_format(&mp3_header())?;
        assert_eq!(metadata.format, AudioFormat::Mp3);
        assert_eq!(metadata.format.as_str(), "mp3");
        assert!(!metadata.format.is_lossless());
        Ok(())
    }

    #[test]
    fn test_detect_webm_format() -> TestResult<()> {
        let metadata = detect_format(&webm_header())?;
        assert_eq!(metadata.format, AudioFormat::WebM);
        assert_eq!(metadata.format.as_str(), "webm");
        Ok(())
    }

    #[test]
    fn test_detect_aac_format() -> TestResult<()> {
        let metadata = detect_format(&aac_header())?;
        assert_eq!(metadata.format, AudioFormat::Aac);
        assert_eq!(metadata.format.as_str(), "aac");
        assert!(!metadata.format.is_lossless());
        Ok(())
    }

    // Negative path tests
    #[test]
    fn test_reject_empty_payload() {
        let result = FormatDetector::detect(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_too_short_payload() {
        let result = FormatDetector::detect(&[0xFF, 0xFE]); // Only 2 bytes
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_random_bytes() {
        let random_data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];
        let result = FormatDetector::detect(&random_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_truncated_wav_header() {
        let truncated = b"RIFF".to_vec(); // Missing size + WAVE
        let result = FormatDetector::detect(&truncated);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_mismatched_riff_signature() {
        let mut bad_wav = Vec::new();
        bad_wav.extend_from_slice(b"RIFF");
        bad_wav.extend_from_slice(&36u32.to_le_bytes());
        bad_wav.extend_from_slice(b"AVI "); // Wrong signature (should be WAVE)
        let result = FormatDetector::detect(&bad_wav);
        assert!(result.is_err());
    }

    // Edge case tests
    #[test]
    fn test_handle_exact_minimum_length() -> TestResult<()> {
        let flac_minimal = b"fLaC".to_vec(); // Exactly 4 bytes
        let metadata = detect_format(&flac_minimal)?;
        assert_eq!(metadata.format, AudioFormat::Flac);
        Ok(())
    }

    #[test]
    fn test_handle_large_payload_prefix() -> TestResult<()> {
        let mut large_payload = wav_header();
        large_payload.extend(vec![0u8; 1024 * 1024]); // 1MB of silence
        let metadata = detect_format(&large_payload)?;
        assert_eq!(metadata.format, AudioFormat::WavPcm);
        Ok(())
    }

    // Property tests
    #[test]
    fn test_format_display_matches_as_str() {
        let formats = [
            AudioFormat::WavPcm,
            AudioFormat::Flac,
            AudioFormat::Mp3,
            AudioFormat::Opus,
            AudioFormat::Vorbis,
            AudioFormat::WebM,
            AudioFormat::Aac,
        ];
        for format in &formats {
            assert_eq!(format.to_string(), format.as_str());
        }
    }

    #[test]
    fn test_lossless_formats_identified() {
        assert!(AudioFormat::WavPcm.is_lossless());
        assert!(AudioFormat::Flac.is_lossless());
        assert!(!AudioFormat::Mp3.is_lossless());
        assert!(!AudioFormat::Opus.is_lossless());
        assert!(!AudioFormat::Aac.is_lossless());
    }

    #[test]
    fn test_metadata_format_only_constructor() {
        let metadata = AudioMetadata::format_only(AudioFormat::Mp3);
        assert_eq!(metadata.format, AudioFormat::Mp3);
        assert_eq!(metadata.channels, None);
        assert_eq!(metadata.sample_rate, None);
        assert_eq!(metadata.bit_depth, None);
        assert_eq!(metadata.duration_sec, None);
    }
}
