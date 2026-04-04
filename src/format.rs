//! Audio format detection and metadata extraction for the audio pipeline.
//!
//! This module provides fast, robust format detection using a hybrid approach:
//! - **Fast path**: Magic-byte detection for common formats (WAV, FLAC, MP3) -
//!   <1µs
//! - **Validation path**: Symphonia probe for complex formats (Opus, `WebM`,
//!   M4A) - <10ms
//!
//! ## Performance Contract
//!
//! - Detection latency: <1ms for 99% of inputs
//! - Total processing: <10ms including validation
//! - Zero panics: All byte access bounds-checked
//!
//! ## Supported Formats
//!
//! - **WAV** (RIFF/PCM): Primary format, instant detection
//! - **FLAC**: Lossless compression, instant detection
//! - **MP3**: MPEG-1/2 Layer 3, frame sync validation
//! - **Opus**: Ogg container with Opus codec
//! - **`WebM`**: Matroska container (audio track)
//! - **M4A/AAC**: MPEG-4 container with AAC codec

use std::io::Cursor;

use crate::error::{Error, Result};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;

/// Audio container and codec format identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioFormat {
    /// RIFF WAV container with PCM encoding.
    WavPcm,
    /// Free Lossless Audio Codec.
    Flac,
    /// MPEG-1/2 Audio Layer 3.
    Mp3,
    /// Opus codec in Ogg container.
    Opus,
    /// `WebM` container (Matroska subset) with audio track.
    WebM,
    /// MPEG-4 container with AAC codec.
    Aac,
}

impl AudioFormat {
    /// Human-readable format name for logging and metrics.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WavPcm => "wav",
            Self::Flac => "flac",
            Self::Mp3 => "mp3",
            Self::Opus => "opus",
            Self::WebM => "webm",
            Self::Aac => "aac",
        }
    }

    /// Whether this format is lossless.
    #[must_use]
    pub const fn is_lossless(self) -> bool {
        matches!(self, Self::WavPcm | Self::Flac)
    }

    /// Whether this format requires container parsing (vs raw frames).
    #[must_use]
    pub const fn is_container_format(self) -> bool {
        matches!(self, Self::WavPcm | Self::Opus | Self::WebM | Self::Aac)
    }
}

impl std::fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Audio metadata extracted during format detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioMetadata {
    /// Detected container/codec format.
    pub format: AudioFormat,
    /// Number of audio channels (if determinable from header).
    pub channels: Option<u16>,
    /// Sample rate in Hz (if determinable from header).
    pub sample_rate: Option<u32>,
    /// Bit depth (if applicable for PCM formats).
    pub bit_depth: Option<u16>,
    /// Total duration in seconds (if available in container).
    pub duration_sec: Option<f64>,
}

impl AudioMetadata {
    /// Create metadata with only format known (minimal detection).
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

    /// Create metadata with format and basic audio properties.
    #[must_use]
    pub const fn with_properties(
        format: AudioFormat,
        channels: u16,
        sample_rate: u32,
        bit_depth: Option<u16>,
    ) -> Self {
        Self {
            format,
            channels: Some(channels),
            sample_rate: Some(sample_rate),
            bit_depth,
            duration_sec: None,
        }
    }
}

/// Audio format detector using hybrid magic-byte + Symphonia validation.
#[derive(Debug, Default, Clone, Copy)]
pub struct FormatDetector;

impl FormatDetector {
    /// Create a new format detector instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Detect audio format from byte stream using fast magic-byte detection.
    ///
    /// This is the primary entry point optimized for speed (<1µs for common
    /// formats). Falls back to Symphonia validation for complex/ambiguous
    /// formats.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if:
    /// - Payload is too short for any valid audio format
    /// - Format is unsupported or unrecognized
    /// - Byte stream is malformed (detected via Symphonia probe)
    pub fn detect(data: &[u8]) -> Result<AudioMetadata> {
        if data.len() < 4 {
            return Err(Error::InvalidInput(
                "audio payload too short (minimum 4 bytes required)".into(),
            ));
        }

        if let Some(format) = Self::detect_magic_bytes(data) {
            return Ok(AudioMetadata::format_only(format));
        }

        Self::detect_with_symphonia(data)
    }

    /// Detect format and extract full metadata using Symphonia probe.
    ///
    /// This method provides comprehensive metadata extraction but is slower
    /// (~10ms). Use when full audio properties are needed (channels, sample
    /// rate, duration).
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if format cannot be determined.
    pub fn detect_with_metadata(data: &[u8]) -> Result<AudioMetadata> {
        Self::detect_with_symphonia(data)
    }

    /// Fast magic-byte detection for common formats.
    ///
    /// Returns `Some(AudioFormat)` if format is recognized via magic bytes,
    /// `None` if validation via Symphonia is needed.
    fn detect_magic_bytes(data: &[u8]) -> Option<AudioFormat> {
        let len = data.len();

        // WAV: RIFF + size + WAVE
        if len >= 12 {
            if let (Some(riff), Some(wave)) = (data.get(0..4), data.get(8..12)) {
                if riff == b"RIFF" && wave == b"WAVE" {
                    return Some(AudioFormat::WavPcm);
                }
            }
        }

        // FLAC
        if len >= 4 {
            if let Some(header) = data.get(0..4) {
                if header == b"fLaC" {
                    return Some(AudioFormat::Flac);
                }
            }
        }

        // MP3: frame sync heuristic, validated by Symphonia downstream
        if len >= 2 {
            if let (Some(&first), Some(&second)) = (data.first(), data.get(1)) {
                if first == 0xFF && (second & 0xE0) == 0xE0 {
                    let layer = (second >> 1) & 0x03;
                    if layer == 0x01 {
                        return Some(AudioFormat::Mp3);
                    }
                }
            }
        }

        // Ogg: needs Symphonia to distinguish Opus from Vorbis
        if len >= 4 {
            if let Some(header) = data.get(0..4) {
                if header == b"OggS" {
                    return None;
                }
            }
        }

        // WebM/Matroska: EBML header
        if len >= 4 {
            if let Some(header) = data.get(0..4) {
                if header == [0x1A, 0x45, 0xDF, 0xA3] {
                    return Some(AudioFormat::WebM);
                }
            }
        }

        // M4A/AAC: ftyp box
        if len >= 12 {
            if let (Some(ftyp), Some(brand)) = (data.get(4..8), data.get(8..12)) {
                if ftyp == b"ftyp" && (brand == b"M4A " || brand == b"mp42" || brand == b"isom") {
                    return Some(AudioFormat::Aac);
                }
            }
        }

        None
    }

    /// Detect format using Symphonia's comprehensive probe.
    ///
    /// This provides robust format validation and metadata extraction.
    fn detect_with_symphonia(data: &[u8]) -> Result<AudioMetadata> {
        let data_vec = data.to_vec();
        let cursor = Cursor::new(data_vec);
        let mss = MediaSourceStream::new(
            Box::new(cursor),
            symphonia::core::io::MediaSourceStreamOptions::default(),
        );

        let hint = Hint::new();
        let probe_result = symphonia::default::get_probe()
            .format(
                &hint,
                mss,
                &symphonia::core::formats::FormatOptions::default(),
                &symphonia::core::meta::MetadataOptions::default(),
            )
            .map_err(|err| {
                Error::InvalidInput(format!("unsupported or malformed audio format: {err}"))
            })?;

        let format_reader = probe_result.format;
        let codec_params = &format_reader
            .default_track()
            .ok_or_else(|| Error::InvalidInput("no audio track found in container".into()))?
            .codec_params;

        let format = match codec_params.codec {
            symphonia::core::codecs::CODEC_TYPE_PCM_S16LE
            | symphonia::core::codecs::CODEC_TYPE_PCM_S24LE
            | symphonia::core::codecs::CODEC_TYPE_PCM_S32LE
            | symphonia::core::codecs::CODEC_TYPE_PCM_F32LE => AudioFormat::WavPcm,
            symphonia::core::codecs::CODEC_TYPE_FLAC => AudioFormat::Flac,
            symphonia::core::codecs::CODEC_TYPE_MP3 => AudioFormat::Mp3,
            symphonia::core::codecs::CODEC_TYPE_OPUS => AudioFormat::Opus,
            symphonia::core::codecs::CODEC_TYPE_VORBIS => {
                return Err(Error::InvalidInput(
                    "Vorbis codec not supported (use Opus instead)".into(),
                ));
            }
            symphonia::core::codecs::CODEC_TYPE_AAC => AudioFormat::Aac,
            _ => {
                return Err(Error::InvalidInput(format!(
                    "unsupported codec: {:?}",
                    codec_params.codec
                )));
            }
        };

        let channels = codec_params.channels.map(|ch| ch.count() as u16);
        let sample_rate = codec_params.sample_rate;
        let bit_depth = codec_params.bits_per_sample.map(|b| b as u16);
        let duration_sec = codec_params
            .n_frames
            .and_then(|frames| sample_rate.map(|rate| frames as f64 / f64::from(rate)));

        Ok(AudioMetadata {
            format,
            channels,
            sample_rate,
            bit_depth,
            duration_sec,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult<T> = std::result::Result<T, String>;

    fn create_detector() -> FormatDetector {
        FormatDetector::new()
    }

    fn detect_format(_detector: FormatDetector, data: &[u8]) -> TestResult<AudioMetadata> {
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
        let detector = create_detector();
        let metadata = detect_format(detector, &wav_header())?;
        assert_eq!(metadata.format, AudioFormat::WavPcm);
        assert_eq!(metadata.format.as_str(), "wav");
        assert!(metadata.format.is_lossless());
        Ok(())
    }

    #[test]
    fn test_detect_flac_format() -> TestResult<()> {
        let detector = create_detector();
        let metadata = detect_format(detector, &flac_header())?;
        assert_eq!(metadata.format, AudioFormat::Flac);
        assert_eq!(metadata.format.as_str(), "flac");
        assert!(metadata.format.is_lossless());
        Ok(())
    }

    #[test]
    fn test_detect_mp3_format() -> TestResult<()> {
        let detector = create_detector();
        let metadata = detect_format(detector, &mp3_header())?;
        assert_eq!(metadata.format, AudioFormat::Mp3);
        assert_eq!(metadata.format.as_str(), "mp3");
        assert!(!metadata.format.is_lossless());
        Ok(())
    }

    #[test]
    fn test_detect_webm_format() -> TestResult<()> {
        let detector = create_detector();
        let metadata = detect_format(detector, &webm_header())?;
        assert_eq!(metadata.format, AudioFormat::WebM);
        assert_eq!(metadata.format.as_str(), "webm");
        Ok(())
    }

    #[test]
    fn test_detect_aac_format() -> TestResult<()> {
        let detector = create_detector();
        let metadata = detect_format(detector, &aac_header())?;
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
        let detector = create_detector();
        let flac_minimal = b"fLaC".to_vec(); // Exactly 4 bytes
        let metadata = detect_format(detector, &flac_minimal)?;
        assert_eq!(metadata.format, AudioFormat::Flac);
        Ok(())
    }

    #[test]
    fn test_handle_large_payload_prefix() -> TestResult<()> {
        let detector = create_detector();
        let mut large_payload = wav_header();
        large_payload.extend(vec![0u8; 1024 * 1024]); // 1MB of silence
        let metadata = detect_format(detector, &large_payload)?;
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
    fn test_container_formats_identified() {
        assert!(AudioFormat::WavPcm.is_container_format());
        assert!(AudioFormat::Opus.is_container_format());
        assert!(AudioFormat::WebM.is_container_format());
        assert!(AudioFormat::Aac.is_container_format());
        assert!(!AudioFormat::Flac.is_container_format());
        assert!(!AudioFormat::Mp3.is_container_format());
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

    #[test]
    fn test_metadata_with_properties_constructor() {
        let metadata = AudioMetadata::with_properties(AudioFormat::WavPcm, 2, 44100, Some(16));
        assert_eq!(metadata.format, AudioFormat::WavPcm);
        assert_eq!(metadata.channels, Some(2));
        assert_eq!(metadata.sample_rate, Some(44100));
        assert_eq!(metadata.bit_depth, Some(16));
        assert_eq!(metadata.duration_sec, None);
    }
}
