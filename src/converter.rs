//! Decode audio files to 16 kHz mono `f32` samples.
//!
//! ```text
//! bytes ─► detect format ─► decode (Symphonia) ─► mix to mono ─► resample to 16 kHz
//! ```
//!
//! Supported inputs: PCM (any bit depth, float, mu-law, A-law) and ADPCM in
//! WAV, AIFF, or CAF; FLAC; ALAC; MP3; AAC-LC in MP4/M4A; Vorbis in Ogg or
//! WebM. Opus is detected but rejected with [`Error::Unsupported`].
//!
//! Any channel count is accepted and averaged to mono. Resampling uses a
//! windowed-sinc filter that passes up to 7.2 kHz and removes content above
//! 8 kHz instead of letting it alias.

use std::path::Path;
use std::time::Duration;

use crate::decoder::{self, ChannelMixer, SampleRateConverter};
use crate::error::{Error, Result};
use crate::format::AudioFormat;
use crate::time::AudioInstant;

/// Input sample rates the converter accepts. Rates outside this range only
/// occur in corrupt or crafted files, and resampling them would need huge
/// filters (from far above) or huge outputs (from far below).
pub const SUPPORTED_INPUT_RATES: std::ops::RangeInclusive<u32> = 4_000..=768_000;

/// Sample rate of [`StandardAudio`].
pub const STANDARD_SAMPLE_RATE: u32 = SampleRateConverter::TARGET_SAMPLE_RATE;

/// Mono, 16 kHz audio with samples in `[-1.0, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct StandardAudio {
    /// Mono samples at [`STANDARD_SAMPLE_RATE`], always finite and within
    /// `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    /// What the input looked like before conversion.
    pub metadata: ConversionMetadata,
}

impl StandardAudio {
    /// Number of samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Duration in seconds.
    #[must_use]
    pub fn duration_sec(&self) -> f64 {
        self.samples.len() as f64 / f64::from(STANDARD_SAMPLE_RATE)
    }

    /// `true` if every sample is below -80 dBFS.
    #[must_use]
    pub fn is_silent(&self) -> bool {
        self.samples.iter().all(|&s| s.abs() < 1e-4)
    }
}

/// Properties of the source audio and the cost of converting it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConversionMetadata {
    /// Detected input format.
    pub original_format: AudioFormat,
    /// Input sample rate in Hz.
    pub original_sample_rate: u32,
    /// Input channel count.
    pub original_channels: u16,
    /// Input bits per sample, when the codec reports it.
    pub original_bit_depth: Option<u16>,
    /// Peak absolute sample value across all input channels.
    pub peak_before: f32,
    /// Peak absolute sample value of the converted output.
    pub peak_after: f32,
    /// Wall-clock time spent converting.
    pub conversion_time: Duration,
}

impl ConversionMetadata {
    /// `peak_after / peak_before`, or 1.0 for silent input.
    ///
    /// Downmixing out-of-phase channels lowers the peak, so values well
    /// below 1.0 can point at a stereo phase problem.
    #[must_use]
    pub fn peak_ratio(&self) -> f32 {
        if self.peak_before.abs() < f32::EPSILON {
            1.0
        } else {
            self.peak_after / self.peak_before
        }
    }
}

/// Converts encoded audio to [`StandardAudio`].
#[derive(Debug, Default, Clone, Copy)]
pub struct AudioFormatConverter;

impl AudioFormatConverter {
    /// Decode an in-memory audio file.
    ///
    /// ```no_run
    /// use speech_prep::AudioFormatConverter;
    ///
    /// # fn main() -> speech_prep::Result<()> {
    /// let bytes = std::fs::read("recording.mp3")?;
    /// let audio = AudioFormatConverter::convert_to_standard(&bytes)?;
    /// println!("{:.1} s, originally {} Hz", audio.duration_sec(), audio.metadata.original_sample_rate);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidInput`] if the bytes are not recognizable or are
    ///   corrupt.
    /// - [`Error::Unsupported`] if the format is recognized but cannot be
    ///   decoded (Opus, or an unsupported codec inside a container).
    pub fn convert_to_standard(audio_bytes: &[u8]) -> Result<StandardAudio> {
        Self::convert(audio_bytes, None)
    }

    /// Read and decode an audio file from disk.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the file cannot be read, otherwise the same as
    /// [`convert_to_standard`](Self::convert_to_standard).
    pub fn convert_file(path: impl AsRef<Path>) -> Result<StandardAudio> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        Self::convert(&bytes, path.extension().and_then(|ext| ext.to_str()))
    }

    fn convert(audio_bytes: &[u8], extension: Option<&str>) -> Result<StandardAudio> {
        let start = AudioInstant::now();

        let decoded = decoder::decode(audio_bytes, extension)?;
        let format = decoded.format;
        if !SUPPORTED_INPUT_RATES.contains(&decoded.sample_rate) {
            return Err(Error::Unsupported(format!(
                "sample rate {} Hz is outside {}-{} Hz",
                decoded.sample_rate,
                SUPPORTED_INPUT_RATES.start(),
                SUPPORTED_INPUT_RATES.end()
            )));
        }
        if decoded.samples.iter().any(|s| !s.is_finite()) {
            return Err(Error::InvalidInput(
                "decoded audio contains NaN or infinite samples".into(),
            ));
        }
        let peak_before = peak(&decoded.samples);

        let mono = ChannelMixer::mix_to_mono(&decoded.samples, decoded.channels)?;
        let mut samples = SampleRateConverter::resample_to_16khz(&mono, decoded.sample_rate)?;
        // Float input may exceed full scale, and the resampler can overshoot
        // slightly on hard edges.
        for sample in &mut samples {
            *sample = sample.clamp(-1.0, 1.0);
        }

        let metadata = ConversionMetadata {
            original_format: format,
            original_sample_rate: decoded.sample_rate,
            original_channels: decoded.channels,
            original_bit_depth: decoded.bit_depth,
            peak_before,
            peak_after: peak(&samples),
            conversion_time: start.elapsed(),
        };

        tracing::debug!(
            format = %format,
            sample_rate = decoded.sample_rate,
            channels = decoded.channels,
            output_samples = samples.len(),
            elapsed_us = metadata.conversion_time.as_micros(),
            "converted audio to 16 kHz mono"
        );

        Ok(StandardAudio { samples, metadata })
    }
}

fn peak(samples: &[f32]) -> f32 {
    samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let spec = hound::WavSpec {
            sample_rate,
            channels,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = std::io::Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
        cursor.into_inner()
    }

    #[test]
    fn mono_16khz_passes_through() {
        let standard =
            AudioFormatConverter::convert_to_standard(&wav(16_000, 1, &[100, 200, -100, -200]))
                .unwrap();

        assert_eq!(standard.samples.len(), 4);
        assert_eq!(standard.metadata.original_format, AudioFormat::WavPcm);
        assert_eq!(standard.metadata.original_sample_rate, 16_000);
        assert_eq!(standard.metadata.original_channels, 1);
        assert_eq!(standard.metadata.original_bit_depth, Some(16));
    }

    #[test]
    fn stereo_44100_is_mixed_and_resampled() {
        let frames = 44_100;
        let samples: Vec<i16> = (0..frames).flat_map(|_| [8_000i16, 8_000]).collect();
        let standard =
            AudioFormatConverter::convert_to_standard(&wav(44_100, 2, &samples)).unwrap();

        assert_eq!(standard.samples.len(), 16_000);
        assert_eq!(standard.metadata.original_channels, 2);
        assert!((standard.duration_sec() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn tracks_peaks() {
        let standard =
            AudioFormatConverter::convert_to_standard(&wav(16_000, 1, &[10_000, -10_000, 5_000]))
                .unwrap();

        let expected = 10_000.0 / 32_768.0;
        assert!((standard.metadata.peak_before - expected).abs() < 1e-4);
        assert!((standard.metadata.peak_ratio() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn opposite_phase_stereo_cancels() {
        let standard = AudioFormatConverter::convert_to_standard(&wav(
            16_000,
            2,
            &[10_000, -10_000, 5_000, -5_000],
        ))
        .unwrap();
        assert!(standard.is_silent());
        assert!(standard.metadata.peak_ratio() < 0.01);
    }

    #[test]
    fn rejects_bytes_that_only_look_like_mp3() {
        let result = AudioFormatConverter::convert_to_standard(&[0xFF, 0xFB, 0x90, 0x00]);
        assert!(result.is_err());
    }

    fn float_wav(samples: &[f32]) -> Vec<u8> {
        let spec = hound::WavSpec {
            sample_rate: 16_000,
            channels: 1,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut cursor = std::io::Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
        cursor.into_inner()
    }

    #[test]
    fn output_is_clamped_to_full_scale() {
        let standard =
            AudioFormatConverter::convert_to_standard(&float_wav(&[2.0, -3.0, 0.5])).unwrap();
        assert_eq!(standard.samples, vec![1.0, -1.0, 0.5]);
        assert_eq!(standard.metadata.peak_before, 3.0);
        assert_eq!(standard.metadata.peak_after, 1.0);
    }

    #[test]
    fn rejects_non_finite_samples() {
        for bad in [f32::NAN, f32::INFINITY] {
            let err =
                AudioFormatConverter::convert_to_standard(&float_wav(&[0.1, bad])).unwrap_err();
            assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
        }
    }

    #[test]
    fn resampled_output_stays_in_range() {
        // A full-scale square wave makes the sinc filter ring past ±1.
        let square: Vec<i16> = (0..44_100)
            .map(|i| {
                if (i / 50) % 2 == 0 {
                    i16::MAX
                } else {
                    i16::MIN
                }
            })
            .collect();
        let standard = AudioFormatConverter::convert_to_standard(&wav(44_100, 1, &square)).unwrap();
        assert!(standard.samples.iter().all(|s| (-1.0..=1.0).contains(s)));
    }

    #[test]
    fn rejects_implausible_sample_rates() {
        for rate in [1_000, 1_040_220_288] {
            let err =
                AudioFormatConverter::convert_to_standard(&wav(rate, 1, &[0; 16])).unwrap_err();
            assert!(matches!(err, Error::Unsupported(_)), "{rate} Hz: {err:?}");
        }
    }

    #[test]
    fn convert_file_reports_io_errors() {
        let err = AudioFormatConverter::convert_file("/nonexistent/file.wav").unwrap_err();
        assert!(matches!(err, Error::Io(_)));
    }
}
