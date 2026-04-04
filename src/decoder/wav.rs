use std::io::{Cursor, Read, Seek};

use crate::error::{Error, Result};

use super::DecodedAudio;

/// WAV/PCM audio decoder using the `hound` crate.
///
/// Supports 16-bit and 24-bit PCM with mono or stereo channels and normalizes
/// all samples into the [-1.0, 1.0] range.
#[derive(Debug, Default, Clone, Copy)]
pub struct WavDecoder;

impl WavDecoder {
    /// Create a new WAV decoder instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Decode WAV audio from a byte slice.
    pub fn decode(data: &[u8]) -> Result<DecodedAudio> {
        let cursor = Cursor::new(data);
        Self::decode_from_reader(cursor)
    }

    /// Decode WAV audio from any reader implementing `Read + Seek`.
    pub fn decode_from_reader<R: Read + Seek>(reader: R) -> Result<DecodedAudio> {
        let mut wav_reader = hound::WavReader::new(reader)
            .map_err(|err| Error::InvalidInput(format!("failed to parse WAV header: {err}")))?;

        let spec = wav_reader.spec();

        if spec.sample_format != hound::SampleFormat::Int {
            return Err(Error::InvalidInput(format!(
                "unsupported WAV format: {:?} (only PCM is supported)",
                spec.sample_format
            )));
        }

        if spec.bits_per_sample != 16 && spec.bits_per_sample != 24 {
            return Err(Error::InvalidInput(format!(
                "unsupported bit depth: {} (only 16-bit and 24-bit PCM supported)",
                spec.bits_per_sample
            )));
        }

        if spec.channels > 2 {
            return Err(Error::InvalidInput(format!(
                "unsupported channel count: {} (only mono and stereo supported)",
                spec.channels
            )));
        }

        let samples = match spec.bits_per_sample {
            16 => Self::decode_16bit(&mut wav_reader)?,
            24 => Self::decode_24bit(&mut wav_reader)?,
            _ => {
                return Err(Error::InvalidInput(format!(
                    "internal error: unhandled bit depth {}",
                    spec.bits_per_sample
                )));
            }
        };

        let frame_count = samples.len() / spec.channels as usize;
        let duration_sec = if spec.sample_rate > 0 {
            frame_count as f64 / f64::from(spec.sample_rate)
        } else {
            0.0
        };

        Ok(DecodedAudio {
            samples,
            sample_rate: spec.sample_rate,
            channels: spec.channels as u8,
            bit_depth: spec.bits_per_sample,
            duration_sec,
        })
    }

    fn decode_16bit<R: Read + Seek>(wav_reader: &mut hound::WavReader<R>) -> Result<Vec<f32>> {
        wav_reader
            .samples::<i16>()
            .map(|sample_result| {
                sample_result.map(Self::normalize_i16).map_err(|err| {
                    Error::InvalidInput(format!("failed to read 16-bit sample: {err}"))
                })
            })
            .collect()
    }

    fn decode_24bit<R: Read + Seek>(wav_reader: &mut hound::WavReader<R>) -> Result<Vec<f32>> {
        wav_reader
            .samples::<i32>()
            .map(|sample_result| {
                sample_result.map(Self::normalize_i24).map_err(|err| {
                    Error::InvalidInput(format!("failed to read 24-bit sample: {err}"))
                })
            })
            .collect()
    }

    #[inline]
    fn normalize_i16(sample: i16) -> f32 {
        f32::from(sample) / 32768.0
    }

    #[inline]
    fn normalize_i24(sample: i32) -> f32 {
        (sample as f32) / 8_388_608.0
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    type TestResult<T> = std::result::Result<T, String>;

    #[test]
    fn test_decode_16bit_mono_44100hz() -> TestResult<()> {
        let wav_data = create_wav_header(44100, 1, 16, 4410)?; // 0.1s mono
        let decoded = WavDecoder::decode(&wav_data).map_err(|e| e.to_string())?;

        assert_eq!(decoded.sample_rate, 44100);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.bit_depth, 16);
        assert_eq!(decoded.samples.len(), 4410);
        assert!((decoded.duration_sec - 0.1).abs() < 1e-6);
        assert!(decoded.is_normalized());

        Ok(())
    }

    #[test]
    fn test_decode_16bit_stereo_48000hz() -> TestResult<()> {
        let wav_data = create_wav_header(48000, 2, 16, 9600)?; // 0.1s stereo (4800 frames)
        let decoded = WavDecoder::decode(&wav_data).map_err(|e| e.to_string())?;

        assert_eq!(decoded.sample_rate, 48000);
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.bit_depth, 16);
        assert_eq!(decoded.samples.len(), 9600);
        assert_eq!(decoded.frame_count(), 4800);
        assert!((decoded.duration_sec - 0.1).abs() < 1e-6);
        assert!(decoded.is_normalized());

        Ok(())
    }

    #[test]
    fn test_decode_24bit_mono_96000hz() -> TestResult<()> {
        let wav_data = create_wav_header(96000, 1, 24, 9600)?; // 0.1s mono
        let decoded = WavDecoder::decode(&wav_data).map_err(|e| e.to_string())?;

        assert_eq!(decoded.sample_rate, 96000);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.bit_depth, 24);
        assert_eq!(decoded.samples.len(), 9600);
        assert!((decoded.duration_sec - 0.1).abs() < 1e-6);
        assert!(decoded.is_normalized());

        Ok(())
    }

    #[test]
    fn test_decode_24bit_stereo_192000hz() -> TestResult<()> {
        let wav_data = create_wav_header(192000, 2, 24, 19200)?; // 0.05s stereo
        let decoded = WavDecoder::decode(&wav_data).map_err(|e| e.to_string())?;

        assert_eq!(decoded.sample_rate, 192000);
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.bit_depth, 24);
        assert_eq!(decoded.samples.len(), 19200);
        assert!(decoded.is_normalized());

        Ok(())
    }

    #[test]
    fn test_decode_sine_wave_preserves_amplitude() -> TestResult<()> {
        let wav_data = create_sine_wave_wav(44100, 1, 16, 440.0, 0.1)?;
        let decoded = WavDecoder::decode(&wav_data).map_err(|e| e.to_string())?;

        let max_amplitude = decoded
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        assert!(
            (max_amplitude - 0.8).abs() < 0.05,
            "expected max amplitude ~0.8, got {max_amplitude}"
        );

        Ok(())
    }

    #[test]
    fn test_reject_empty_data() {
        let result = WavDecoder::decode(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_zero_samples() -> TestResult<()> {
        let wav_data = create_wav_header(44_100, 1, 16, 0)?;
        let decoded = WavDecoder::decode(&wav_data).map_err(|e| e.to_string())?;
        assert_eq!(decoded.samples.len(), 0);
        assert_eq!(decoded.frame_count(), 0);
        Ok(())
    }

    #[test]
    fn test_decode_single_sample() -> TestResult<()> {
        let wav_data = create_wav_header(44_100, 1, 16, 1)?;
        let decoded = WavDecoder::decode(&wav_data).map_err(|e| e.to_string())?;
        assert_eq!(decoded.samples.len(), 1);
        assert_eq!(decoded.frame_count(), 1);
        Ok(())
    }

    #[test]
    fn test_normalization_bounds_16bit() {
        let min_i16 = WavDecoder::normalize_i16(i16::MIN);
        let max_i16 = WavDecoder::normalize_i16(i16::MAX);
        let zero = WavDecoder::normalize_i16(0);

        assert!((-1.0..=1.0).contains(&min_i16));
        assert!((-1.0..=1.0).contains(&max_i16));
        assert!(zero.abs() < f32::EPSILON);
    }

    #[test]
    fn test_frame_count_calculation() -> TestResult<()> {
        let mono = create_wav_header(44_100, 1, 16, 4_410)?;
        let decoded_mono = WavDecoder::decode(&mono).map_err(|e| e.to_string())?;
        assert_eq!(decoded_mono.frame_count(), 4_410);

        let stereo = create_wav_header(44_100, 2, 16, 8_820)?;
        let decoded_stereo = WavDecoder::decode(&stereo).map_err(|e| e.to_string())?;
        assert_eq!(decoded_stereo.frame_count(), 4_410);
        Ok(())
    }

    #[test]
    fn test_duration_calculation_accuracy() -> TestResult<()> {
        let wav_data = create_wav_header(48_000, 2, 16, 96_000)?; // 1s stereo
        let decoded = WavDecoder::decode(&wav_data).map_err(|e| e.to_string())?;
        assert!((decoded.duration_sec - 1.0).abs() < 1e-6);
        Ok(())
    }

    fn create_wav_header(
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u16,
        num_samples: usize,
    ) -> TestResult<Vec<u8>> {
        let spec = hound::WavSpec {
            sample_rate,
            channels,
            bits_per_sample,
            sample_format: hound::SampleFormat::Int,
        };

        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec)
                .map_err(|err| format!("failed to create WAV writer: {err}"))?;

            for _ in 0..num_samples {
                match bits_per_sample {
                    16 => writer
                        .write_sample(0i16)
                        .map_err(|err| format!("failed to write 16-bit sample: {err}"))?,
                    24 => writer
                        .write_sample(0i32)
                        .map_err(|err| format!("failed to write 24-bit sample: {err}"))?,
                    _ => {
                        return Err(format!("unsupported bit depth ({bits_per_sample})"));
                    }
                }
            }

            writer
                .finalize()
                .map_err(|err| format!("failed to finalize WAV: {err}"))?;
        }

        Ok(cursor.into_inner())
    }

    fn create_sine_wave_wav(
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u16,
        frequency: f32,
        duration_sec: f32,
    ) -> TestResult<Vec<u8>> {
        let spec = hound::WavSpec {
            sample_rate,
            channels,
            bits_per_sample,
            sample_format: hound::SampleFormat::Int,
        };

        let mut cursor = Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|err| format!("failed to create WAV writer for sine wave: {err}"))?;

        let num_samples = (sample_rate as f32 * duration_sec) as usize;
        let amplitude = match bits_per_sample {
            16 => 32767.0 * 0.8,
            24 => 8_388_607.0 * 0.8,
            _ => return Err(format!("unsupported bit depth ({bits_per_sample})")),
        };

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample_f32 = amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin();

            for _ in 0..channels {
                match bits_per_sample {
                    16 => writer
                        .write_sample(sample_f32 as i16)
                        .map_err(|err| format!("failed to write sine sample: {err}"))?,
                    24 => writer
                        .write_sample(sample_f32 as i32)
                        .map_err(|err| format!("failed to write sine sample: {err}"))?,
                    _ => return Err(format!("unsupported bit depth ({bits_per_sample})")),
                }
            }
        }

        writer
            .finalize()
            .map_err(|err| format!("failed to finalize sine wave WAV: {err}"))?;
        Ok(cursor.into_inner())
    }
}
