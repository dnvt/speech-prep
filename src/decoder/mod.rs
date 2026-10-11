//! Decoding, channel mixing, and resampling to the crate's standard format.

mod mixer;
mod resampler;

use std::io::Cursor;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, Decoder, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub(crate) use mixer::ChannelMixer;
pub(crate) use resampler::SampleRateConverter;

use crate::error::{Error, Result};

/// Decoded audio before mixing and resampling.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DecodedAudio {
    /// Interleaved samples in `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    /// Bits per sample of the source, when the codec reports it.
    pub bit_depth: Option<u16>,
}

/// Decode the first audio track of an in-memory file with Symphonia.
///
/// `extension` is an optional hint such as `"wav"` or `"mp3"`; the container
/// is still probed from the bytes.
pub(crate) fn decode(data: &[u8], extension: Option<&str>) -> Result<DecodedAudio> {
    let source = MediaSourceStream::new(
        Box::new(Cursor::new(data.to_vec())),
        MediaSourceStreamOptions::default(),
    );

    let mut hint = Hint::new();
    if let Some(ext) = extension {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|err| Error::InvalidInput(format!("unrecognized or malformed audio: {err}")))?;
    let mut reader = probed.format;

    let (mut track_id, mut decoder) = open_track(reader.as_ref())?;
    let params = decoder.codec_params().clone();

    let mut sample_rate = params.sample_rate;
    let mut channels = params.channels.map(|c| c.count());
    let mut samples = Vec::new();
    let mut buffer: Option<SampleBuffer<f32>> = None;
    let (mut packets, mut skipped) = (0usize, 0usize);

    loop {
        let packet = match reader.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            // A new logical stream started (e.g. chained Ogg). Reopen it and
            // carry on if it has the same layout.
            Err(SymphoniaError::ResetRequired) => {
                (track_id, decoder) = open_track(reader.as_ref())?;
                let next = decoder.codec_params();
                check_layout(
                    &mut sample_rate,
                    &mut channels,
                    next.sample_rate,
                    next.channels.map(|c| c.count()),
                )?;
                continue;
            }
            Err(err) => return Err(Error::InvalidInput(format!("failed to read audio: {err}"))),
        };
        if packet.track_id() != track_id {
            continue;
        }
        packets += 1;

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // A corrupt packet in an otherwise valid stream; skip it.
            Err(SymphoniaError::DecodeError(_)) => {
                skipped += 1;
                continue;
            }
            Err(err) => {
                return Err(Error::InvalidInput(format!(
                    "failed to decode audio: {err}"
                )));
            }
        };

        let spec = *decoded.spec();
        check_layout(
            &mut sample_rate,
            &mut channels,
            Some(spec.rate),
            Some(spec.channels.count()),
        )?;

        let needed = decoded.capacity() * spec.channels.count();
        let buf = match &mut buffer {
            Some(buf) if buf.capacity() >= needed => buf,
            _ => buffer.insert(SampleBuffer::new(decoded.capacity() as u64, spec)),
        };
        buf.copy_interleaved_ref(decoded);
        samples.extend_from_slice(buf.samples());
    }

    // Tolerate the odd corrupt frame, but not a stream that mostly fails.
    if skipped > 0 {
        if skipped * MAX_SKIPPED_DENOMINATOR > packets {
            return Err(Error::InvalidInput(format!(
                "{skipped} of {packets} audio packets could not be decoded"
            )));
        }
        tracing::warn!(skipped, packets, "skipped undecodable audio packets");
    }

    let sample_rate = sample_rate
        .filter(|&rate| rate > 0)
        .ok_or_else(|| Error::InvalidInput("audio has no sample rate".into()))?;
    let channels = channels
        .filter(|&count| count > 0)
        .ok_or_else(|| Error::InvalidInput("audio has no channels".into()))?;
    let channels =
        u16::try_from(channels).map_err(|_| Error::Unsupported(format!("{channels} channels")))?;

    Ok(DecodedAudio {
        samples,
        sample_rate,
        channels,
        bit_depth: params.bits_per_sample.and_then(|b| u16::try_from(b).ok()),
    })
}

/// More than 1 in this many packets failing to decode is an error.
const MAX_SKIPPED_DENOMINATOR: usize = 10;

/// Pick the first audio track and build a decoder for it.
fn open_track(reader: &dyn FormatReader) -> Result<(u32, Box<dyn Decoder>)> {
    let track = reader
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| Error::InvalidInput("no audio track found".into()))?;

    let decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|_| {
            Error::Unsupported(
                "no decoder for this codec (supported: PCM, FLAC, MP3, AAC-LC, Vorbis)".into(),
            )
        })?;
    Ok((track.id, decoder))
}

/// Record the stream layout on first sight and reject later changes, which
/// would otherwise corrupt the interleaved output.
fn check_layout(
    rate: &mut Option<u32>,
    channels: &mut Option<usize>,
    new_rate: Option<u32>,
    new_channels: Option<usize>,
) -> Result<()> {
    for (known, new, what) in [
        (
            rate.map(|r| r as usize),
            new_rate.map(|r| r as usize),
            "sample rate",
        ),
        (*channels, new_channels, "channel count"),
    ] {
        if let (Some(known), Some(new)) = (known, new) {
            if known != new {
                return Err(Error::Unsupported(format!(
                    "{what} changes partway through the stream ({known} to {new})"
                )));
            }
        }
    }
    if rate.is_none() {
        *rate = new_rate;
    }
    if channels.is_none() {
        *channels = new_channels;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn wav<S: hound::Sample + Copy>(spec: hound::WavSpec, samples: &[S]) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
        cursor.into_inner()
    }

    fn int_spec(sample_rate: u32, channels: u16, bits: u16) -> hound::WavSpec {
        hound::WavSpec {
            sample_rate,
            channels,
            bits_per_sample: bits,
            sample_format: hound::SampleFormat::Int,
        }
    }

    #[test]
    fn decodes_16_bit_stereo() {
        let bytes = wav(int_spec(48_000, 2, 16), &[16_384i16, -16_384, 0, 0]);
        let decoded = decode(&bytes, None).unwrap();

        assert_eq!(decoded.sample_rate, 48_000);
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.bit_depth, Some(16));
        assert_eq!(decoded.samples, vec![0.5, -0.5, 0.0, 0.0]);
    }

    #[test]
    fn decodes_24_bit() {
        let bytes = wav(int_spec(16_000, 1, 24), &[4_194_304i32, -8_388_608]);
        let decoded = decode(&bytes, None).unwrap();
        assert_eq!(decoded.bit_depth, Some(24));
        assert_eq!(decoded.samples, vec![0.5, -1.0]);
    }

    #[test]
    fn decodes_float_wav() {
        let spec = hound::WavSpec {
            sample_rate: 16_000,
            channels: 1,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let decoded = decode(&wav(spec, &[0.25f32, -0.75]), None).unwrap();
        assert_eq!(decoded.samples, vec![0.25, -0.75]);
    }

    #[test]
    fn decodes_six_channels() {
        let bytes = wav(int_spec(16_000, 6, 16), &[0i16; 60]);
        let decoded = decode(&bytes, None).unwrap();
        assert_eq!(decoded.channels, 6);
        assert_eq!(decoded.samples.len(), 60);
    }

    #[test]
    fn empty_wav_has_no_samples() {
        let bytes = wav::<i16>(int_spec(16_000, 1, 16), &[]);
        let decoded = decode(&bytes, None).unwrap();
        assert!(decoded.samples.is_empty());
    }

    #[test]
    fn rejects_garbage() {
        assert!(decode(&[0xDE, 0xAD, 0xBE, 0xEF, 0, 1, 2, 3], None).is_err());
        assert!(decode(&[], None).is_err());
    }
}
