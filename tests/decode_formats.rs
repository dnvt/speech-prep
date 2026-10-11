//! Decode each supported container/codec from a small generated fixture.
//! See `tests/fixtures/generate.sh` for how the files were made.

use speech_prep::format::AudioFormat;
use speech_prep::{AudioFormatConverter, Error};

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn assert_decodes(name: &str, format: AudioFormat) {
    let audio =
        AudioFormatConverter::convert_file(fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let meta = audio.metadata;

    assert_eq!(meta.original_format, format, "{name}");
    assert_eq!(meta.original_sample_rate, 44_100, "{name}");
    assert_eq!(meta.original_channels, 2, "{name}");

    // 0.5 s of audio; lossy codecs add encoder padding.
    let duration = audio.duration_sec();
    assert!((0.45..0.65).contains(&duration), "{name}: {duration} s");

    // ffmpeg's sine source peaks at 1/8, halved by the generator: 0.0625.
    // Lossy codecs overshoot a little.
    let peak = meta.peak_before;
    assert!((0.05..0.12).contains(&peak), "{name}: peak {peak}");
    assert!(!audio.is_silent(), "{name}");
}

#[test]
fn flac() {
    assert_decodes("tone.flac", AudioFormat::Flac);
}

#[test]
fn mp3() {
    assert_decodes("tone.mp3", AudioFormat::Mp3);
}

#[test]
fn aac_in_m4a() {
    assert_decodes("tone.m4a", AudioFormat::Aac);
}

#[test]
fn vorbis_in_ogg() {
    assert_decodes("tone.ogg", AudioFormat::Vorbis);
}

#[test]
fn opus_is_reported_as_unsupported() {
    let err = AudioFormatConverter::convert_file(fixture("tone.opus")).unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
}
