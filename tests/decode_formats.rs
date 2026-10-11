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

#[test]
fn chained_ogg_with_same_layout_decodes_both_streams() {
    let one = std::fs::read(fixture("tone.ogg")).unwrap();
    let single = AudioFormatConverter::convert_to_standard(&one).unwrap();

    let chained = [one.as_slice(), one.as_slice()].concat();
    let both = AudioFormatConverter::convert_to_standard(&chained).unwrap();
    assert_eq!(both.samples.len(), 2 * single.samples.len());
}

#[test]
fn malformed_matroska_is_an_error_not_a_panic() {
    // EBML magic followed by a zero length byte, found by fuzzing. Symphonia
    // 0.5 overflows a shift on it.
    let bytes = [
        0x1a, 0x45, 0xdf, 0xa3, 0x00, 0x16, 0x0f, 0x01, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55,
        0x55,
    ];
    assert!(AudioFormatConverter::convert_to_standard(&bytes).is_err());
    assert!(speech_prep::format::FormatDetector::detect_with_metadata(&bytes).is_err());
}

#[test]
fn layout_change_mid_stream_is_an_error() {
    let read = |name| std::fs::read(fixture(name)).unwrap();

    // Two chained Ogg streams at 44.1 and 22.05 kHz.
    let chained = [read("tone.ogg"), read("tone-22k.ogg")].concat();
    let err = AudioFormatConverter::convert_to_standard(&chained).unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");

    // A stereo MP3 followed by a mono one: half the frames fail to decode.
    let mixed = [read("tone.mp3"), read("tone-mono.mp3")].concat();
    assert!(AudioFormatConverter::convert_to_standard(&mixed).is_err());
}

#[test]
fn metadata_detection_agrees_with_decoder() {
    use speech_prep::format::FormatDetector;
    for (name, format, rate) in [
        ("tone.flac", AudioFormat::Flac, 44_100),
        ("tone.mp3", AudioFormat::Mp3, 44_100),
        ("tone.m4a", AudioFormat::Aac, 44_100),
        ("tone.ogg", AudioFormat::Vorbis, 44_100),
        ("tone.opus", AudioFormat::Opus, 48_000),
    ] {
        let bytes = std::fs::read(fixture(name)).unwrap();
        let meta = FormatDetector::detect_with_metadata(&bytes).unwrap();
        assert_eq!(meta.format, format, "{name}");
        assert_eq!(meta.sample_rate, Some(rate), "{name}");
    }
}
