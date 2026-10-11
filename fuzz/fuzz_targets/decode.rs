//! Decoding arbitrary bytes must return `Ok` or `Err`, never panic or hang.
#![no_main]

use libfuzzer_sys::fuzz_target;
use speech_prep::AudioFormatConverter;

fuzz_target!(
    init: {
        // libfuzzer-sys aborts on any panic, including ones speech-prep
        // catches from Symphonia. Use the default hook; panics that escape
        // still abort at the FFI boundary and are reported.
        let _ = std::panic::take_hook();
    },
    |data: &[u8]| {
    if let Ok(audio) = AudioFormatConverter::convert_to_standard(data) {
        assert!(audio.samples.iter().all(|s| s.is_finite()));
    }
    }
);
