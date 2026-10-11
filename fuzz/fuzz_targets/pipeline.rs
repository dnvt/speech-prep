//! Stream arbitrary samples through the pipeline in arbitrary block sizes.
#![no_main]

use libfuzzer_sys::fuzz_target;
use speech_prep::Pipeline;

fuzz_target!(
    init: {
        // libfuzzer-sys aborts on any panic, including ones speech-prep
        // catches from Symphonia. Use the default hook; panics that escape
        // still abort at the FFI boundary and are reported.
        let _ = std::panic::take_hook();
    },
    |data: &[u8]| {
    let Some((&block_byte, rest)) = data.split_first() else {
        return;
    };
    let block = usize::from(block_byte) * 8 + 1;
    let samples: Vec<f32> = rest
        .chunks_exact(2)
        .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32_768.0)
        .collect();

    let mut pipeline = Pipeline::new().unwrap();
    let mut emitted = 0;
    for chunk in samples.chunks(block) {
        for c in pipeline.process_samples(chunk).unwrap().chunks {
            emitted += c.samples.len();
        }
    }
    for c in pipeline.flush().unwrap().chunks {
        emitted += c.samples.len();
    }
    // Every input sample comes out exactly once (chunk edges may round by a
    // sample).
    assert!(emitted.abs_diff(samples.len()) <= samples.len() / 1_000 + 8);
    }
);
