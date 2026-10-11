//! Rough per-stage timings on synthetic audio.
//!
//! ```bash
//! cargo bench --bench throughput
//! ```
//!
//! Prints the median time to process 10 s of 16 kHz audio and the resulting
//! real-time factor (processing time / audio duration). Numbers depend on
//! the machine and build; this is a local profiling aid, not a guarantee.

use std::f32::consts::PI;
use std::hint::black_box;
use std::io::Cursor;
use std::time::{Duration, Instant};

use speech_prep::preprocessing::{
    DcHighPassFilter, NoiseReducer, NoiseReductionConfig, Normalizer, PreprocessingConfig,
    QualityAssessor, VadContext,
};
use speech_prep::{AudioFormatConverter, Chunker, Pipeline, VadConfig, VadDetector};

const SAMPLE_RATE: u32 = 16_000;
const SECONDS: usize = 10;
const RUNS: usize = 15;

/// Alternating 1 s of tone and 1 s of low-level noise.
fn synthetic_speech() -> Vec<f32> {
    let mut state = 0x1234_5678u32;
    (0..SAMPLE_RATE as usize * SECONDS)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = (state as f32 / u32::MAX as f32 - 0.5) * 0.01;
            let t = i as f32 / SAMPLE_RATE as f32;
            let voiced = (i / SAMPLE_RATE as usize) % 2 == 1;
            noise
                + if voiced {
                    0.4 * (2.0 * PI * 180.0 * t).sin()
                } else {
                    0.0
                }
        })
        .collect()
}

fn stereo_wav_44k(seconds: usize) -> Vec<u8> {
    let spec = hound::WavSpec {
        sample_rate: 44_100,
        channels: 2,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
    for i in 0..44_100 * seconds {
        let s = ((2.0 * PI * 220.0 * i as f32 / 44_100.0).sin() * 10_000.0) as i16;
        writer.write_sample(s).unwrap();
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();
    cursor.into_inner()
}

fn bench(name: &str, mut f: impl FnMut()) {
    f(); // warm-up
    let mut times: Vec<Duration> = (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .collect();
    times.sort();
    let median = times[RUNS / 2];
    let rtf = median.as_secs_f64() / SECONDS as f64;
    println!(
        "{name:<22} {:>9.2} ms   RTF {rtf:.4}",
        median.as_secs_f64() * 1e3
    );
}

fn main() {
    let audio = synthetic_speech();
    let wav = stereo_wav_44k(SECONDS);
    println!("{SECONDS} s of audio, median of {RUNS} runs\n");

    bench("decode 44.1k stereo", || {
        black_box(AudioFormatConverter::convert_to_standard(&wav).unwrap());
    });

    bench("vad", || {
        let vad = VadDetector::new(VadConfig::default()).unwrap();
        black_box(vad.detect(&audio).unwrap());
    });

    let segments = VadDetector::new(VadConfig::default())
        .unwrap()
        .detect(&audio)
        .unwrap();
    bench("chunker", || {
        black_box(
            Chunker::default()
                .chunk(&audio, SAMPLE_RATE, &segments)
                .unwrap(),
        );
    });

    bench("dc + high-pass", || {
        let mut filter = DcHighPassFilter::new(PreprocessingConfig::default()).unwrap();
        black_box(filter.process(&audio, None).unwrap());
    });

    bench("noise reduction", || {
        let mut reducer = NoiseReducer::new(NoiseReductionConfig::default()).unwrap();
        let ctx = Some(VadContext { is_silence: false });
        black_box(reducer.reduce(&audio, ctx).unwrap());
    });

    bench("normalization", || {
        black_box(
            Normalizer::new(0.1, 10.0)
                .unwrap()
                .normalize(&audio)
                .unwrap(),
        );
    });

    bench("quality metrics", || {
        black_box(QualityAssessor::new(SAMPLE_RATE).assess(&audio).unwrap());
    });

    bench("pipeline (samples)", || {
        let mut pipeline = Pipeline::new().unwrap();
        black_box(pipeline.process_samples(&audio).unwrap());
        black_box(pipeline.flush().unwrap());
    });
}
