//! Run the full pipeline on an audio file and print the chunks.
//!
//! ```bash
//! cargo run --release --example process_file -- path/to/recording.wav
//! ```
//!
//! Accepts WAV, FLAC, MP3, M4A/AAC, and Ogg Vorbis.

use speech_prep::{AudioFormatConverter, Pipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: process_file <audio-file>");
        std::process::exit(2);
    };

    let audio = AudioFormatConverter::convert_file(&path)?;
    let meta = audio.metadata;
    println!(
        "{path}: {} {} Hz, {} ch -> {:.2} s at 16 kHz mono",
        meta.original_format,
        meta.original_sample_rate,
        meta.original_channels,
        audio.duration_sec()
    );

    let mut pipeline = Pipeline::new()?;
    let mut chunks = pipeline.process_samples(&audio.samples)?.chunks;
    chunks.extend(pipeline.flush()?.chunks);

    println!(
        "\n{:>9}  {:>9}  {:<8}  {:>6}  {:>7}",
        "start", "end", "kind", "rms", "snr dB"
    );
    for chunk in &chunks {
        let kind = if chunk.is_silence() {
            "silence"
        } else if chunk.is_speech() {
            "speech"
        } else {
            "mixed"
        };
        let snr = chunk
            .snr_db
            .map_or_else(|| "-".to_string(), |db| format!("{db:.1}"));
        println!(
            "{:>9}  {:>9}  {kind:<8}  {:>6.3}  {snr:>7}",
            chunk.start_time.to_string(),
            chunk.end_time.to_string(),
            chunk.energy,
        );
    }

    let speech_secs: f64 = chunks
        .iter()
        .filter(|c| c.is_speech())
        .filter_map(|c| c.duration().ok())
        .map(|d| d.as_secs_f64())
        .fold(0.0, |total, secs| total + secs);
    println!("\n{} chunks, {speech_secs:.2} s of speech", chunks.len());
    Ok(())
}
