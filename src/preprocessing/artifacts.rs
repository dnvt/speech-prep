//! QA artifact helpers for preprocessing pipeline.
//! Allows optional writing of before/after audio chunks to disk for manual
//! review.

use std::fmt;
use std::fs::{create_dir_all, File};
use std::io::BufWriter;
use std::path::Path;

use hound::{SampleFormat, WavSpec, WavWriter};

/// Simple scoped WAV writer (16-bit mono) for QA artifacts.
pub struct WavArtifactWriter {
    writer: WavWriter<BufWriter<File>>,
}

impl fmt::Debug for WavArtifactWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WavArtifactWriter").finish()
    }
}

impl WavArtifactWriter {
    /// Create a new WAV artifact writer at the provided path.
    pub fn create<P: AsRef<Path>>(path: P, sample_rate: u32) -> hound::Result<Self> {
        if let Some(parent) = path.as_ref().parent() {
            let _ = create_dir_all(parent);
        }
        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let writer = WavWriter::create(path, spec)?;
        Ok(Self { writer })
    }

    /// Append samples to the artifact file, clamping to 16-bit PCM range.
    pub fn write_samples(&mut self, samples: &[f32]) -> hound::Result<()> {
        for &sample in samples {
            let clamped = sample.clamp(-1.0, 1.0);
            let scaled = (clamped * f32::from(i16::MAX)) as i16;
            self.writer.write_sample(scaled)?;
        }
        Ok(())
    }

    /// Finalize and flush the WAV artifact to disk.
    pub fn finalize(self) -> hound::Result<()> {
        self.writer.finalize()
    }
}
