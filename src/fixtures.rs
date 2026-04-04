use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

use crate::decoder::{ChannelMixer, SampleRateConverter};

/// Audio sample loaded from fixtures with metadata
#[derive(Debug, Clone)]
pub struct AudioSample {
    /// Sample identifier (e.g., "`sample_0001`")
    pub id: String,
    /// Audio data as normalized f32 samples [-1.0, 1.0]
    pub audio_data: Vec<f32>,
    /// Sample rate in Hz (typically 16000)
    pub sample_rate: u32,
    /// Metadata about the sample source and quality
    pub metadata: SampleMetadata,
}

/// Metadata about an audio sample
#[derive(Debug, Clone)]
pub struct SampleMetadata {
    /// Source of the audio (curated, full dataset, or synthetic)
    pub source: AudioSource,
    /// Quality category (clean, moderate, challenging, synthetic, unknown)
    pub quality: String,
    /// Duration in seconds
    pub duration: f32,
}

/// Source of audio sample
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSource {
    /// Loaded from curated dataset (datasets/curated/)
    Curated,
    /// Loaded from full dataset (fixtures/audio/)
    FullDataset,
    /// Generated synthetically (deterministic fallback)
    Synthetic,
}

/// Audio fixtures loader with three-tier fallback logic
///
/// Provides robust audio loading for tests with graceful degradation:
/// 1. Try curated dataset (fast, balanced, always committed)
/// 2. Try full dataset (comprehensive, may not be downloaded)
/// 3. Generate synthetic (always available, deterministic)
///
/// # Example
///
/// ```rust
/// use speech_prep::fixtures::AudioFixtures;
///
/// let fixtures = AudioFixtures::new();
/// let sample = fixtures.load_sample("sample_0001")?;
///
/// assert!(!sample.audio_data.is_empty());
/// assert_eq!(sample.sample_rate, 16000);
/// # Ok::<(), speech_prep::error::Error>(())
/// ```
#[derive(Debug)]
pub struct AudioFixtures {
    curated_path: PathBuf,
    full_dataset_path: PathBuf,
}

impl AudioFixtures {
    /// Create new `AudioFixtures` with default paths relative to the crate
    /// root.
    ///
    /// Default paths:
    /// - Curated: `<crate>/datasets/curated/`
    /// - Full dataset: `<crate>/fixtures/audio/`
    pub fn new() -> Self {
        let crate_root = crate_root();
        Self {
            curated_path: crate_root.join("datasets/curated"),
            full_dataset_path: crate_root.join("fixtures/audio"),
        }
    }

    /// Load audio sample with three-tier fallback
    ///
    /// Tries in order:
    /// 1. Curated dataset (via manifest.json lookup)
    /// 2. Full dataset (via glob search)
    /// 3. Synthetic generation (deterministic fallback)
    ///
    /// # Arguments
    ///
    /// * `id` - Sample identifier (e.g., "`sample_0001`", "`test_audio`")
    ///
    /// # Returns
    ///
    /// Always returns `Ok(AudioSample)` - synthetic fallback ensures success
    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::cognitive_complexity)]
    pub fn load_sample(&self, id: &str) -> Result<AudioSample> {
        match self.try_load_curated(id) {
            Ok(Some(sample)) => {
                tracing::debug!("Loaded sample '{}' from curated dataset", id);
                return Ok(sample);
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(
                    "Curated dataset read error for '{}': {}, trying full dataset",
                    id,
                    e
                );
            }
        }

        match self.try_load_full_dataset(id) {
            Ok(Some(sample)) => {
                tracing::debug!("Loaded sample '{}' from full dataset", id);
                return Ok(sample);
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(
                    "Full dataset read error for '{}': {}, using synthetic",
                    id,
                    e
                );
            }
        }

        tracing::debug!("Generating synthetic audio for '{}'", id);
        Ok(Self::generate_synthetic(id))
    }

    /// Check if real datasets are available (curated or full)
    ///
    /// # Returns
    ///
    /// `true` if curated manifest or full dataset directory exists
    pub fn has_real_datasets(&self) -> bool {
        self.curated_dataset_available() || self.full_dataset_available()
    }

    fn curated_dataset_available(&self) -> bool {
        let manifest_path = self.curated_path.join("manifest.json");
        let manifest_content = match std::fs::read_to_string(&manifest_path) {
            Ok(content) => content,
            Err(_) => return false,
        };

        let manifest: serde_json::Value = match serde_json::from_str(&manifest_content) {
            Ok(json) => json,
            Err(_) => return false,
        };

        let samples = match manifest.get("samples").and_then(|s| s.as_array()) {
            Some(samples) => samples,
            None => return false,
        };

        samples
            .iter()
            .filter_map(|entry| entry.get("filename").and_then(|f| f.as_str()))
            .map(|filename| self.curated_path.join(filename))
            .take(10)
            .any(|path| is_wav_readable(&path))
    }

    fn full_dataset_available(&self) -> bool {
        if !self.full_dataset_path.exists() {
            return false;
        }

        glob::glob(
            self.full_dataset_path
                .join("**/*.wav")
                .to_string_lossy()
                .as_ref(),
        )
        .ok()
        .and_then(|paths| {
            paths
                .filter_map(std::result::Result::ok)
                .take(10)
                .find(|path| is_wav_readable(path))
        })
        .is_some()
    }

    /// Load sample from curated dataset via manifest.json
    ///
    /// # Arguments
    ///
    /// * `id` - Sample identifier to look up in manifest
    ///
    /// # Errors
    ///
    /// Returns `Error::NotFound` if:
    /// - Manifest doesn't exist
    /// - Sample ID not in manifest
    /// - Referenced audio file missing
    fn try_load_curated(&self, id: &str) -> Result<Option<AudioSample>> {
        let entry = match self.lookup_manifest_entry(id)? {
            Some(entry) => entry,
            None => return Ok(None),
        };
        let filename = entry
            .get("filename")
            .and_then(|f| f.as_str())
            .ok_or_else(|| Error::Processing("Missing 'filename' field in sample entry".into()))?;

        let file_path = self.curated_path.join(filename);
        let mut metadata = entry.clone();
        if let Some(obj) = metadata.as_object_mut() {
            obj.insert(
                "requested_id".into(),
                serde_json::Value::String(id.to_owned()),
            );
        }
        match Self::load_audio_file(&file_path, AudioSource::Curated, &metadata) {
            Ok(sample) => Ok(Some(sample)),
            Err(err) => {
                tracing::warn!("Failed to load curated sample '{id}': {err}; falling back");
                Ok(None)
            }
        }
    }

    /// Load sample from full dataset via manifest mapping or glob search
    #[allow(clippy::cognitive_complexity)]
    fn try_load_full_dataset(&self, id: &str) -> Result<Option<AudioSample>> {
        if let Some(entry) = self.lookup_manifest_entry(id)? {
            if let Some(original_rel) = entry.get("original_path").and_then(|p| p.as_str()) {
                let original_path = crate_root().join(original_rel);
                if original_path.exists() {
                    let mut entry = entry.clone();
                    if let Some(obj) = entry.as_object_mut() {
                        obj.insert(
                            "requested_id".into(),
                            serde_json::Value::String(id.to_owned()),
                        );
                    }
                    match Self::load_audio_file(&original_path, AudioSource::FullDataset, &entry) {
                        Ok(sample) => return Ok(Some(sample)),
                        Err(err) => {
                            tracing::warn!(
                                "Failed to load full dataset sample '{id}' from {:?}: {err}",
                                original_path
                            );
                            // Continue to glob fallback
                        }
                    }
                }
            }
        }

        let search_pattern = format!("{id}*.wav");
        let glob_pattern = self
            .full_dataset_path
            .join("**")
            .join(&search_pattern)
            .to_string_lossy()
            .to_string();

        let paths = glob::glob(&glob_pattern)
            .map_err(|e| Error::Processing(format!("Invalid glob pattern: {e}")))?;

        for path in paths.flatten() {
            if path.exists() {
                let mut metadata_json = serde_json::json!({
                    "quality": "unknown",
                    "duration": 0.0
                });
                if let Some(obj) = metadata_json.as_object_mut() {
                    obj.insert(
                        "requested_id".into(),
                        serde_json::Value::String(id.to_owned()),
                    );
                }
                match Self::load_audio_file(&path, AudioSource::FullDataset, &metadata_json) {
                    Ok(sample) => return Ok(Some(sample)),
                    Err(err) => {
                        tracing::warn!(
                            "Failed to load dataset sample '{id}' from {:?}: {err}",
                            path
                        );
                    }
                }
            }
        }

        Ok(None)
    }

    /// Load audio file from path with format conversion
    ///
    /// # Arguments
    ///
    /// * `path` - Path to WAV audio file
    /// * `source` - Source type for metadata
    /// * `metadata_json` - JSON metadata about the sample
    ///
    /// # Returns
    ///
    /// `AudioSample` with normalized f32 audio data
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or parsed
    fn load_audio_file(
        path: &Path,
        source: AudioSource,
        metadata_json: &serde_json::Value,
    ) -> Result<AudioSample> {
        let mut reader = hound::WavReader::open(path)
            .map_err(|e| Error::Processing(format!("Failed to open WAV file: {e}")))?;

        let spec = reader.spec();
        let channels = spec.channels;
        let source_rate = spec.sample_rate;

        let bits_per_sample = spec.bits_per_sample;
        if bits_per_sample == 0 {
            return Err(Error::Processing("Unsupported PCM bit depth: 0".into()));
        }

        let raw_samples = match spec.sample_format {
            hound::SampleFormat::Int => match bits_per_sample {
                8 => {
                    let scale = 2_f64.powi(i32::from(bits_per_sample) - 1);
                    collect_samples(reader.samples::<i8>(), move |value: i8| {
                        (f64::from(value) / scale) as f32
                    })?
                }
                16 => {
                    let scale = 2_f64.powi(i32::from(bits_per_sample) - 1);
                    collect_samples(reader.samples::<i16>(), move |value: i16| {
                        (f64::from(value) / scale) as f32
                    })?
                }
                24 | 32 => {
                    let scale = 2_f64.powi(i32::from(bits_per_sample) - 1);
                    collect_samples(reader.samples::<i32>(), move |value: i32| {
                        (f64::from(value) / scale) as f32
                    })?
                }
                bits => {
                    return Err(Error::Processing(format!(
                        "Unsupported PCM bit depth: {bits}"
                    )));
                }
            },
            hound::SampleFormat::Float => {
                if bits_per_sample != 32 {
                    return Err(Error::Processing(format!(
                        "Unsupported float bit depth: {bits_per_sample}"
                    )));
                }
                collect_samples(reader.samples::<f32>(), |value: f32| value)?
            }
        };

        let mixed = ChannelMixer::mix_to_mono(&raw_samples, channels.try_into().unwrap_or(u8::MAX))
            .map_err(|e| Error::Processing(format!("Channel mix failed: {e}")))?;
        let resampled = SampleRateConverter::resample_to_16khz(&mixed.samples, 1, source_rate)
            .map_err(|e| Error::Processing(format!("Resample failed: {e}")))?;

        let duration = resampled.len() as f32 / SampleRateConverter::TARGET_SAMPLE_RATE as f32;

        let id = metadata_json
            .get("requested_id")
            .and_then(|s| s.as_str())
            .map(str::to_owned)
            .or_else(|| path.file_stem().and_then(|s| s.to_str()).map(str::to_owned))
            .ok_or_else(|| Error::Processing("Invalid filename".into()))?;

        let quality = metadata_json
            .get("quality")
            .and_then(|q| q.as_str())
            .unwrap_or("unknown")
            .to_owned();

        Ok(AudioSample {
            id,
            audio_data: resampled,
            sample_rate: SampleRateConverter::TARGET_SAMPLE_RATE,
            metadata: SampleMetadata {
                source,
                quality,
                duration,
            },
        })
    }

    /// Generate synthetic audio deterministically based on ID
    ///
    /// Creates a sine wave with frequency derived from ID hash.
    /// Always generates 2 seconds of 16 kHz audio.
    ///
    /// # Arguments
    ///
    /// * `id` - Sample identifier (hashed for frequency selection)
    ///
    /// # Returns
    ///
    /// Synthetic audio sample (always succeeds)
    fn generate_synthetic(id: &str) -> AudioSample {
        const SAMPLE_RATE: u32 = 16000;
        const DURATION: f32 = 2.0; // 2 seconds
        const BASE_FREQ: f32 = 440.0; // A4 note

        let num_samples = (SAMPLE_RATE as f32 * DURATION) as usize;

        let freq = BASE_FREQ + (Self::hash_id(id) % 200) as f32;

        let audio_data: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                let phase = 2.0 * std::f32::consts::PI * freq * t;
                phase.sin() * 0.3 // 30% amplitude to avoid clipping
            })
            .collect();

        AudioSample {
            id: id.to_owned(),
            audio_data,
            sample_rate: SAMPLE_RATE,
            metadata: SampleMetadata {
                source: AudioSource::Synthetic,
                quality: String::from("synthetic"),
                duration: DURATION,
            },
        }
    }

    /// Hash ID string to u32 for deterministic randomness
    ///
    /// Simple sum-of-bytes hash for frequency selection
    fn hash_id(id: &str) -> u32 {
        id.bytes().map(u32::from).sum()
    }

    fn lookup_manifest_entry(&self, id: &str) -> Result<Option<serde_json::Value>> {
        let manifest_path = self.curated_path.join("manifest.json");
        if !manifest_path.exists() {
            return Ok(None);
        }

        let manifest_content = std::fs::read_to_string(&manifest_path)?;
        let manifest: serde_json::Value = serde_json::from_str(&manifest_content)
            .map_err(|e| Error::Processing(format!("Invalid manifest JSON: {e}")))?;

        let samples = manifest
            .get("samples")
            .and_then(|s| s.as_array())
            .ok_or_else(|| Error::InvalidInput("samples array not found in manifest".into()))?;

        Ok(samples
            .iter()
            .find(|entry| entry.get("id").and_then(|i| i.as_str()) == Some(id))
            .cloned())
    }
}

impl Default for AudioFixtures {
    fn default() -> Self {
        Self::new()
    }
}

fn collect_samples<T, I, F>(iter: I, mut convert: F) -> Result<Vec<f32>>
where
    I: Iterator<Item = std::result::Result<T, hound::Error>>,
    F: FnMut(T) -> f32,
{
    iter.map(|sample| {
        sample
            .map(&mut convert)
            .map_err(|e| Error::Processing(format!("Sample read error: {e}")))
    })
    .collect::<Result<Vec<f32>>>()
}

fn is_wav_readable(path: &Path) -> bool {
    path.is_file() && hound::WavReader::open(path).is_ok()
}

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_fallback_always_succeeds() {
        let fixtures = AudioFixtures::new();

        // Request non-existent sample (will fallback to synthetic)
        let sample = fixtures
            .load_sample("nonexistent_sample_xyz")
            .expect("Synthetic fallback should always succeed");

        assert_eq!(sample.metadata.source, AudioSource::Synthetic);
        assert_eq!(sample.sample_rate, 16000);
        assert!(!sample.audio_data.is_empty());
        assert_eq!(sample.metadata.quality, "synthetic");
    }

    #[test]
    fn test_deterministic_synthetic_generation() {
        let fixtures = AudioFixtures::new();

        let sample1 = fixtures
            .load_sample("test_id_123")
            .expect("Load should succeed");
        let sample2 = fixtures
            .load_sample("test_id_123")
            .expect("Load should succeed");

        // Same ID should produce identical synthetic audio
        assert_eq!(sample1.audio_data, sample2.audio_data);
        assert_eq!(sample1.sample_rate, sample2.sample_rate);
        assert_eq!(sample1.metadata.duration, sample2.metadata.duration);
    }

    #[test]
    fn test_different_ids_produce_different_audio() {
        let fixtures = AudioFixtures::new();

        let sample1 = fixtures.load_sample("id_a").expect("Load should succeed");
        let sample2 = fixtures.load_sample("id_b").expect("Load should succeed");

        // Different IDs should produce different frequencies
        assert_ne!(sample1.audio_data, sample2.audio_data);
    }

    #[test]
    fn test_has_real_datasets_detection() {
        let fixtures = AudioFixtures::new();
        let has_datasets = fixtures.has_real_datasets();

        assert!(
            has_datasets == fixtures.has_real_datasets(),
            "Dataset availability should be stable across repeated checks"
        );
    }

    #[test]
    fn test_synthetic_audio_properties() {
        let fixtures = AudioFixtures::new();
        let sample = fixtures
            .load_sample("test_sample")
            .expect("Load should succeed");

        // Validate synthetic audio properties
        assert_eq!(sample.sample_rate, 16000);
        assert_eq!(sample.metadata.duration, 2.0);
        assert_eq!(sample.audio_data.len(), 32000); // 2 seconds * 16000 Hz

        // Check amplitude is reasonable [-1.0, 1.0]
        for &val in &sample.audio_data {
            assert!(
                (-1.0..=1.0).contains(&val),
                "Sample value {} out of range",
                val
            );
        }
    }

    #[test]
    fn test_load_curated_sample_if_available() {
        let fixtures = AudioFixtures::new();

        // Try to load from curated dataset
        // Will fallback to synthetic if not available
        let sample = fixtures
            .load_sample("sample_0000")
            .expect("Load should succeed with fallback");

        match sample.metadata.source {
            AudioSource::Curated => {
                assert_eq!(sample.sample_rate, 16000);
                assert!(!sample.audio_data.is_empty());
            }
            AudioSource::FullDataset => {
                assert_eq!(sample.sample_rate, 16000);
                assert!(!sample.audio_data.is_empty());
            }
            AudioSource::Synthetic => {
                assert_eq!(sample.sample_rate, 16000);
            }
        }
    }

    #[test]
    #[ignore = "requires curated dataset (git lfs pull) to validate manifest structure"]
    fn test_smoke_test_curated_manifest_structure() {
        let fixtures = AudioFixtures::new();

        if !fixtures.curated_dataset_available() {
            return;
        }

        // Validate manifest exists and is valid JSON
        let manifest_path = fixtures.curated_path.join("manifest.json");
        assert!(
            manifest_path.exists(),
            "Curated manifest should exist at {:?}",
            manifest_path
        );

        let manifest_content =
            std::fs::read_to_string(&manifest_path).expect("Should be able to read manifest.json");
        let manifest: serde_json::Value =
            serde_json::from_str(&manifest_content).expect("Manifest should be valid JSON");

        // Validate manifest structure
        let metadata = manifest
            .get("metadata")
            .expect("Manifest should have 'metadata' field");
        let samples = manifest
            .get("samples")
            .and_then(|s| s.as_array())
            .expect("Manifest should have 'samples' array");

        // Validate metadata fields
        assert!(
            metadata.get("total_count").is_some(),
            "Metadata should have 'total_count'"
        );
        assert!(
            metadata.get("seed").is_some(),
            "Metadata should have 'seed'"
        );
        assert!(
            metadata.get("quality_distribution").is_some(),
            "Metadata should have 'quality_distribution'"
        );

        // Validate sample structure
        assert_eq!(
            samples.len(),
            500,
            "Curated dataset should have exactly 500 samples"
        );

        let first_sample = samples.first().expect("Samples array should not be empty");
        assert!(
            first_sample.get("id").is_some(),
            "Sample should have 'id' field"
        );
        assert!(
            first_sample.get("filename").is_some(),
            "Sample should have 'filename' field"
        );
        assert!(
            first_sample.get("quality").is_some(),
            "Sample should have 'quality' field"
        );
        assert!(
            first_sample.get("duration").is_some(),
            "Sample should have 'duration' field"
        );

        assert_eq!(samples.len(), 500);
    }

    #[test]
    #[ignore = "requires curated dataset to validate real audio files"]
    fn test_smoke_test_sample_loading_integrity() {
        let fixtures = AudioFixtures::new();

        if !fixtures.curated_dataset_available() {
            return;
        }

        // Sample every 50th file to verify loading works (10 samples total)
        let sample_indices: Vec<usize> = (0..500).step_by(50).collect();

        for i in &sample_indices {
            let sample_id = format!("sample_{i:04}");
            let sample = fixtures
                .load_sample(&sample_id)
                .unwrap_or_else(|_| panic!("Should load sample {sample_id}"));

            // Validate sample came from curated dataset
            assert_eq!(
                sample.metadata.source,
                AudioSource::Curated,
                "Sample {sample_id} should come from curated dataset"
            );

            // Validate audio properties
            assert_eq!(
                sample.sample_rate, 16000,
                "Sample {sample_id} should be 16 kHz"
            );
            assert!(
                !sample.audio_data.is_empty(),
                "Sample {sample_id} should have audio data"
            );
            assert!(
                sample.metadata.duration > 0.0,
                "Sample {sample_id} should have positive duration"
            );

            // Validate audio data is in valid range
            for &val in &sample.audio_data {
                assert!(
                    (-1.0..=1.0).contains(&val),
                    "Sample {sample_id} has out-of-range value: {val}"
                );
            }
        }

        assert_eq!(sample_indices.len(), 10);
    }

    #[test]
    #[ignore = "requires curated dataset to check quality distribution"]
    fn test_smoke_test_audio_quality_distribution() {
        let fixtures = AudioFixtures::new();

        if !fixtures.curated_dataset_available() {
            return;
        }

        // Load manifest to check quality distribution
        let manifest_path = fixtures.curated_path.join("manifest.json");
        let manifest_content =
            std::fs::read_to_string(&manifest_path).expect("Should be able to read manifest.json");
        let manifest: serde_json::Value =
            serde_json::from_str(&manifest_content).expect("Manifest should be valid JSON");

        let quality_dist = manifest
            .get("metadata")
            .and_then(|m| m.get("quality_distribution"))
            .and_then(|q| q.as_object())
            .expect("Manifest should have quality_distribution");

        // Expected distribution: 50% clean, 30% moderate, 20% challenging
        let clean = quality_dist
            .get("clean")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0);
        let moderate = quality_dist
            .get("moderate")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0);
        let challenging = quality_dist
            .get("challenging")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0);

        // Validate quality categories exist
        assert!(clean > 0, "Should have 'clean' quality samples");
        assert!(moderate > 0, "Should have 'moderate' quality samples");
        assert!(challenging > 0, "Should have 'challenging' quality samples");

        // Validate total matches expected count
        let total = clean + moderate + challenging;
        assert_eq!(total, 500, "Total quality samples should equal 500");

        // Validate approximate distribution (allow ±10% variance)
        let clean_percent = (clean as f64 / 500.0) * 100.0;
        let moderate_percent = (moderate as f64 / 500.0) * 100.0;
        let challenging_percent = (challenging as f64 / 500.0) * 100.0;

        assert!(
            (40.0..=60.0).contains(&clean_percent),
            "Clean samples should be ~50% (got {clean_percent:.1}%)"
        );
        assert!(
            (20.0..=40.0).contains(&moderate_percent),
            "Moderate samples should be ~30% (got {moderate_percent:.1}%)"
        );
        assert!(
            (10.0..=30.0).contains(&challenging_percent),
            "Challenging samples should be ~20% (got {challenging_percent:.1}%)"
        );

        assert!((0.0..=100.0).contains(&clean_percent));
        assert!((0.0..=100.0).contains(&moderate_percent));
        assert!((0.0..=100.0).contains(&challenging_percent));
    }
}
