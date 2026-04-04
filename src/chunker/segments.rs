use std::time::Duration;

use crate::error::{Error, Result};
use crate::time::AudioTimestamp;

use super::planner::plan_chunk_sizes;
use super::{ChunkBoundary, Chunker, ProcessedChunk};
use crate::SpeechChunk;

#[allow(clippy::multiple_inherent_impl)]
impl Chunker {
    pub(crate) fn create_silence_chunk(
        audio: &[f32],
        sample_rate: u32,
        start_time: AudioTimestamp,
        duration: Duration,
        audio_start: AudioTimestamp,
    ) -> Result<ProcessedChunk> {
        let start_sample = Self::time_to_sample(start_time, audio_start, sample_rate)?;
        let sample_count = Self::duration_to_samples(duration, sample_rate);
        let end_sample = (start_sample + sample_count).min(audio.len());

        let samples = audio
            .get(start_sample..end_sample)
            .ok_or_else(|| Error::InvalidInput("silence chunk out of bounds".into()))?
            .to_vec();

        let (energy, has_clipping) = Self::compute_energy_and_clipping(&samples);

        Ok(ProcessedChunk {
            samples,
            start_boundary: ChunkBoundary::Silence,
            end_boundary: ChunkBoundary::Silence,
            start_time,
            end_time: start_time.add_duration(duration),
            speech_ratio: 0.0,
            energy,
            snr_db: None,
            has_clipping,
            overlap_prev: None,
            overlap_next: None,
            overlap_ms: 0,
        })
    }

    pub(crate) fn process_speech_segment(
        &self,
        audio: &[f32],
        sample_rate: u32,
        segment: &SpeechChunk,
        noise_baseline: Option<f32>,
        audio_start: AudioTimestamp,
    ) -> Result<Vec<ProcessedChunk>> {
        let segment_duration = segment
            .end_time
            .duration_since(segment.start_time)
            .ok_or_else(|| Error::Processing("segment end_time < start_time".into()))?;
        let target_samples =
            Self::duration_to_samples(self.config.target_duration, sample_rate).max(1);
        let tolerance_samples =
            Self::duration_to_samples(self.config.duration_tolerance, sample_rate);
        let max_chunk_duration = self.max_chunk_duration();
        let max_chunk_samples = Self::duration_to_samples(max_chunk_duration, sample_rate).max(1);
        let min_chunk_duration = self.min_chunk_duration();
        let min_chunk_samples = Self::duration_to_samples(min_chunk_duration, sample_rate).max(1);

        let segment_std_duration = segment_duration;
        if segment_std_duration <= max_chunk_duration {
            let start_sample = Self::time_to_sample(segment.start_time, audio_start, sample_rate)?;
            let segment_sample_count = Self::duration_to_samples(segment_std_duration, sample_rate);
            let end_sample = start_sample + segment_sample_count;

            let samples = audio
                .get(start_sample..end_sample)
                .ok_or_else(|| Error::InvalidInput("speech segment out of bounds".into()))?
                .to_vec();

            let (energy, has_clipping) = Self::compute_energy_and_clipping(&samples);
            let snr_db = noise_baseline.and_then(|noise| Self::calculate_snr_db(energy, noise));

            return Ok(vec![ProcessedChunk {
                samples,
                start_boundary: ChunkBoundary::SpeechStart,
                end_boundary: ChunkBoundary::SpeechEnd,
                start_time: segment.start_time,
                end_time: segment.end_time,
                speech_ratio: 1.0,
                energy,
                snr_db,
                has_clipping,
                overlap_prev: None,
                overlap_next: None,
                overlap_ms: 0,
            }]);
        }

        let start_sample = Self::time_to_sample(segment.start_time, audio_start, sample_rate)?;
        let segment_sample_count = Self::duration_to_samples(segment_std_duration, sample_rate);
        let chunk_sizes = plan_chunk_sizes(
            segment_sample_count,
            target_samples,
            tolerance_samples,
            min_chunk_samples,
            max_chunk_samples,
        );

        let mut chunks = Vec::with_capacity(chunk_sizes.len());
        let mut current_sample = start_sample;

        for (index, &chunk_samples) in chunk_sizes.iter().enumerate() {
            let end_sample = current_sample + chunk_samples;
            let chunk_audio = audio
                .get(current_sample..end_sample)
                .ok_or_else(|| Error::InvalidInput("chunk split out of bounds".into()))?
                .to_vec();

            let chunk_duration_secs = chunk_samples as f64 / f64::from(sample_rate);
            let chunk_duration = Duration::from_secs_f64(chunk_duration_secs);

            let chunk_offset_secs = (current_sample - start_sample) as f64 / f64::from(sample_rate);
            let chunk_start_time = segment
                .start_time
                .add_duration(Duration::from_secs_f64(chunk_offset_secs));
            let chunk_end_time = chunk_start_time.add_duration(chunk_duration);

            let is_first = index == 0;
            let is_last = index == chunk_sizes.len() - 1;

            let start_boundary = if is_first {
                ChunkBoundary::SpeechStart
            } else {
                ChunkBoundary::Continuation
            };

            let end_boundary = if is_last {
                ChunkBoundary::SpeechEnd
            } else {
                ChunkBoundary::Continuation
            };

            let (energy, has_clipping) = Self::compute_energy_and_clipping(&chunk_audio);
            let snr_db = noise_baseline.and_then(|noise| Self::calculate_snr_db(energy, noise));

            chunks.push(ProcessedChunk {
                samples: chunk_audio,
                start_boundary,
                end_boundary,
                start_time: chunk_start_time,
                end_time: chunk_end_time,
                speech_ratio: 1.0,
                energy,
                snr_db,
                has_clipping,
                overlap_prev: None,
                overlap_next: None,
                overlap_ms: 0,
            });

            current_sample = end_sample;
        }

        Ok(chunks)
    }

    pub(crate) fn time_to_sample(
        time: AudioTimestamp,
        audio_start: AudioTimestamp,
        sample_rate: u32,
    ) -> Result<usize> {
        let duration = time
            .duration_since(audio_start)
            .ok_or_else(|| Error::Processing("time must be >= audio_start".into()))?;
        Ok(Self::duration_to_samples(duration, sample_rate))
    }

    pub(crate) fn duration_to_samples(duration: Duration, sample_rate: u32) -> usize {
        let secs = duration.as_secs_f64();
        (secs * f64::from(sample_rate)).round() as usize
    }

    pub(crate) fn max_chunk_duration(&self) -> Duration {
        let target_with_tolerance = self
            .config
            .target_duration
            .checked_add(self.config.duration_tolerance)
            .unwrap_or(self.config.max_duration);

        target_with_tolerance.min(self.config.max_duration)
    }

    pub(crate) fn min_chunk_duration(&self) -> Duration {
        let target_minus_tolerance = self
            .config
            .target_duration
            .checked_sub(self.config.duration_tolerance)
            .unwrap_or(Duration::ZERO);

        target_minus_tolerance
            .max(self.config.min_duration)
            .min(self.max_chunk_duration())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::ChunkerConfig;
    use crate::time::AudioDuration;

    fn chunker() -> Chunker {
        Chunker::new(ChunkerConfig::default())
    }

    fn sample_segment(start_ms: u64, end_ms: u64) -> SpeechChunk {
        SpeechChunk {
            start_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(start_ms)),
            end_time: AudioTimestamp::EPOCH.add_duration(AudioDuration::from_millis(end_ms)),
            confidence: 0.9,
            avg_energy: 0.5,
            frame_count: 32,
        }
    }

    #[test]
    fn test_creates_single_chunk_when_segment_short() {
        let chunker = chunker();
        let sample_rate = 16_000;
        let audio = vec![0.5; 8_000]; // 0.5 seconds
        let segment = sample_segment(0, 500);

        let chunks = chunker
            .process_speech_segment(
                &audio,
                sample_rate,
                &segment,
                Some(0.01),
                AudioTimestamp::EPOCH,
            )
            .expect("process segment");

        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_speech());
    }

    #[test]
    fn test_splits_long_segment() {
        let chunker = chunker();
        let sample_rate = 16_000;
        let audio = vec![0.5; 40_000]; // 2.5 seconds
        let segment = sample_segment(0, 2_500);

        let chunks = chunker
            .process_speech_segment(
                &audio,
                sample_rate,
                &segment,
                Some(0.01),
                AudioTimestamp::EPOCH,
            )
            .expect("process segment");

        assert!(chunks.len() > 1, "Long segment should be split");
        assert!(chunks.iter().all(ProcessedChunk::is_speech));
    }
}
