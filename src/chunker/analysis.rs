use std::time::Duration;

use crate::time::AudioTimestamp;

use super::Chunker;
use crate::SpeechChunk;

#[allow(clippy::multiple_inherent_impl)]
impl Chunker {
    /// Compute noise baseline from silence regions for SNR calculation.
    #[allow(clippy::cognitive_complexity)]
    pub(crate) fn compute_noise_baseline(
        audio: &[f32],
        sample_rate: u32,
        vad_segments: &[SpeechChunk],
        audio_start: AudioTimestamp,
    ) -> Option<f32> {
        if audio.is_empty() {
            return None;
        }

        if vad_segments.is_empty() {
            let noise_energy = Self::compute_rms_energy(audio);
            return (noise_energy > 1e-10).then_some(noise_energy);
        }

        let mut silence_energy_sum = 0.0f32;
        let mut silence_energy_count = 0u32;
        let total_samples = audio.len();
        let total_duration_secs = total_samples as f64 / f64::from(sample_rate);
        let total_duration = Duration::from_secs_f64(total_duration_secs);

        if let Some(first_segment) = vad_segments.first() {
            if first_segment.start_time > audio_start {
                if let Some(silence_duration) = first_segment.start_time.duration_since(audio_start)
                {
                    let silence_samples = Self::duration_to_samples(silence_duration, sample_rate);
                    if let Some(silence_audio) = audio.get(0..silence_samples.min(total_samples)) {
                        if !silence_audio.is_empty() {
                            silence_energy_sum += Self::compute_rms_energy(silence_audio);
                            silence_energy_count += 1;
                        }
                    }
                }
            }
        }

        for window in vad_segments.windows(2) {
            let Some(prev_segment) = window.first() else {
                continue;
            };
            let Some(next_segment) = window.get(1) else {
                continue;
            };

            if next_segment.start_time > prev_segment.end_time {
                if let Ok(gap_start_sample) =
                    Self::time_to_sample(prev_segment.end_time, audio_start, sample_rate)
                {
                    if let Ok(gap_end_sample) =
                        Self::time_to_sample(next_segment.start_time, audio_start, sample_rate)
                    {
                        if let Some(silence_audio) =
                            audio.get(gap_start_sample..gap_end_sample.min(total_samples))
                        {
                            if !silence_audio.is_empty() {
                                silence_energy_sum += Self::compute_rms_energy(silence_audio);
                                silence_energy_count += 1;
                            }
                        }
                    }
                }
            }
        }

        if let Some(last_segment) = vad_segments.last() {
            let total_end = audio_start.add_duration(total_duration);

            if total_end > last_segment.end_time {
                if let Ok(trailing_start) =
                    Self::time_to_sample(last_segment.end_time, audio_start, sample_rate)
                {
                    if let Some(silence_audio) = audio.get(trailing_start..total_samples) {
                        if !silence_audio.is_empty() {
                            silence_energy_sum += Self::compute_rms_energy(silence_audio);
                            silence_energy_count += 1;
                        }
                    }
                }
            }
        }

        if silence_energy_count == 0 {
            return None;
        }

        let avg = silence_energy_sum / silence_energy_count as f32;
        (avg > 1e-10).then_some(avg)
    }

    pub(crate) fn compute_rms_energy(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let mut sum_squares = 0.0f32;
        for &sample in samples {
            sum_squares = sample.mul_add(sample, sum_squares);
        }
        let mean_square = sum_squares / samples.len() as f32;
        mean_square.sqrt()
    }

    pub(crate) fn compute_energy_and_clipping(samples: &[f32]) -> (f32, bool) {
        const CLIPPING_THRESHOLD: f32 = 0.999;

        if samples.is_empty() {
            return (0.0, false);
        }

        let mut sum_squares = 0.0f32;
        let mut has_clipping = false;
        for &sample in samples {
            let abs = sample.abs();
            if abs >= CLIPPING_THRESHOLD {
                has_clipping = true;
            }
            sum_squares = sample.mul_add(sample, sum_squares);
        }
        let mean_square = sum_squares / samples.len() as f32;
        (mean_square.sqrt(), has_clipping)
    }

    pub(crate) fn calculate_snr_db(signal_rms: f32, noise_rms: f32) -> Option<f32> {
        const EPSILON: f32 = 1e-10;

        if signal_rms < EPSILON || noise_rms < EPSILON {
            return None;
        }

        let snr_ratio = signal_rms / noise_rms;
        Some(20.0 * snr_ratio.log10())
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

    #[test]
    fn test_baseline_detects_leading_silence() {
        let _chunker = chunker();
        let sample_rate = 16_000;
        let audio_start = AudioTimestamp::EPOCH;

        let mut audio = vec![0.001; 3_200];
        audio.extend(vec![0.5; 3_200]);

        let segments = vec![SpeechChunk {
            start_time: audio_start.add_duration(AudioDuration::from_millis(200)),
            end_time: audio_start.add_duration(AudioDuration::from_millis(400)),
            confidence: 0.9,
            avg_energy: 0.5,
            frame_count: 32,
        }];

        let baseline = Chunker::compute_noise_baseline(&audio, sample_rate, &segments, audio_start);
        assert!(baseline.is_some());
        assert!(baseline.unwrap() > 0.0);
    }

    #[test]
    fn test_baseline_none_without_silence() {
        let _chunker = chunker();
        let sample_rate = 16_000;
        let audio_start = AudioTimestamp::EPOCH;
        let audio = vec![0.5; 3_200];

        let segments = vec![SpeechChunk {
            start_time: audio_start,
            end_time: audio_start.add_duration(AudioDuration::from_millis(200)),
            confidence: 0.9,
            avg_energy: 0.5,
            frame_count: 32,
        }];

        let baseline = Chunker::compute_noise_baseline(&audio, sample_rate, &segments, audio_start);
        assert!(baseline.is_none());
    }

    #[test]
    fn test_snr_none_when_noise_zero() {
        assert!(Chunker::calculate_snr_db(0.5, 0.0).is_none());
    }
}
