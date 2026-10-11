//! End-to-end processing: decode, detect speech, chunk, clean up.
//!
//! ```text
//!  bytes ─► AudioFormatConverter ─┐
//!                                 ▼
//!  16 kHz mono samples ─────► VadDetector ─► Chunker ─► DcHighPassFilter ─► NoiseReducer ─► chunks
//! ```
//!
//! [`Pipeline`] is stateful. Feed one stream through
//! [`process_bytes`](Pipeline::process_bytes) or
//! [`process_samples`](Pipeline::process_samples) in order, then call
//! [`flush`](Pipeline::flush) to emit whatever is still buffered. Audio that
//! belongs to a speech segment that has not ended yet is held back until the
//! segment closes, so a speech chunk is never split across two calls.

use std::time::Duration;

use crate::chunker::{Chunker, ProcessedChunk, apply_overlaps};
use crate::converter::{AudioFormatConverter, STANDARD_SAMPLE_RATE};
use crate::error::{Error, Result};
use crate::preprocessing::{
    DcHighPassFilter, NoiseReducer, NoiseReductionConfig, PreprocessingConfig, VadContext,
};
use crate::time::{AudioInstant, AudioTimestamp};
use crate::vad::{SpeechChunk, VadConfig, VadDetector};

/// Output of one [`Pipeline`] call.
#[derive(Debug, Clone, Default)]
pub struct ProcessingResult {
    /// Cleaned chunks, in stream order. May be empty if all new audio is
    /// still part of an open speech segment.
    pub chunks: Vec<ProcessedChunk>,
    /// Wall-clock time per stage for this call.
    pub latencies: StageLatencies,
}

/// Wall-clock time spent in each stage of one call.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageLatencies {
    /// Decoding, mixing, and resampling. Zero for `process_samples`.
    pub decode: Duration,
    /// Voice activity detection.
    pub vad: Duration,
    /// Chunking.
    pub chunking: Duration,
    /// DC removal, high-pass, and noise reduction for all chunks.
    pub preprocessing: Duration,
    /// The whole call.
    pub total: Duration,
}

/// Stateful decode → VAD → chunk → clean-up pipeline for one stream.
///
/// ```
/// use speech_prep::Pipeline;
///
/// # fn main() -> speech_prep::Result<()> {
/// let mut pipeline = Pipeline::new()?;
///
/// // 16 kHz mono samples, e.g. from a microphone, in any block size.
/// let mut audio = vec![0.0f32; 32_000];
/// for (i, s) in audio[8_000..24_000].iter_mut().enumerate() {
///     *s = 0.5 * (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / 16_000.0).sin();
/// }
///
/// let mut chunks = Vec::new();
/// for block in audio.chunks(1_600) {
///     chunks.extend(pipeline.process_samples(block)?.chunks);
/// }
/// chunks.extend(pipeline.flush()?.chunks);
///
/// assert!(chunks.iter().any(|c| c.is_speech()));
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Pipeline {
    vad: VadDetector,
    chunker: Chunker,
    dc_filter: DcHighPassFilter,
    noise_reducer: NoiseReducer,
    /// Samples not yet emitted as chunks.
    pending: Vec<f32>,
    /// Stream index of `pending[0]`.
    pending_start: usize,
    /// Tail of the last emitted chunk, used as `overlap_prev` for the next
    /// call's first chunk.
    last_tail: Option<Vec<f32>>,
}

impl Pipeline {
    /// Create a pipeline with default settings for 16 kHz speech.
    ///
    /// # Errors
    ///
    /// Only fails if a default configuration is invalid, which would be a bug.
    pub fn new() -> Result<Self> {
        Self::with_components(
            VadDetector::new(VadConfig::default())?,
            Chunker::default(),
            DcHighPassFilter::new(PreprocessingConfig::default())?,
            NoiseReducer::new(NoiseReductionConfig::default())?,
        )
    }

    /// Create a pipeline from configured components.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] if any component is not configured
    /// for 16 kHz input, which is what the pipeline produces.
    pub fn with_components(
        vad: VadDetector,
        chunker: Chunker,
        dc_filter: DcHighPassFilter,
        noise_reducer: NoiseReducer,
    ) -> Result<Self> {
        let rates = [
            ("VadConfig::sample_rate", vad.config().sample_rate),
            (
                "PreprocessingConfig::sample_rate_hz",
                dc_filter.config().sample_rate_hz,
            ),
            (
                "NoiseReductionConfig::sample_rate_hz",
                noise_reducer.config().sample_rate_hz,
            ),
        ];
        for (name, rate) in rates {
            if rate != STANDARD_SAMPLE_RATE {
                return Err(Error::Configuration(format!(
                    "{name} is {rate} Hz; the pipeline runs at {STANDARD_SAMPLE_RATE} Hz"
                )));
            }
        }

        let mut pipeline = Self {
            vad,
            chunker,
            dc_filter,
            noise_reducer,
            pending: Vec::new(),
            pending_start: 0,
            last_tail: None,
        };
        // Components may have been used before; the pipeline counts samples
        // from zero.
        pipeline.reset();
        Ok(pipeline)
    }

    /// Decode an encoded audio file or fragment and process it.
    ///
    /// # Errors
    ///
    /// Returns decoding errors from
    /// [`AudioFormatConverter::convert_to_standard`], or a processing error.
    pub fn process_bytes(&mut self, audio_bytes: &[u8]) -> Result<ProcessingResult> {
        let start = AudioInstant::now();
        let audio = AudioFormatConverter::convert_to_standard(audio_bytes)?;
        let decode = start.elapsed();

        let mut result = self.process_samples(&audio.samples)?;
        result.latencies.decode = decode;
        result.latencies.total += decode;
        Ok(result)
    }

    /// Process the next block of 16 kHz mono samples.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if a sample is NaN or infinite, or a
    /// processing error from one of the stages.
    pub fn process_samples(&mut self, samples: &[f32]) -> Result<ProcessingResult> {
        if samples.iter().any(|s| !s.is_finite()) {
            return Err(Error::InvalidInput("samples must be finite".into()));
        }

        let start = AudioInstant::now();
        let mut latencies = StageLatencies::default();

        let vad_start = AudioInstant::now();
        let segments = self.vad.detect(samples)?;
        latencies.vad = vad_start.elapsed();

        self.pending.extend_from_slice(samples);
        let pending_end = self.pending_start + self.pending.len();

        // Emit only audio the VAD has finished classifying: everything before
        // the open segment (or before the next unanalyzed frame), plus the
        // full extent of any segment that just closed.
        let closed_end = segments
            .iter()
            .map(|seg| self.sample_index(seg.end_time))
            .max()
            .unwrap_or(0);
        let emit_end = self
            .vad
            .undecided_from()
            .max(closed_end)
            .clamp(self.pending_start, pending_end);

        // Without new speech, wait for a chunk's worth of silence instead of
        // emitting a tiny chunk on every call.
        let min_silence = Chunker::duration_to_samples(
            self.chunker.config().target_duration,
            STANDARD_SAMPLE_RATE,
        );
        let chunks = if segments.is_empty() && emit_end - self.pending_start < min_silence {
            Vec::new()
        } else {
            self.emit(&segments, emit_end, &mut latencies)?
        };
        latencies.total = start.elapsed();
        Ok(ProcessingResult { chunks, latencies })
    }

    /// Close any open speech segment, emit all buffered audio, and reset the
    /// pipeline so it can start a new stream.
    ///
    /// # Errors
    ///
    /// Returns a processing error from one of the stages.
    pub fn flush(&mut self) -> Result<ProcessingResult> {
        let start = AudioInstant::now();
        let mut latencies = StageLatencies::default();

        let vad_start = AudioInstant::now();
        let segments = self.vad.flush()?;
        latencies.vad = vad_start.elapsed();

        let emit_end = self.pending_start + self.pending.len();
        let chunks = self.emit(&segments, emit_end, &mut latencies)?;

        self.reset();
        latencies.total = start.elapsed();
        Ok(ProcessingResult { chunks, latencies })
    }

    /// Drop all buffered audio and filter state.
    pub fn reset(&mut self) {
        let stream_start = self.vad.config().stream_start_time;
        self.vad.reset(stream_start);
        self.dc_filter.reset();
        self.noise_reducer.reset();
        self.pending.clear();
        self.pending_start = 0;
        self.last_tail = None;
    }

    /// Chunk and clean `pending` up to stream sample `emit_end`.
    fn emit(
        &mut self,
        segments: &[SpeechChunk],
        emit_end: usize,
        latencies: &mut StageLatencies,
    ) -> Result<Vec<ProcessedChunk>> {
        let count = emit_end.saturating_sub(self.pending_start);
        if count == 0 {
            return Ok(Vec::new());
        }
        let audio = self
            .pending
            .get(..count)
            .ok_or_else(|| Error::Processing("emit range exceeds buffered audio".into()))?;

        let window_start = self.sample_time(self.pending_start);
        let window_end = self.sample_time(emit_end);
        let segments: Vec<SpeechChunk> = segments
            .iter()
            .filter_map(|seg| {
                let start = seg.start_time.max(window_start);
                let end = seg.end_time.min(window_end);
                (end > start).then_some(SpeechChunk {
                    start_time: start,
                    end_time: end,
                    ..*seg
                })
            })
            .collect();

        let chunk_start = AudioInstant::now();
        let mut chunks = self.chunker.chunk_with_stream_start(
            audio,
            STANDARD_SAMPLE_RATE,
            &segments,
            window_start,
        )?;
        latencies.chunking = chunk_start.elapsed();

        let preprocess_start = AudioInstant::now();
        for chunk in &mut chunks {
            let ctx = Some(VadContext {
                is_silence: chunk.is_silence(),
            });
            let filtered = self.dc_filter.process(&chunk.samples, ctx)?;
            chunk.samples = self.noise_reducer.reduce(&filtered, ctx)?;

            let (energy, has_clipping) = Chunker::compute_energy_and_clipping(&chunk.samples);
            chunk.energy = energy;
            chunk.has_clipping = has_clipping;
        }

        let overlap = Chunker::duration_to_samples(
            self.chunker.config().overlap_duration,
            STANDARD_SAMPLE_RATE,
        );
        apply_overlaps(&mut chunks, overlap, STANDARD_SAMPLE_RATE);
        if let Some(first) = chunks.first_mut() {
            first.overlap_prev = self.last_tail.take();
            if first.overlap_next.is_none() {
                let len = first.overlap_prev.as_ref().map_or(0, Vec::len);
                first.overlap_ms = (len as u64 * 1000 / u64::from(STANDARD_SAMPLE_RATE)) as u32;
            }
        }
        self.last_tail = chunks.last().map(|last| {
            let from = last.samples.len().saturating_sub(overlap);
            last.samples.get(from..).unwrap_or_default().to_vec()
        });
        latencies.preprocessing = preprocess_start.elapsed();

        self.pending.drain(..count);
        self.pending_start = emit_end;
        Ok(chunks)
    }

    fn sample_index(&self, time: AudioTimestamp) -> usize {
        let offset = time
            .duration_since(self.vad.config().stream_start_time)
            .unwrap_or_default();
        (offset.as_nanos() * u128::from(STANDARD_SAMPLE_RATE) / 1_000_000_000) as usize
    }

    fn sample_time(&self, sample: usize) -> AudioTimestamp {
        let offset = AudioTimestamp::from_samples(sample as u64, STANDARD_SAMPLE_RATE);
        self.vad
            .config()
            .stream_start_time
            .add_duration(Duration::from_nanos(offset.nanos()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: usize = 16_000;

    /// `lead` s of silence, `tone` s of 1 kHz, `tail` s of silence.
    fn burst(lead: f32, tone: f32, tail: f32) -> Vec<f32> {
        let (lead, tone, tail) = (
            (lead * SR as f32) as usize,
            (tone * SR as f32) as usize,
            (tail * SR as f32) as usize,
        );
        let mut audio = vec![0.0f32; lead + tone + tail];
        for (i, s) in audio[lead..lead + tone].iter_mut().enumerate() {
            *s = 0.5 * (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / SR as f32).sin();
        }
        audio
    }

    fn wav_bytes(samples: &[f32]) -> Vec<u8> {
        let spec = hound::WavSpec {
            sample_rate: SR as u32,
            channels: 1,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = std::io::Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
        for &s in samples {
            writer.write_sample((s * 32_767.0) as i16).unwrap();
        }
        writer.finalize().unwrap();
        cursor.into_inner()
    }

    fn run_streamed(audio: &[f32], block: usize) -> Vec<ProcessedChunk> {
        let mut pipeline = Pipeline::new().unwrap();
        let mut chunks = Vec::new();
        for b in audio.chunks(block) {
            chunks.extend(pipeline.process_samples(b).unwrap().chunks);
        }
        chunks.extend(pipeline.flush().unwrap().chunks);
        chunks
    }

    fn assert_contiguous(chunks: &[ProcessedChunk], total_samples: usize) {
        let mut expected_start = AudioTimestamp::EPOCH;
        for chunk in chunks {
            let gap = chunk.start_time.as_secs() - expected_start.as_secs();
            assert!(
                gap.abs() < 1e-3,
                "gap of {gap}s before chunk at {}",
                chunk.start_time
            );
            expected_start = chunk.end_time;
        }
        let covered: usize = chunks.iter().map(|c| c.samples.len()).sum();
        assert!(
            covered.abs_diff(total_samples) <= chunks.len(),
            "covered {covered} of {total_samples}"
        );
    }

    #[test]
    fn returns_cleaned_chunks_from_bytes() {
        let audio = burst(0.5, 1.0, 0.5);
        let mut pipeline = Pipeline::new().unwrap();
        let mut chunks = pipeline.process_bytes(&wav_bytes(&audio)).unwrap().chunks;
        chunks.extend(pipeline.flush().unwrap().chunks);

        assert!(chunks.iter().any(ProcessedChunk::is_speech));
        assert!(chunks.iter().any(ProcessedChunk::is_silence));
        assert_contiguous(&chunks, audio.len());
    }

    #[test]
    fn open_segment_is_held_until_it_closes() {
        let mut pipeline = Pipeline::new().unwrap();
        // Ends in the middle of the tone: the speech part must not be emitted
        // yet, and certainly not as silence.
        let first = pipeline
            .process_samples(&burst(0.5, 0.5, 0.0))
            .unwrap()
            .chunks;
        assert!(first.iter().all(ProcessedChunk::is_silence));
        let emitted: f64 = first
            .iter()
            .map(|c| c.duration().unwrap().as_secs_f64())
            .sum();
        assert!(
            emitted <= 0.55,
            "emitted {emitted}s before the open segment"
        );

        let rest = pipeline.flush().unwrap().chunks;
        assert!(rest.iter().any(ProcessedChunk::is_speech));
    }

    fn speech_samples(chunks: &[ProcessedChunk]) -> usize {
        chunks
            .iter()
            .filter(|c| c.is_speech())
            .map(|c| c.samples.len())
            .sum()
    }

    fn pipeline_with(vad: VadDetector) -> Pipeline {
        Pipeline::with_components(
            vad,
            Chunker::default(),
            DcHighPassFilter::new(PreprocessingConfig::default()).unwrap(),
            NoiseReducer::new(NoiseReductionConfig::default()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn block_size_does_not_change_result() {
        let audio = burst(0.5, 1.0, 0.5);
        let reference = speech_samples(&run_streamed(&audio, audio.len()));
        assert!(reference > 15_000, "reference {reference}");

        for block in [1, 16, 64, 160, 1_600, 8_000] {
            let chunks = run_streamed(&audio, block);
            assert_contiguous(&chunks, audio.len());
            let speech = speech_samples(&chunks);
            assert!(
                speech.abs_diff(reference) <= 320,
                "block {block}: {speech} speech samples vs {reference}"
            );
            // Tiny blocks must not turn into tiny chunks.
            assert!(chunks.len() <= 8, "block {block}: {} chunks", chunks.len());
        }
    }

    #[test]
    fn high_frame_overlap_does_not_duplicate_audio() {
        let tone = |n: usize| burst(0.0, n as f32 / SR as f32, 0.0);
        for gap in [400, 568, 700] {
            let mut audio = vec![0.0f32; 4_000];
            audio.extend(tone(5_000));
            audio.extend(vec![0.0; gap]);
            audio.extend(tone(5_000));
            audio.extend(vec![0.0; 4_000]);

            let mut pipeline = pipeline_with(
                VadDetector::new(VadConfig {
                    frame_overlap: 0.75,
                    ..VadConfig::default()
                })
                .unwrap(),
            );
            let mut chunks = pipeline.process_samples(&audio).unwrap().chunks;
            chunks.extend(pipeline.flush().unwrap().chunks);
            assert_contiguous(&chunks, audio.len());
        }
    }

    #[test]
    fn previously_used_detector_is_reset() {
        let vad = VadDetector::new(VadConfig::default()).unwrap();
        vad.detect(&vec![0.0; SR]).unwrap();
        let mut pipeline = pipeline_with(vad);

        let audio = burst(0.5, 1.0, 0.5);
        let mut chunks = Vec::new();
        for block in audio.chunks(1_600) {
            chunks.extend(pipeline.process_samples(block).unwrap().chunks);
        }
        chunks.extend(pipeline.flush().unwrap().chunks);

        let reference = speech_samples(&run_streamed(&audio, audio.len()));
        assert!(speech_samples(&chunks).abs_diff(reference) <= 320);
    }

    #[test]
    fn overlap_ms_matches_overlap_vectors() {
        let audio = burst(0.5, 1.0, 0.5);
        for block in [1_600, 4_000] {
            for chunk in run_streamed(&audio, block) {
                let len = chunk
                    .overlap_next
                    .as_ref()
                    .or(chunk.overlap_prev.as_ref())
                    .map_or(0, Vec::len);
                assert_eq!(chunk.overlap_ms as usize, len * 1000 / SR, "{chunk:?}");
            }
        }
    }

    #[test]
    fn flush_resets_for_next_stream() {
        let audio = burst(0.5, 1.0, 0.5);
        let mut pipeline = Pipeline::new().unwrap();
        pipeline.process_samples(&audio).unwrap();
        pipeline.flush().unwrap();

        let second = pipeline.process_samples(&audio).unwrap().chunks;
        assert_eq!(
            second.first().map(|c| c.start_time),
            Some(AudioTimestamp::EPOCH)
        );
    }

    #[test]
    fn rejects_non_finite_samples() {
        let mut pipeline = Pipeline::new().unwrap();
        assert!(pipeline.process_samples(&[0.0, f32::NAN]).is_err());
    }

    #[test]
    fn rejects_mismatched_sample_rate() {
        let vad = VadDetector::new(VadConfig {
            sample_rate: 8_000,
            ..VadConfig::default()
        })
        .unwrap();
        let result = Pipeline::with_components(
            vad,
            Chunker::default(),
            DcHighPassFilter::new(PreprocessingConfig::default()).unwrap(),
            NoiseReducer::new(NoiseReductionConfig::default()).unwrap(),
        );
        assert!(matches!(result, Err(Error::Configuration(_))));
    }

    #[test]
    fn empty_bytes_are_rejected() {
        assert!(Pipeline::new().unwrap().process_bytes(&[]).is_err());
    }
}
