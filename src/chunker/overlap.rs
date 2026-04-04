use crate::chunker::ProcessedChunk;

/// Add overlaps between adjacent chunks for downstream context.
///
/// Post-processes a vector of chunks to extract overlap regions from
/// adjacent chunks.
#[allow(clippy::indexing_slicing)]
pub fn apply_overlaps(chunks: &mut [ProcessedChunk], overlap_samples: usize, sample_rate: u32) {
    if chunks.is_empty() || overlap_samples == 0 {
        return;
    }

    for i in 0..chunks.len() {
        if i > 0 {
            let prev_samples = &chunks[i - 1].samples;
            chunks[i].overlap_prev = if prev_samples.len() >= overlap_samples {
                let overlap_start = prev_samples.len() - overlap_samples;
                Some(prev_samples[overlap_start..].to_vec())
            } else {
                Some(prev_samples.clone())
            };
        }

        if i < chunks.len() - 1 {
            let current_samples = &chunks[i].samples;
            chunks[i].overlap_next = if current_samples.len() >= overlap_samples {
                let overlap_start = current_samples.len() - overlap_samples;
                Some(current_samples[overlap_start..].to_vec())
            } else {
                Some(current_samples.clone())
            };
        }

        let actual_overlap_samples = chunks[i]
            .overlap_next
            .as_ref()
            .map(Vec::len)
            .or_else(|| chunks[i].overlap_prev.as_ref().map(Vec::len))
            .unwrap_or(0);

        chunks[i].overlap_ms = if actual_overlap_samples == 0 {
            0
        } else {
            let numerator = actual_overlap_samples as u64 * 1000 + u64::from(sample_rate) / 2;
            (numerator / u64::from(sample_rate)) as u32
        };
    }
}

#[cfg(test)]
mod tests {
    use crate::time::AudioTimestamp;

    use super::*;
    use crate::chunker::{ChunkBoundary, ProcessedChunk};

    fn chunk(samples: Vec<f32>) -> ProcessedChunk {
        ProcessedChunk {
            samples,
            start_boundary: ChunkBoundary::SpeechStart,
            end_boundary: ChunkBoundary::SpeechEnd,
            start_time: AudioTimestamp::EPOCH,
            end_time: AudioTimestamp::EPOCH,
            speech_ratio: 1.0,
            energy: 0.5,
            snr_db: None,
            has_clipping: false,
            overlap_prev: None,
            overlap_next: None,
            overlap_ms: 0,
        }
    }

    #[test]
    fn test_populates_prev_and_next_overlaps() {
        let mut chunks = vec![chunk(vec![0.0, 0.1, 0.2]), chunk(vec![0.3, 0.4, 0.5])];
        apply_overlaps(&mut chunks, 2, 16_000);

        assert_eq!(chunks[1].overlap_prev.as_deref(), Some(&[0.1, 0.2][..]));
        assert_eq!(chunks[0].overlap_next.as_deref(), Some(&[0.1, 0.2][..]));
        assert!(chunks[0].overlap_prev.is_none());
        assert!(chunks[1].overlap_next.is_none());
    }

    #[test]
    fn test_handles_short_chunks_without_panicking() {
        let mut chunks = vec![chunk(vec![0.0]), chunk(vec![0.1])];
        apply_overlaps(&mut chunks, 4, 16_000);

        assert_eq!(chunks[1].overlap_prev.as_deref(), Some(&[0.0][..]));
        assert_eq!(chunks[0].overlap_next.as_deref(), Some(&[0.0][..]));
    }
}
