use crate::chunker::ProcessedChunk;

/// Fill each chunk's `overlap_prev` with the last `overlap_samples` of the
/// chunk before it, and set `overlap_ms` to match.
pub fn apply_overlaps(chunks: &mut [ProcessedChunk], overlap_samples: usize, sample_rate: u32) {
    if overlap_samples == 0 {
        return;
    }
    for i in 1..chunks.len() {
        let tail = chunks.get(i - 1).map(|prev| {
            let from = prev.samples.len().saturating_sub(overlap_samples);
            prev.samples.get(from..).unwrap_or_default().to_vec()
        });
        if let Some(chunk) = chunks.get_mut(i) {
            chunk.set_overlap_prev(tail, sample_rate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::ChunkBoundary;
    use crate::time::AudioTimestamp;

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
            overlap_ms: 0,
        }
    }

    #[test]
    fn copies_tail_of_previous_chunk() {
        let mut chunks = vec![chunk(vec![0.0, 0.1, 0.2]), chunk(vec![0.3, 0.4, 0.5])];
        apply_overlaps(&mut chunks, 2, 1_000);

        assert!(chunks[0].overlap_prev.is_none());
        assert_eq!(chunks[1].overlap_prev.as_deref(), Some(&[0.1, 0.2][..]));
        assert_eq!(chunks[1].overlap_ms, 2);
        assert_eq!(
            chunks[1].samples_with_context(),
            vec![0.1, 0.2, 0.3, 0.4, 0.5]
        );
    }

    #[test]
    fn short_previous_chunk_gives_shorter_overlap() {
        let mut chunks = vec![chunk(vec![0.0]), chunk(vec![0.1])];
        apply_overlaps(&mut chunks, 4, 16_000);
        assert_eq!(chunks[1].overlap_prev.as_deref(), Some(&[0.0][..]));
    }
}
