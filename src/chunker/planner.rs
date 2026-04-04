/// Plan chunk sizes (in samples) respecting target ± tolerance bounds and
/// configured min/max duration constraints.
pub fn plan_chunk_sizes(
    total_samples: usize,
    target_samples: usize,
    tolerance_samples: usize,
    min_chunk_samples: usize,
    max_chunk_samples: usize,
) -> Vec<usize> {
    if total_samples == 0 {
        return Vec::new();
    }

    let target = target_samples.max(1);
    let base_min = min_chunk_samples.max(1);
    let tolerance_lower = target.saturating_sub(tolerance_samples);
    let mut min_samples = base_min.max(tolerance_lower);

    let mut max_samples = max_chunk_samples.max(base_min);
    let tolerance_upper = target.saturating_add(tolerance_samples);
    max_samples = max_samples.min(tolerance_upper.max(min_samples));

    if min_samples > max_samples {
        min_samples = base_min;
        max_samples = max_chunk_samples.max(min_samples);
    }

    let mut sizes = Vec::new();
    let mut remaining = total_samples;

    while remaining > 0 {
        let mut chunk = remaining.min(target);
        if chunk < min_samples && remaining > chunk {
            chunk = min_samples.min(remaining);
        }
        if chunk > max_samples {
            chunk = max_samples.min(remaining);
        }

        let mut remaining_after = remaining.saturating_sub(chunk);

        if remaining_after > 0 && remaining_after < min_samples {
            let deficit = min_samples - remaining_after;
            let max_extra = max_samples.saturating_sub(chunk);
            let adjustment = deficit.min(max_extra);
            chunk += adjustment;
            if chunk > remaining {
                chunk = remaining;
            }
            remaining_after = remaining - chunk;
        }

        sizes.push(chunk);
        remaining = remaining_after;
    }

    rebalance_tail(&mut sizes, min_samples, max_samples);

    sizes
}

/// Rebalance the last chunk if it falls below the minimum, borrowing from
/// earlier chunks or merging into the previous one.
fn rebalance_tail(sizes: &mut Vec<usize>, min_samples: usize, max_samples: usize) {
    if sizes.len() <= 1 {
        return;
    }

    let last_idx = sizes.len() - 1;
    let last_size = match sizes.get(last_idx).copied() {
        Some(s) => s,
        None => return,
    };

    if last_size >= min_samples {
        return;
    }

    let mut deficit = min_samples - last_size;
    for i in (0..last_idx).rev() {
        let available = sizes
            .get(i)
            .copied()
            .unwrap_or(0)
            .saturating_sub(min_samples);
        if available == 0 {
            continue;
        }

        let transfer = available.min(deficit);
        if let Some(slot) = sizes.get_mut(i) {
            *slot -= transfer;
        }
        if let Some(slot) = sizes.get_mut(last_idx) {
            *slot += transfer;
        }

        deficit -= transfer;
        if deficit == 0 {
            break;
        }
    }

    let last_size = sizes.get(last_idx).copied().unwrap_or(0);
    if last_size < min_samples {
        let prev_idx = last_idx.saturating_sub(1);
        let prev_size = sizes.get(prev_idx).copied().unwrap_or(0);
        if prev_idx < sizes.len() - 1 && prev_size + last_size <= max_samples && sizes.len() >= 2 {
            if let Some(slot) = sizes.get_mut(prev_idx) {
                *slot += last_size;
            }
            sizes.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splits_into_reasonable_chunk_sizes() {
        let sizes = plan_chunk_sizes(1_000, 400, 100, 200, 600);
        assert_eq!(sizes.iter().sum::<usize>(), 1_000);
        assert!(sizes.iter().all(|&chunk| (200..=600).contains(&chunk)));
    }

    #[test]
    fn test_merges_tail_when_it_falls_below_minimum() {
        let sizes = plan_chunk_sizes(750, 500, 0, 300, 600);
        assert_eq!(sizes, vec![500, 250]);
    }

    #[test]
    fn test_respects_max_chunk_limit() {
        let sizes = plan_chunk_sizes(2_000, 400, 100, 200, 500);
        assert!(sizes.iter().all(|&chunk| chunk <= 500));
        assert_eq!(sizes.iter().sum::<usize>(), 2_000);
    }
}
