//! This separate integration-test binary measures one bounded metadata input.
//! A missing 1 MiB comment must not allocate its advertised payload before EOF.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use speech_prep::AudioFormatConverter;
use speech_prep::format::FormatDetector;

const COMMENT_LENGTH: usize = 1024 * 1024;
static TRACKING: AtomicBool = AtomicBool::new(false);
static LARGE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

fn record(layout: Layout) {
    if TRACKING.load(Ordering::Relaxed) && layout.size() >= COMMENT_LENGTH {
        LARGE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: All allocations and deallocations are delegated unchanged to System.
// The counters do not allocate and are atomic so they remain safe on any thread.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) && new_size >= COMMENT_LENGTH {
            LARGE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn malformed_vorbis_comment() -> Vec<u8> {
    let tone = include_bytes!("fixtures/tone.ogg");
    let header_end = 27 + usize::from(tone[26]);
    let body_len: usize = tone[27..header_end].iter().map(|&n| usize::from(n)).sum();
    // Keep the fixture's valid Vorbis identification page and stream serial.
    let mut input = tone[..header_end + body_len].to_vec();

    let mut comment = b"\x03vorbis".to_vec();
    comment.extend_from_slice(&0u32.to_le_bytes()); // no vendor string
    comment.extend_from_slice(&1u32.to_le_bytes()); // one comment
    comment.extend_from_slice(&(COMMENT_LENGTH as u32).to_le_bytes());
    // Deliberately omit the comment payload. The entire input stays below 128 B.
    let mut page = b"OggS\x00\x00".to_vec();
    page.extend_from_slice(&0u64.to_le_bytes());
    page.extend_from_slice(&tone[14..18]);
    page.extend_from_slice(&1u32.to_le_bytes());
    page.extend_from_slice(&0u32.to_le_bytes()); // CRC placeholder
    page.push(1); // one lacing entry
    page.push(u8::try_from(comment.len()).unwrap());
    page.extend_from_slice(&comment);

    let mut crc = 0u32;
    for &byte in &page {
        crc ^= u32::from(byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04c1_1db7
            } else {
                crc << 1
            };
        }
    }
    page[22..26].copy_from_slice(&crc.to_le_bytes());
    input.extend_from_slice(&page);
    input
}

fn without_payload_allocation<T>(operation: impl FnOnce() -> T) -> T {
    LARGE_ALLOCATIONS.store(0, Ordering::Relaxed);
    TRACKING.store(true, Ordering::Relaxed);
    let result = operation();
    TRACKING.store(false, Ordering::Relaxed);
    assert_eq!(
        LARGE_ALLOCATIONS.load(Ordering::Relaxed),
        0,
        "allocated the absent, advertised comment payload"
    );
    result
}

#[test]
fn missing_vorbis_comment_payload_does_not_allocate_its_claimed_length() {
    let input = malformed_vorbis_comment();
    assert!(input.len() < 128);
    // Detection and conversion both probe metadata before audio size checks.
    let _ = without_payload_allocation(|| FormatDetector::detect(&input));
    let converted =
        without_payload_allocation(|| AudioFormatConverter::convert_to_standard(&input));
    assert!(converted.is_err(), "missing setup/audio must not decode");
}
