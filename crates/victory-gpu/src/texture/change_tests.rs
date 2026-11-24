//! Noticing that a texture's memory changed, and detiling it when it did.

use super::*;
use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};

/// Bytes without a pattern to them.
fn noise(len: usize) -> Vec<u8> {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    (0..len.div_ceil(8))
        .flat_map(|_| next().to_le_bytes())
        .take(len)
        .collect()
}

#[test]
fn every_byte_of_a_texture_counts() {
    // XXH3's 1 KiB blocks of 64-byte stripes, whole and cut short.
    for len in [16 << 10, (16 << 10) - 1, 3 * 1024 + 65, 1024, 256, 16] {
        for mut bytes in [vec![0; len], noise(len)] {
            let before = content_hash(&bytes);
            assert_eq!(content_hash(&bytes.clone()), before);
            for at in 0..len {
                for flip in [1, 0x80, 0xff] {
                    bytes[at] ^= flip;
                    assert_ne!(content_hash(&bytes), before, "byte {at} of {len}");
                    bytes[at] ^= flip;
                }
            }
        }
    }
}

#[test]
fn moved_contents_count() {
    let mut bytes = noise(256 << 10);
    let before = content_hash(&bytes);
    // Lanes, stripes, blocks, micro tiles and tiles trade places.
    for size in [8, 16, 64, 256, 1024, 4096, 64 << 10] {
        for (a, b) in [(0, 1), (0, 2), (1, 3)] {
            let swap = |bytes: &mut [u8]| {
                let (head, tail) = bytes.split_at_mut(b * size);
                head[a * size..][..size].swap_with_slice(&mut tail[..size]);
            };
            swap(&mut bytes);
            assert_ne!(content_hash(&bytes), before, "{size} bytes, {a} and {b}");
            swap(&mut bytes);
        }
    }
    // The same write in two places does not cancel out.
    for apart in [8, 16, 64, 128, 1024, 64 << 10] {
        bytes[100] ^= 0x55;
        bytes[100 + apart] ^= 0x55;
        assert_ne!(content_hash(&bytes), before, "{apart} apart");
        bytes[100] ^= 0x55;
        bytes[100 + apart] ^= 0x55;
    }
    assert_eq!(content_hash(&bytes), before);
}

/// A BC7 texture with three mips in `memory`, over a staging buffer.
fn staged(memory: &mut Vec<u8>) -> Option<Staging> {
    let layout = Layout::new(&Desc {
        dimension: Dimension::Tex2d,
        width: 512,
        height: 256,
        depth: 1,
        mips: 3,
        element_bytes: 16,
        block: 4,
        samples: 1,
        swizzle: Swizzle::SW_64KB_S,
    })
    .expect("a layout");
    *memory = noise(layout.size as usize);
    let mut footprints = Vec::new();
    let mut rows = Vec::new();
    let mut total = 0;
    for m in &layout.mips {
        let pitch = (m.width * 16).next_multiple_of(256);
        footprints.push(D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: total,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Width: m.width * 4,
                Height: m.height * 4,
                Depth: 1,
                RowPitch: pitch,
                ..Default::default()
            },
        });
        rows.push(m.height);
        total += u64::from(pitch * m.height).next_multiple_of(512);
    }
    let (buffer, data) = host()?.upload_buffer(total)?;
    Some(Staging {
        buffer,
        #[cfg(feature = "diagnostics")]
        size: total,
        data,
        footprints,
        rows,
        layout,
        mips: 3,
        memory: memory.as_ptr(),
        last: Mutex::default(),
    })
}

/// The staging buffer's subresources, and what they are to hold.
fn contents(s: &Staging, memory: &[u8]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    s.footprints
        .iter()
        .zip(&s.rows)
        .enumerate()
        .map(|(mip, (f, &rows))| {
            let pitch = f.Footprint.RowPitch as usize;
            let len = pitch * rows as usize;
            // SAFETY: the subresource's part of the mapped buffer.
            let held = unsafe { std::slice::from_raw_parts(s.data.add(f.Offset as usize), len) };
            let mut expected = held.to_vec();
            s.layout
                .detile(memory, mip as u32, 0, &mut expected, pitch, len)
                .expect("detiles");
            (held.to_vec(), expected)
        })
        .unzip()
}

#[test]
fn unwatched_memory_is_detiled_only_when_it_changed() {
    let mut memory = Vec::new();
    let Some(s) = staged(&mut memory) else {
        return; // no host device
    };
    let size = memory.len();
    assert_eq!(s.refresh_now(), size);
    let (held, expected) = contents(&s, &memory);
    assert!(held == expected, "the first refresh");

    // Unchanged memory leaves the staging buffer alone.
    // SAFETY: the start of the mapped staging buffer.
    unsafe { s.data.write_volatile(!held[0][0]) };
    assert_eq!(s.refresh_now(), size);
    // SAFETY: as above.
    assert_eq!(unsafe { s.data.read_volatile() }, !held[0][0]);

    // One changed byte anywhere in it brings it up to date.
    for at in (0..size).step_by(4093).chain([size - 1]) {
        memory[at] ^= 0x10;
        s.refresh_now();
        let (held, expected) = contents(&s, &memory);
        assert!(held == expected, "a change at {at}");
    }

    // So does going back to contents it had before.
    for _ in 0..2 {
        memory[0] ^= 0xff;
        s.refresh_now();
        let (held, expected) = contents(&s, &memory);
        assert!(held == expected, "changed and back");
    }
}

/// A world frame's check of unchanged textures, 104 MB of them, with
/// BLAKE3 (before) and with `content_hash`. Run with
/// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_hash`.
#[test]
#[ignore = "benchmark"]
fn texture_hash_per_frame() {
    const FRAME: usize = 104 << 20;
    // Five frames' worth, so that none of it is still in the cache.
    let pool = noise(5 * FRAME);
    let frame = |size: usize, hash: &dyn Fn(&[u8])| {
        let mut times: Vec<f64> = (0..20)
            .map(|i| {
                let memory = &pool[i % 5 * FRAME..][..FRAME];
                let started = std::time::Instant::now();
                memory.chunks_exact(size).for_each(hash);
                started.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        times.sort_by(f64::total_cmp);
        times[times.len() / 2]
    };
    for size in [64 << 10, 1 << 20, 4 << 20, 16 << 20] {
        let before = frame(size, &|b| {
            std::hint::black_box(blake3::hash(b));
        });
        let after = frame(size, &|b| {
            std::hint::black_box(content_hash(b));
        });
        println!(
            "textures of {:>5} KiB: {before:5.1} -> {after:5.1} ms a frame ({:.1}x)",
            size >> 10,
            before / after
        );
    }
}
