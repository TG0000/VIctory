use super::*;

/// Parts kept of a copy, against a byte-by-byte model.
#[test]
fn snapshots_keep_each_declared_byte_once() {
    const LEN: usize = 300;
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut random = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    for _ in 0..2000 {
        let (mut parts, mut bytes, mut model) = (Vec::<Part>::new(), Vec::new(), [false; LEN]);
        for _ in 0..random(8) {
            let from = random(LEN);
            let to = from + random(LEN - from + 1);
            for (from, to) in uncovered(&parts, from, to) {
                let at = bytes.len();
                bytes.extend((from..to).map(|i| i as u8));
                parts.push(Part { from, to, at });
            }
            parts.sort_unstable_by_key(|p| p.from);
            model[from..to].fill(true);
        }
        assert_eq!(bytes.len(), model.iter().filter(|&&m| m).count());
        let mut room = Vec::new();
        for _ in 0..20 {
            let from = random(LEN);
            let to = from + 1 + random(LEN - from);
            let want: Option<Vec<u8>> = (from..to).map(|i| model[i].then_some(i as u8)).collect();
            assert_eq!(kept(&parts, &bytes, (from, to), &mut room), want.as_deref());
        }
    }
}

/// Two overlapping declared ranges and one apart: the bytes the GPU
/// changed go back, the title's own concurrent writes stay, and what
/// changed outside every declared range does not go back.
#[test]
fn copies_back_what_changed_in_declared_ranges() {
    const LEN: usize = 8 * PAGE;
    let mut title = vec![0x11u8; LEN + PAGE];
    let start = (title.as_mut_ptr() as usize).next_multiple_of(PAGE);
    let Some((buffer, data)) = host().and_then(|h| h.shared_buffer(LEN as u64)) else {
        return; // no host GPU here
    };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = Shadow::new(buffer, data, gpu, start as u64, LEN as u64);
    let ranges = [(100, 2 * PAGE), (PAGE, 3 * PAGE), (6 * PAGE + 8, 64)];
    assert_eq!(shadow.update(start, LEN), LEN);
    for &(at, len) in &ranges {
        shadow.writing(start + at, len);
    }
    let written = [150, PAGE + 5, 2 * PAGE, 4 * PAGE - 1, 6 * PAGE + 8];
    // SAFETY: inside the mapping and the title's memory.
    unsafe {
        for at in written.into_iter().chain([5 * PAGE]) {
            data.add(at).write(0x22);
        }
        ((start + 151) as *mut u8).write(0x33);
    }
    for &(at, len) in &ranges {
        shadow.copy_back(start + at, len);
    }
    for i in 0..LEN {
        // SAFETY: the title's memory.
        let byte = unsafe { ((start + i) as *const u8).read_volatile() };
        let want = match i {
            151 => 0x33,
            i if written.contains(&i) => 0x22,
            _ => 0x11,
        };
        assert_eq!(byte, want, "at {i:#x}");
    }
    assert!(shadow.before.lock().unwrap().is_empty());
    assert!(shadow.pending.lock().unwrap().is_empty());
    drop(title);
}
