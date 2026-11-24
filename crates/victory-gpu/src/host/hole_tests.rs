use super::*;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    VirtualProtect,
};


/// `pages` committed pages, none of their bytes zero.
fn memory(pages: usize) -> usize {
    let len = pages * KERNEL_PAGE;
    // SAFETY: a fresh allocation, filled; the tests free it.
    unsafe {
        let p = VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        assert!(!p.is_null());
        for i in 0..len {
            p.cast::<u8>().add(i).write((i * 7 + i / KERNEL_PAGE) as u8 | 1);
        }
        p as usize
    }
}

fn free(at: usize) {
    // SAFETY: an allocation of `memory`.
    unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) };
}

fn decommit(at: usize, page: usize, n: usize) {
    // SAFETY: pages of an allocation of `memory`.
    let ok = unsafe { VirtualFree((at + page * KERNEL_PAGE) as *mut _, n * KERNEL_PAGE, MEM_DECOMMIT) };
    assert_ne!(ok, 0);
}

/// Fails harmlessly on a page that is not committed.
fn protect(at: usize, page: usize, protection: u32) {
    let mut old = 0;
    // SAFETY: a page of an allocation of `memory`.
    unsafe { VirtualProtect((at + page * KERNEL_PAGE) as *const _, KERNEL_PAGE, protection, &mut old) };
}

/// Holes of every kind and place: zeros there, the bytes elsewhere, as
/// reading page by page gives.
#[test]
fn copies_around_holes_as_page_by_page() {
    const PAGES: usize = 200;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut random = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    for round in 0..120 {
        let at = memory(PAGES);
        for _ in 0..round % 12 {
            let (page, n) = (random(PAGES), 1 + random(40));
            decommit(at, page, n.min(PAGES - page));
        }
        for _ in 0..round % 5 {
            // Unreadable too, though committed.
            protect(at, random(PAGES), PAGE_NOACCESS);
        }
        let from = random(3 * KERNEL_PAGE);
        let len = 1 + random(PAGES * KERNEL_PAGE - from);
        let mut got = vec![0xaa; len];
        copy_mapped(got.as_mut_ptr(), (at + from) as *const u8, len, false);
        let mut want = vec![0xbb; len];
        copy_by_pages(want.as_mut_ptr(), at + from, len);
        assert!(got == want, "round {round} from {from} len {len}");
        free(at);
    }
    // All of it unmapped.
    let at = memory(4);
    decommit(at, 0, 4);
    let mut got = vec![0xaa; 3 * KERNEL_PAGE];
    copy_mapped(got.as_mut_ptr(), (at + 100) as *const u8, got.len(), false);
    assert!(got.iter().all(|&b| b == 0));
    free(at);
}
