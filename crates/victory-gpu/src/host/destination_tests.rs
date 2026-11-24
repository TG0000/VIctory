use super::*;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};

const KERNEL_PAGE: usize = 4 << 10;

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

fn byte(at: usize, page: usize) -> u8 {
    // SAFETY: a readable page of an allocation of `memory`.
    unsafe { ((at + page * KERNEL_PAGE + 100) as *const u8).read_volatile() }
}

/// A destination in title memory that is not all mapped is left alone.
#[test]
fn title_destinations_are_written_whole_or_not_at_all() {
    let from = vec![0u8; 25 * KERNEL_PAGE];
    let at = memory(40);
    decommit(at, 30, 1);
    copy_mapped((at + 6 * KERNEL_PAGE) as *mut u8, from.as_ptr(), from.len(), true);
    assert_ne!(byte(at, 6), 0);
    copy_mapped((at + 5 * KERNEL_PAGE) as *mut u8, from.as_ptr(), from.len(), true);
    assert!((5..30).all(|page| byte(at, page) == 0));
    free(at);
}
