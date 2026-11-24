use super::*;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    VirtualProtect,
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

/// Fails harmlessly on a page that is not committed.
fn protect(at: usize, page: usize, protection: u32) {
    let mut old = 0;
    // SAFETY: a page of an allocation of `memory`.
    unsafe { VirtualProtect((at + page * KERNEL_PAGE) as *const _, KERNEL_PAGE, protection, &mut old) };
}

fn byte(at: usize, page: usize) -> u8 {
    // SAFETY: a readable page of an allocation of `memory`.
    unsafe { ((at + page * KERNEL_PAGE + 100) as *const u8).read_volatile() }
}

/// What shaders wrote goes to the pages that are mapped and writable.
#[test]
fn written_back_around_unmapped_and_read_only_pages() {
    let at = memory(40);
    decommit(at, 10, 2);
    protect(at, 20, PAGE_READONLY);
    let from = vec![0u8; 40 * KERNEL_PAGE - 10];
    write_mapped((at + 5) as *mut u8, from.as_ptr(), from.len());
    for page in (0..40).filter(|p| !(10..12).contains(p)) {
        assert_eq!(byte(at, page) == 0, page != 20, "page {page}");
    }
    free(at);
}
