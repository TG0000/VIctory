use super::*;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS,
    PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect, VirtualQuery,
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

fn protect(at: usize, page: usize, protection: u32) {
    let mut old = 0;
    // SAFETY: a page of an allocation of `memory`.
    let ok = unsafe { VirtualProtect((at + page * KERNEL_PAGE) as *const _, KERNEL_PAGE, protection, &mut old) };
    assert_ne!(ok, 0);
}

/// Checking a destination changes nothing in it: a guard page of the
/// title's stays one, and neither it nor memory without access is written.
#[test]
fn guarded_and_inaccessible_destinations_are_left_alone() {
    let from = vec![0u8; 8 * KERNEL_PAGE];
    for protection in [PAGE_READWRITE | PAGE_GUARD, PAGE_NOACCESS] {
        let at = memory(8);
        protect(at, 5, protection);
        copy_mapped((at + 100) as *mut u8, from.as_ptr(), 8 * KERNEL_PAGE - 100, true);
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery((at + 5 * KERNEL_PAGE) as *const _, &mut info, size_of_val(&info)) };
        assert_eq!(info.Protect, protection);
        assert!((0..8).filter(|&page| page != 5).all(|page| byte(at, page) != 0));
        free(at);
    }
}

/// Title threads submit, with the stacks the title gave them.
#[test]
fn a_small_stack_is_enough() {
    let copied = std::thread::Builder::new().stack_size(64 << 10).spawn(|| {
        let at = memory(40);
        let from = vec![0u8; 40 * KERNEL_PAGE];
        copy_mapped(at as *mut u8, from.as_ptr(), from.len(), true);
        let copied = (0..40).all(|page| byte(at, page) == 0);
        free(at);
        copied
    });
    assert!(copied.unwrap().join().unwrap());
}
