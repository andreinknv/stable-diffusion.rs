//! What the machine can actually spare.
//!
//! The budget in [`crate::ops`] is deterministic: a shape goes in, a byte
//! count comes out, and the same input always gives the same answer. That is
//! what makes it testable, and it is the right shape for catching an `n^4`
//! blowup or a 9.66 GB im2col.
//!
//! It is also blind. It compares one allocation against a fixed ceiling and
//! has no idea whether the machine has 8 GB or 128 GB, or that something else
//! is holding 13 GB of it right now. A run can sail past it and still spend
//! fifteen minutes paging.
//!
//! So this is the other half, deliberately kept apart: a *runtime* check
//! against what is free at this moment. Callers apply it where a large job
//! starts — once per generation, not per operation — so the deterministic
//! tests stay deterministic.

use crate::{Error, Result};

/// Fraction of currently-available memory a single job may plan to use.
///
/// Four fifths, and the figure is calibrated rather than picked. Diffusion
/// holds its weights resident for the whole run, so a strict fraction refuses
/// configurations that demonstrably work: SDXL in f16 needs about 9.3 GB of
/// the 15.2 GB free on the machine this was tuned on, and renders fine. The
/// same model in f32 needs 16.3 GB, does not fit, and used to spend fifteen
/// minutes paging before failing. 0.8 separates those two.
const DEFAULT_HEADROOM: f64 = 0.8;

/// Environment override, as a fraction in `(0, 1]`.
pub const HEADROOM_ENV: &str = "SD_MEMORY_HEADROOM";

/// Physical memory installed, or `None` where we cannot ask.
pub fn total_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        // hw.memsize is the one sysctl guaranteed to exist and mean this.
        let mut out: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = c"hw.memsize";
        // SAFETY: `name` is a valid NUL-terminated C string, and `out`/`len`
        // describe a correctly sized u64. sysctlbyname writes at most `len`.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                &mut out as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        return (rc == 0).then_some(out);
    }
    #[cfg(target_os = "linux")]
    {
        return meminfo_kb("MemTotal:").map(|kb| kb * 1024);
    }
    #[allow(unreachable_code)]
    None
}

/// Memory that could be used right now without paging, or `None` where we
/// cannot ask.
///
/// "Available" rather than "free": pages held by caches and inactive mappings
/// can be reclaimed, and counting only free pages understates it enough to
/// refuse work that would have run comfortably.
pub fn available_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        // `mach_host_self` is deprecated in libc in favour of the `mach2`
        // crate. Taking a whole dependency for one call, in a project that
        // audits its dependency tree, is the worse trade — the function
        // works and its signature is fixed by the kernel ABI.
        #[allow(deprecated)]
        // SAFETY: `host_statistics64` fills `stats` with `count` u32-sized
        // words; both are sized from the type it expects.
        unsafe {
            let port = libc::mach_host_self();
            let mut stats: libc::vm_statistics64 = std::mem::zeroed();
            let mut count = (std::mem::size_of::<libc::vm_statistics64>()
                / std::mem::size_of::<libc::integer_t>())
                as libc::mach_msg_type_number_t;
            let rc = libc::host_statistics64(
                port,
                libc::HOST_VM_INFO64,
                &mut stats as *mut _ as *mut libc::integer_t,
                &mut count,
            );
            if rc != 0 {
                return None;
            }
            let page = libc::sysconf(libc::_SC_PAGESIZE) as u64;
            // Free, plus the pages the kernel can reclaim without writing
            // anything out. Speculative pages are read-ahead and free for the
            // taking; external (file-backed) inactive pages are clean.
            let reclaimable = stats.free_count as u64
                + stats.speculative_count as u64
                + stats.external_page_count as u64
                + stats.purgeable_count as u64;
            return Some(reclaimable.saturating_mul(page));
        }
    }
    #[cfg(target_os = "linux")]
    {
        // MemAvailable is the kernel's own estimate and already accounts for
        // reclaimable cache — better than anything computed from MemFree.
        return meminfo_kb("MemAvailable:").map(|kb| kb * 1024);
    }
    #[allow(unreachable_code)]
    None
}

#[cfg(target_os = "linux")]
fn meminfo_kb(key: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    text.lines()
        .find_map(|l| l.strip_prefix(key))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|n| n.parse().ok())
}

fn headroom() -> f64 {
    std::env::var(HEADROOM_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|f| *f > 0.0 && *f <= 1.0)
        .unwrap_or(DEFAULT_HEADROOM)
}

/// Human-readable byte count. Mirrors `ops::human_bytes`.
fn human(bytes: u64) -> String {
    crate::ops::human_bytes(bytes)
}

/// Refuse `bytes` when it would exceed `fraction` of `available`.
///
/// Pure, so it can be tested without depending on the machine it runs on.
/// `available` of `None` means the platform could not be asked, and the
/// answer is to allow — a guard that refuses on missing information is worse
/// than one that admits it does not know.
pub fn check_against(bytes: u64, available: Option<u64>, fraction: f64, what: &str) -> Result<()> {
    let Some(available) = available else {
        return Ok(());
    };
    let ceiling = (available as f64 * fraction) as u64;
    if bytes <= ceiling {
        return Ok(());
    }
    Err(Error::Msg(format!(
        "refusing to start: {what} needs about {}, and only {} is free right now \
         ({:.0}% of it is {}).\n\n\
         This is a check against the machine's current state, not against a fixed \
         limit — something else may be using the memory. Close it, use a smaller \
         size, or raise the fraction with {HEADROOM_ENV}=<0..1> if you know the \
         run will fit.",
        human(bytes),
        human(available),
        fraction * 100.0,
        human(ceiling),
    )))
}

/// [`check_against`], asking the OS for the current figure.
pub fn check_headroom(bytes: u64, what: &str) -> Result<()> {
    check_against(bytes, available_bytes(), headroom(), what)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_job_within_the_headroom_is_allowed() {
        assert!(check_against(1_000, Some(10_000), 0.5, "x").is_ok());
        // Exactly at the line is allowed; the check is "exceeds".
        assert!(check_against(5_000, Some(10_000), 0.5, "x").is_ok());
    }

    #[test]
    fn a_job_beyond_the_headroom_is_refused_and_says_why() {
        let err = check_against(9_000, Some(10_000), 0.5, "a decode")
            .expect_err("9000 exceeds half of 10000");
        let msg = err.to_string();
        assert!(msg.contains("a decode"), "should name the job: {msg}");
        assert!(
            msg.contains(HEADROOM_ENV),
            "should name the override: {msg}"
        );
    }

    #[test]
    fn unknown_availability_allows_rather_than_refuses() {
        // A guard that refuses when it cannot measure would make the library
        // unusable on any platform we have not taught it about.
        assert!(check_against(u64::MAX, None, 0.5, "x").is_ok());
    }

    #[test]
    fn the_headroom_fraction_scales_what_is_permitted() {
        // The same job, allowed or refused by the fraction alone.
        assert!(check_against(6_000, Some(10_000), 0.9, "x").is_ok());
        assert!(check_against(6_000, Some(10_000), 0.5, "x").is_err());
    }

    #[test]
    fn the_machine_reports_something_plausible() {
        // Not an assertion about this machine's size — just that if the
        // platform answers at all, the answer is not nonsense.
        if let Some(total) = total_bytes() {
            assert!(total > 256 * 1024 * 1024, "implausible total: {total}");
            if let Some(avail) = available_bytes() {
                assert!(avail <= total, "available {avail} exceeds total {total}");
                eprintln!(
                    "machine: {} total, {} available",
                    human(total),
                    human(avail)
                );
            }
        } else {
            eprintln!("SKIP: this platform does not report memory");
        }
    }
}
