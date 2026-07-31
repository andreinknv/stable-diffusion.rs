//! The tensor seam: MLX behind one crate, and the handful of things that are
//! not tensors at all.
//!
//! **Only this crate names a backend.** `scripts/check-seam.sh` enforces it,
//! and that rule is what made replacing candle with MLX a bounded change
//! rather than a rewrite of every model.
//!
//! What lives here beyond the backend is the small set of things every crate
//! needs and none of them should each invent: the error type, a seeded
//! generator whose draw order is a promise, the memory guard, and the fixture
//! skip that makes a missing checkpoint a skip rather than a failure.

pub mod error;
pub use error::{Error, Result};

/// Memory refusals — the guard declining, as opposed to something being wrong.
///
/// A refusal is not a fault: the caller can tile the work, ask for a smaller
/// image, or proceed anyway. Distinguished by a marker in the message so a
/// caller can tell the two apart without matching on a string it wrote itself.
pub mod refusal {
    use super::Error;

    /// Every refusal message begins with this.
    pub const MARKER: &str = "refusing to";

    /// Build a refusal. `detail` continues the sentence — `refuse("allocate:
    /// ...")` reads "refusing to allocate: ...".
    pub fn refuse(detail: impl std::fmt::Display) -> Error {
        Error::Refused(format!("{MARKER} {detail}"))
    }

    /// Whether an error is the guard declining rather than a fault.
    pub fn is_refusal(e: &Error) -> bool {
        matches!(e, Error::Refused(_)) || e.to_string().contains(MARKER)
    }
}

/// The MLX backend.
#[cfg(feature = "mlx")]
pub mod mlx;

/// A candle-free GGUF reader, bit-exact against candle's.
#[cfg(feature = "mlx")]
pub mod mlx_gguf;

pub mod sysmem;

/// Small helpers that touch no tensor.
pub mod ops {
    /// Bytes as a human-readable size.
    ///
    /// Lives here rather than in `sysmem` because both the memory guard and
    /// the residency reports print it, and two implementations would drift on
    /// whether a gibibyte is 1000 or 1024 megabytes.
    pub fn human_bytes(bytes: u64) -> String {
        const UNITS: [(&str, u64); 4] = [
            ("GiB", 1 << 30),
            ("MiB", 1 << 20),
            ("KiB", 1 << 10),
            ("B", 1),
        ];
        for (suffix, scale) in UNITS {
            if bytes >= scale {
                return format!("{:.1} {suffix}", bytes as f64 / scale as f64);
            }
        }
        format!("{bytes} B")
    }
}

pub mod rng {
    #[cfg(feature = "mlx")]
    use super::Result;

    /// A standard-normal draw as an MLX array, `[n, c, h, w]` in **NHWC**.
    ///
    /// The transpose happens here rather than at every call site because the
    /// draw order is what a seed pins: `normals` fills NCHW-major, and
    /// re-ordering afterwards is what keeps an MLX image identical to a candle
    /// one from the same seed.
    #[cfg(feature = "mlx")]
    pub fn randn_nhwc(
        rng: &mut SeededRng,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<crate::mlx::Array> {
        let v = rng.normals(n * c * h * w);
        let mut out = vec![0.0f32; v.len()];
        for bi in 0..n {
            for ci in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        out[((bi * h + y) * w + x) * c + ci] = v[((bi * c + ci) * h + y) * w + x];
                    }
                }
            }
        }
        crate::mlx::Array::from_slice_f32(&out, &[n, h, w, c])
    }

    /// splitmix64 — small, fast, and good enough for sampling noise.
    #[derive(Debug, Clone)]
    pub struct SeededRng {
        state: u64,
    }

    impl SeededRng {
        pub fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform in `(0, 1]`. Never returns 0, so `ln()` below is safe.
        fn next_f64(&mut self) -> f64 {
            // 53 significant bits, shifted off zero.
            let bits = self.next_u64() >> 11;
            (bits as f64 + 1.0) / (9007199254740992.0 + 1.0)
        }

        /// Standard normal values via Box-Muller.
        pub fn normals(&mut self, n: usize) -> Vec<f32> {
            let mut out = Vec::with_capacity(n);
            while out.len() < n {
                let u1 = self.next_f64();
                let u2 = self.next_f64();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = std::f64::consts::TAU * u2;
                out.push((r * theta.cos()) as f32);
                if out.len() < n {
                    out.push((r * theta.sin()) as f32);
                }
            }
            out
        }
    }
}

/// Assertions for the golden-tensor harness.
/// Skip a test for want of reference data.
///
/// Takes the same arguments as `eprintln!`. Prints a uniform `SKIP:` line, and
/// **panics instead** when `SD_REQUIRE_FIXTURES` is set — see
/// [`testing::skip_without_fixtures`] for why that switch exists.
///
/// Use it only for *missing data*. Environmental skips — no GPU, a memory
/// refusal, an unset `SD_TEST_*` path — stay plain `eprintln!`, because those
/// are not something generating fixtures would fix.
#[macro_export]
macro_rules! skip_missing_fixture {
    ($($arg:tt)*) => {
        $crate::testing::skip_without_fixtures(&format!($($arg)*))
    };
}

/// Helpers the golden tests share.
pub mod testing {
    /// `sd_tensor::testing::DEFAULT_ATOL`, the bound most references use.
    pub const DEFAULT_ATOL: f64 = 1e-4;
    /// The relative half of the same bound.
    pub const DEFAULT_RTOL: f64 = 1e-3;

    /// Set this to turn every "no reference data" skip into a failure.
    ///
    /// **Without it a truncated fixture set looks like a clean run.** CI sets
    /// it; a developer without the checkpoints does not.
    pub const REQUIRE: &str = "SD_REQUIRE_FIXTURES";

    /// Skip a test for want of a fixture — or fail, if [`REQUIRE`] is set.
    pub fn skip_without_fixtures(message: &str) {
        if std::env::var(REQUIRE).is_ok() {
            panic!("{message}\n\n{REQUIRE} is set, so a missing fixture is a failure.");
        }
        eprintln!("{message}");
    }
}
