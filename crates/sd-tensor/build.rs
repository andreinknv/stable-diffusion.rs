//! Link MLX, and only when the `mlx` feature asks for it.
//!
//! Nothing here runs for a default or `--features metal` build: the whole body
//! is behind `CARGO_FEATURE_MLX`, so the candle path keeps the build graph it
//! has today.
//!
//! Paths come from `MLX_C_PREFIX` / `MLX_PREFIX` when set, else from Homebrew.
//! Both are checked rather than assumed, because a missing dylib surfaces at
//! link time as a wall of undefined symbols that says nothing about the cause.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MLX_C_PREFIX");
    println!("cargo:rerun-if-env-changed=MLX_PREFIX");

    if std::env::var_os("CARGO_FEATURE_MLX").is_none() {
        return;
    }

    let mlx_c = prefix("MLX_C_PREFIX", "mlx-c");
    let mlx = prefix("MLX_PREFIX", "mlx");

    require(&mlx_c.join("lib/libmlxc.dylib"), "mlx-c", "MLX_C_PREFIX");
    require(&mlx.join("lib/libmlx.dylib"), "mlx", "MLX_PREFIX");

    println!(
        "cargo:rustc-link-search=native={}",
        mlx_c.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        mlx.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=mlxc");
    println!("cargo:rustc-link-lib=dylib=mlx");
    // MLX is C++; the C API does not remove the need for its runtime.
    println!("cargo:rustc-link-lib=dylib=c++");
    // Metal and friends, which libmlx calls into.
    for framework in ["Metal", "Foundation", "QuartzCore", "Accelerate"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
}

/// `$VAR` if set, else `brew --prefix <formula>`.
fn prefix(var: &str, formula: &str) -> PathBuf {
    if let Some(p) = std::env::var_os(var) {
        return PathBuf::from(p);
    }
    let out = Command::new("brew")
        .args(["--prefix", formula])
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "the `mlx` feature needs {formula}. Set {var} to its prefix, or install \
                 Homebrew so `brew --prefix {formula}` can find it ({e})."
            )
        });
    if !out.status.success() {
        panic!(
            "`brew --prefix {formula}` failed. Install it with `brew install {formula}`, \
             or set {var} to an existing prefix."
        );
    }
    PathBuf::from(String::from_utf8_lossy(&out.stdout).trim())
}

fn require(lib: &Path, formula: &str, var: &str) {
    assert!(
        lib.exists(),
        "{} does not exist. Install {formula} (`brew install {formula}`) or point {var} \
         at a prefix that contains it.",
        lib.display()
    );
}
