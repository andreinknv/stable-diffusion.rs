//! Wan's video transformer against `diffusers`.
//!
//! ```bash
//! .venv/bin/python xtask/golden/dump_reference.py wan --output tests/golden
//! cargo test -p sd-models --features mlx --test mlx_golden_wan -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::wan::{self, WanConfig};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

const ATOL: f32 = 2e-4;

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/wan")
}

fn fixtures() -> Option<(HashMap<String, Array>, HashMap<String, Array>)> {
    let (r, w) = (
        golden().join("reference.safetensors"),
        golden().join("wan.safetensors"),
    );
    if !r.exists() || !w.exists() {
        return None;
    }
    Some((
        load_safetensors(&r).expect("reference"),
        load_safetensors(&w).expect("weights"),
    ))
}

fn config() -> WanConfig {
    WanConfig {
        num_heads: 4,
        head_dim: 32,
        layers: 2,
        in_channels: 16,
        out_channels: 16,
        text_dim: 48,
        freq_dim: 256,
        ffn_dim: 256,
        patch_size: (1, 2, 2),
        theta: 10_000.0,
        eps: 1e-6,
    }
}

fn worst(got: &Array, want: &Array, s: &Stream) -> f32 {
    let (g, e) = (
        got.to_vec_f32(s).expect("got"),
        want.to_vec_f32(s).expect("want"),
    );
    assert_eq!(g.len(), e.len(), "element count");
    g.iter()
        .zip(&e)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// **The rotary split is uneven**, and the tables are checked before anything
/// uses them.
#[test]
fn the_rotary_tables_match_diffusers() {
    let Some((refs, _)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no wan fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = config();
    // Height and width take 2 * (32 / 6) = 10 each; time takes the rest.
    assert_eq!(cfg.rope_dims(), (12, 10, 10));

    let (cos, sin) = wan::rope_tables(2, 4, 3, &cfg, &s).expect("rope");
    for (name, got, key) in [("cos", &cos, "rope_cos"), ("sin", &sin, "rope_sin")] {
        let want = refs.get(key).unwrap_or_else(|| panic!("no {key}"));
        let flat = got.reshape(&want.shape(), &s).expect("reshape");
        let d = worst(&flat, want, &s);
        eprintln!("  rope {name}  max_abs {d:.3e}");
        assert!(d <= 1e-6, "rope {name} is {d:.3e} from the reference");
    }
}

#[test]
fn the_transformer_matches_diffusers() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no wan fixture. See the module docs.");
        return;
    };
    let s = Stream::gpu();
    let cfg = config();

    let latent = refs.get("latent").expect("latent");
    let text = refs.get("text").expect("text");
    let timestep = refs.get("timestep").expect("timestep");

    let got = wan::forward(latent, text, timestep, &cfg, &w, &s).expect("forward");
    let want = refs.get("output").expect("output");
    assert_eq!(got.shape(), want.shape(), "output shape");

    let d = worst(&got, want, &s);
    let peak = want
        .to_vec_f32(&s)
        .unwrap()
        .iter()
        .fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("wan output  peak {peak:.3}  max_abs {d:.3e}");
    assert!(d <= ATOL, "the Wan transformer is {d:.3e} out");
}

/// **QK-norm is across heads.** The weight is the full width, not one head's.
#[test]
fn the_qk_norm_spans_the_whole_width() {
    let Some((_, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no wan fixture.");
        return;
    };
    let cfg = config();
    let n = w
        .get("blocks.0.attn1.norm_q.weight")
        .expect("norm_q")
        .shape();
    assert_eq!(
        n[0],
        cfg.dim(),
        "Wan norms q across the whole width; every other model here does it \
         per head with a head_dim-wide weight, and the two look identical"
    );
    assert_ne!(n[0], cfg.head_dim, "and it is not head_dim");
}

/// **Patchify and unpatchify are not inverses here**, and that is correct.
///
/// The same trap SD 3.5 already cost this project: the two ends of the model
/// order a patch's features differently. `patchify` feeds the *convolution*
/// kernel, stored `(out, in, pt, ph, pw)`, so a patch runs channel-first;
/// `proj_out` emits `(pt, ph, pw, c)`, channel-last. Composing them returns a
/// tensor of exactly the right shape with every patch shuffled internally.
///
/// Written as a test because the obvious round-trip assertion *fails on
/// correct code*, and the natural next move — "fix" one of them until it
/// passes — breaks a transformer that is verified to 2.2e-6.
#[test]
fn patchify_and_unpatchify_use_opposite_orders() {
    let s = Stream::cpu();
    let (b, c, f, h, w) = (1usize, 16usize, 2usize, 8usize, 6usize);
    let v: Vec<f32> = (0..b * c * f * h * w).map(|i| i as f32).collect();
    let x = Array::from_slice_f32(&v, &[b, c, f, h, w]).unwrap();

    let p = wan::patchify(&x, (1, 2, 2), &s).unwrap();
    // 2 frames, 4 rows, 3 columns of tokens; 1*2*2*16 features each.
    assert_eq!(p.shape(), vec![1, 24, 64]);

    // Shapes compose; contents do not.
    let back = wan::unpatchify(&p, f, h, w, (1, 2, 2), c, &s).unwrap();
    assert_eq!(back.shape(), x.shape(), "the shapes do round trip");
    assert_ne!(
        back.to_vec_f32(&s).unwrap(),
        x.to_vec_f32(&s).unwrap(),
        "if these ever agree, one end has been changed to match the other and \
         the transformer no longer matches diffusers"
    );

    // Each half is still a permutation: no value is created or lost.
    let mut a = back.to_vec_f32(&s).unwrap();
    let mut e = x.to_vec_f32(&s).unwrap();
    a.sort_by(f32::total_cmp);
    e.sort_by(f32::total_cmp);
    assert_eq!(a, e, "both ends must be permutations of the same values");
}
