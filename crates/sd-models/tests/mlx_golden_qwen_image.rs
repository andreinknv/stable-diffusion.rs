//! Qwen-Image's transformer against `diffusers`.
//!
//! ```bash
//! .venv/bin/python xtask/golden/dump_reference.py qwen_image --output tests/golden
//! cargo test -p sd-models --features mlx --test mlx_golden_qwen_image -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::mlx::qwen_image::{self, QwenImageConfig};
use sd_tensor::mlx::{load_safetensors, Array, Stream};

const ATOL: f32 = 2e-4;

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/qwen_image")
}

fn fixtures() -> Option<(HashMap<String, Array>, HashMap<String, Array>)> {
    let (r, w) = (
        golden().join("reference.safetensors"),
        golden().join("qwen_image.safetensors"),
    );
    if !r.exists() || !w.exists() {
        return None;
    }
    Some((
        load_safetensors(&r).expect("reference"),
        load_safetensors(&w).expect("weights"),
    ))
}

fn config() -> QwenImageConfig {
    QwenImageConfig {
        num_heads: 4,
        head_dim: 32,
        layers: 2,
        in_channels: 64,
        out_channels: 16,
        joint_attention_dim: 48,
        axes_dims: vec![8, 12, 12],
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

/// **The rotary tables, before anything uses them.**
///
/// Qwen-Image centres its spatial coordinates and starts the text stream past
/// the image's largest index. Z-Image cost an afternoon on exactly this class
/// of off-by-one, found only by comparing the reference's own frequencies —
/// so these are checked directly rather than inferred from a block's output.
#[test]
fn the_rotary_tables_match_diffusers() {
    let Some((refs, _)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no qwen_image fixture.");
        return;
    };
    let s = Stream::gpu();
    let cfg = config();
    let (frames, h, w_grid, txt_len) = (1usize, 4usize, 3usize, 5usize);

    let (img, txt) = qwen_image::rope_tables(frames, h, w_grid, txt_len, &cfg, &s).expect("rope");

    for (name, got, want_key) in [
        ("img cos", &img.0, "vid_freqs_real"),
        ("img sin", &img.1, "vid_freqs_imag"),
        ("txt cos", &txt.0, "txt_freqs_real"),
        ("txt sin", &txt.1, "txt_freqs_imag"),
    ] {
        let want = refs
            .get(want_key)
            .unwrap_or_else(|| panic!("no {want_key}"));
        // The reference stores `[seq, d/2]`; ours carries broadcast axes.
        let flat = got
            .reshape(&want.shape(), &s)
            .unwrap_or_else(|e| panic!("{name}: reshape {e}"));
        let d = worst(&flat, want, &s);
        eprintln!("  {name:<8} max_abs {d:.3e}");
        assert!(
            d <= 1e-6,
            "{name} is {d:.3e} from the reference's own table"
        );
    }
}

#[test]
fn the_transformer_matches_diffusers() {
    let Some((refs, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no qwen_image fixture. See the module docs.");
        return;
    };
    let s = Stream::gpu();
    let cfg = config();

    let img = refs.get("hidden_states").expect("hidden_states");
    let txt = refs
        .get("encoder_hidden_states")
        .expect("encoder_hidden_states");
    let timestep = refs.get("timestep").expect("timestep");

    let got = qwen_image::forward(img, txt, timestep, 1, 4, 3, &cfg, &w, &s).expect("forward");
    let want = refs.get("output").expect("output");
    assert_eq!(got.shape(), want.shape(), "output shape");

    let d = worst(&got, want, &s);
    let peak = want
        .to_vec_f32(&s)
        .unwrap()
        .iter()
        .fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!("qwen-image output  peak {peak:.3}  max_abs {d:.3e}");
    assert!(d <= ATOL, "the Qwen-Image transformer is {d:.3e} out");
}

/// **Modulation is per block, per stream** — six vectors each, unlike FLUX.2
/// where three tensors serve the whole model.
#[test]
fn every_block_carries_its_own_modulation() {
    let Some((_, w)) = fixtures() else {
        sd_tensor::skip_missing_fixture!("SKIP: no qwen_image fixture.");
        return;
    };
    let cfg = config();
    for i in 0..cfg.layers {
        for stream in ["img_mod", "txt_mod"] {
            let name = format!("transformer_blocks.{i}.{stream}.1.weight");
            let t = w.get(&name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(
                t.shape()[0],
                6 * cfg.dim(),
                "{name}: six modulation vectors"
            );
        }
    }
}
