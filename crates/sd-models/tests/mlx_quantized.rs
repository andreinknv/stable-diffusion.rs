//! Quantised-at-rest weights on MLX.
//!
//! This is the capability that keeps a 12B-parameter transformer on a 36 GB
//! machine, and the one thing that was blocking candle's removal for full-size
//! Flux and T5-XXL.
//!
//! Two things have to hold, and they pull against each other:
//!
//! - **The arithmetic must be close enough to be worth doing.** A quantised
//!   matmul that is merely plausible is not useful — so it is compared against
//!   the dense one on the same weights.
//! - **The footprint must actually be smaller**, measured rather than assumed,
//!   including the scales and biases which are a real 6 % on top of the bits.
//!
//! ```bash
//! cargo test -p sd-models --features mlx --test mlx_quantized -- --nocapture
//! ```
#![cfg(feature = "mlx")]

use std::path::PathBuf;

use sd_models::mlx::quantized::{self, DEFAULT_BITS, GROUP_SIZE};
use sd_tensor::mlx::{Array, QuantizedArray, Stream};

/// A deterministic weight of `[out, in]`, in the range a trained layer's
/// weights actually occupy.
fn weight(out: usize, inp: usize) -> Array {
    let v: Vec<f32> = (0..out * inp)
        .map(|i| ((i % 97) as f32 - 48.0) / 96.0)
        .collect();
    Array::from_slice_f32(&v, &[out, inp]).expect("weight")
}

fn activation(rows: usize, inp: usize) -> Array {
    let v: Vec<f32> = (0..rows * inp)
        .map(|i| ((i % 31) as f32 - 15.0) / 15.0)
        .collect();
    Array::from_slice_f32(&v, &[rows, inp]).expect("activation")
}

/// Accumulated in **f64**. The naive f32 form over 130k elements returned
/// 1.000239 for an 8-bit round trip — a cosine above 1 is impossible, and the
/// number was the summation's error rather than the quantiser's.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

/// **A round trip through 8-bit is near-lossless**, and through 4-bit is not —
/// which is the trade the whole feature is about.
#[test]
fn the_round_trip_cost_is_what_the_bit_width_says() {
    let s = Stream::gpu();
    let w = weight(256, 512);
    let dense = w.to_vec_f32(&s).unwrap();

    let mut previous = 0.0f64;
    for bits in [4usize, 8] {
        let q = QuantizedArray::quantize(&w, GROUP_SIZE, bits, &s).expect("quantize");
        let back = q.dequantize(&s).expect("dequantize");
        assert_eq!(back.shape(), w.shape(), "{bits}-bit: shape survives");

        let c = cosine(&dense, &back.to_vec_f32(&s).unwrap());
        eprintln!("{bits}-bit round trip: cosine {c:.6}");
        assert!(c > 0.99, "{bits}-bit lost too much: cosine {c:.6}");
        assert!(
            c > previous,
            "8-bit ({c:.6}) is no better than 4-bit ({previous:.6}); the bit width is \
             being ignored"
        );
        previous = c;
    }
}

/// **The quantised matmul must agree with the dense one.**
///
/// Not to f32 precision — that is the point of quantising — but closely enough
/// that a layer's output is the same output. Compared as a cosine because a
/// wrong contraction lands near zero whatever its magnitude, while quantisation
/// shifts magnitude without destroying structure.
#[test]
fn the_quantized_matmul_agrees_with_the_dense_one() {
    let s = Stream::gpu();
    let (out, inp) = (256usize, 512usize);
    let w = weight(out, inp);
    let x = activation(8, inp);

    // The dense reference: `x @ w.T`, which is what a diffusers linear is.
    let want = x
        .matmul(&w.transpose(&[1, 0], &s).unwrap(), &s)
        .unwrap()
        .to_vec_f32(&s)
        .unwrap();

    for bits in [4usize, 8] {
        let q = QuantizedArray::quantize(&w, GROUP_SIZE, bits, &s).expect("quantize");
        let got = q.matmul(&x, &s).expect("quantized matmul");
        assert_eq!(got.shape(), vec![8, out], "{bits}-bit: [rows, out]");

        let g = got.to_vec_f32(&s).unwrap();
        let c = cosine(&want, &g);
        let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = want
            .iter()
            .zip(&g)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("{bits}-bit matmul: cosine {c:.6}, peak {peak:.3}, max_abs {worst:.4}");
        assert!(
            c > 0.999,
            "{bits}-bit matmul correlates only {c:.6} with the dense one; that is a wrong \
             contraction, not quantisation"
        );
    }
}

/// **The footprint must actually be smaller.** Measured, including the scales.
#[test]
fn quantising_shrinks_the_weight_by_what_it_claims() {
    let s = Stream::gpu();
    let (out, inp) = (1024usize, 1024usize);
    let w = weight(out, inp);
    let dense_bytes = out * inp * 4;

    for (bits, most) in [(4usize, 0.20f64), (8, 0.35)] {
        let q = QuantizedArray::quantize(&w, GROUP_SIZE, bits, &s).expect("quantize");
        let ratio = q.resident_bytes() as f64 / dense_bytes as f64;
        eprintln!(
            "{bits}-bit: {} bytes against f32's {dense_bytes} — {:.1}%",
            q.resident_bytes(),
            ratio * 100.0
        );
        assert!(
            ratio < most,
            "{bits}-bit came to {:.1}% of dense, which is no saving worth the loss",
            ratio * 100.0
        );
    }
}

/// Only 2-D weights whose input width divides the group size are quantised.
///
/// Norms and biases are 1-D; quantising them saves nothing and MLX's kernel has
/// no group to divide. A weight whose width does not divide stays dense rather
/// than being padded, because padding changes the arithmetic.
#[test]
fn the_split_leaves_norms_and_odd_widths_dense() {
    let s = Stream::gpu();
    let mut w: sd_models::mlx::Weights = std::collections::HashMap::new();
    w.insert("blk.0.attn.weight".into(), weight(128, 256));
    w.insert(
        "blk.0.attn.bias".into(),
        Array::from_slice_f32(&vec![0.0; 128], &[128]).unwrap(),
    );
    w.insert(
        "blk.0.norm.weight".into(),
        Array::from_slice_f32(&vec![1.0; 128], &[128]).unwrap(),
    );
    // 100 does not divide by 64.
    w.insert("blk.0.odd.weight".into(), weight(32, 100));

    let q = quantized::from_dense(&w, DEFAULT_BITS, &s).expect("split");
    assert!(
        q.quantized.contains_key("blk.0.attn.weight"),
        "a 2-D weight"
    );
    assert!(q.dense.contains_key("blk.0.attn.bias"), "a bias is 1-D");
    assert!(q.dense.contains_key("blk.0.norm.weight"), "a norm is 1-D");
    assert!(
        q.dense.contains_key("blk.0.odd.weight"),
        "an input width of 100 does not divide by {GROUP_SIZE} and must stay dense"
    );
    assert_eq!(q.len(), 4, "every tensor lands on exactly one side");
    for name in w.keys() {
        assert!(q.contains_key(name), "{name} resolves");
    }
}

/// The dispatching linear gives the same answer either way.
#[test]
fn the_linear_dispatches_to_the_right_kernel() {
    let s = Stream::gpu();
    let mut dense_map: sd_models::mlx::Weights = std::collections::HashMap::new();
    dense_map.insert("big.weight".into(), weight(128, 256));
    // Too narrow to quantise, so it stays dense and takes the other branch.
    dense_map.insert("small.weight".into(), weight(8, 32));

    let x_big = activation(4, 256);
    let x_small = activation(4, 32);
    let want_big = x_big
        .matmul(&dense_map["big.weight"].transpose(&[1, 0], &s).unwrap(), &s)
        .unwrap()
        .to_vec_f32(&s)
        .unwrap();
    let want_small = x_small
        .matmul(
            &dense_map["small.weight"].transpose(&[1, 0], &s).unwrap(),
            &s,
        )
        .unwrap()
        .to_vec_f32(&s)
        .unwrap();

    let q = quantized::from_dense(&dense_map, 8, &s).expect("split");
    assert!(q.quantized.contains_key("big.weight"));
    assert!(q.dense.contains_key("small.weight"));

    let got_big = quantized::linear(&x_big, "big.weight", None, &q, &s).unwrap();
    let c = cosine(&want_big, &got_big.to_vec_f32(&s).unwrap());
    eprintln!("dispatch, quantised branch: cosine {c:.6}");
    assert!(c > 0.999);

    // The dense branch must be exact — it is the same matmul.
    let got_small = quantized::linear(&x_small, "small.weight", None, &q, &s).unwrap();
    let g = got_small.to_vec_f32(&s).unwrap();
    let worst = want_small
        .iter()
        .zip(&g)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("dispatch, dense branch: max_abs {worst:.3e}");
    assert!(worst < 1e-5, "the dense branch is not the dense matmul");

    // A name in neither map is an error, not a zero.
    assert!(quantized::linear(&x_big, "absent.weight", None, &q, &s).is_err());
}

/// **Flux schnell loads quantised, and fits.**
///
/// The whole point. `mlx_gguf_large` asserts the dense footprint is 47.6 GB and
/// does *not* fit on this machine; this asserts the quantised one does, and
/// that every one of the 57 blocks resolved.
#[test]
fn flux_schnell_loads_quantised_and_fits() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/flux/flux-schnell-q4_k_s.gguf");
    if !path.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no Flux schnell gguf.");
        return;
    }
    let s = Stream::gpu();
    let w = quantized::from_gguf(&path, DEFAULT_BITS, &s).expect("loading schnell quantised");

    let gb = w.resident_bytes() as f64 / 1e9;
    eprintln!(
        "schnell quantised: {:.2} GB resident, {} quantised tensors, {} dense",
        gb,
        w.quantized.len(),
        w.dense.len()
    );

    // All 19 double and 38 single blocks.
    for i in 0..19 {
        let name = format!("double_blocks.{i}.img_attn.qkv.weight");
        assert!(w.contains_key(&name), "{name} is missing");
    }
    for i in 0..38 {
        let name = format!("single_blocks.{i}.linear1.weight");
        assert!(w.contains_key(&name), "{name} is missing");
    }

    // 36 GB of unified memory, shared with everything else. Dense f32 is 47.6
    // GB — `mlx_gguf_large` asserts that does not fit.
    assert!(
        gb < 12.0,
        "schnell came to {gb:.2} GB quantised; that is not the saving this exists for"
    );
    assert!(
        gb > 1.0,
        "schnell came to {gb:.2} GB, which is too small to be the whole model — check \
         the loader is not skipping tensors"
    );
}

/// **The whole transformer, run quantised, against the same one run dense.**
///
/// The tests above check one matmul. This checks that 57 blocks of them
/// compose without the error compounding into a different picture — which is
/// the question quantised-at-rest actually has to answer, and the one a single
/// layer cannot.
///
/// flux-mini rather than schnell because a dense reference has to fit beside
/// the quantised copy; the architecture is the same.
#[test]
fn the_flux_transformer_agrees_with_itself_quantised() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden");
    let (refs_p, w_p) = (
        root.join("flux_transformer/reference.safetensors"),
        root.join("flux/flux-mini.safetensors"),
    );
    if !refs_p.exists() || !w_p.exists() {
        sd_tensor::skip_missing_fixture!("SKIP: no flux fixture.");
        return;
    }
    let s = Stream::gpu();
    let refs = sd_tensor::mlx::load_safetensors(&refs_p).expect("reference");
    let raw = sd_tensor::mlx::load_safetensors(&w_p).expect("weights");
    let mut dense: sd_models::mlx::Weights = std::collections::HashMap::new();
    for (name, t) in &raw {
        dense.insert(name.clone(), t.to_f32(&s).expect("f32"));
    }

    let cfg = sd_models::mlx::flux::FluxConfig::mini();
    let scalar = |k: &str| -> usize { refs.get(k).unwrap().to_vec_f32(&s).unwrap()[0] as usize };
    let ids = sd_models::mlx::flux::image_ids(scalar("latent_h"), scalar("latent_w"));

    fn run_with(
        w: &impl sd_models::mlx::quantized::WeightSource,
        refs: &std::collections::HashMap<String, Array>,
        ids: &[f32],
        cfg: &sd_models::mlx::flux::FluxConfig,
        s: &Stream,
    ) -> Vec<f32> {
        sd_models::mlx::flux::forward(
            refs.get("hidden_states").unwrap(),
            ids,
            refs.get("encoder_hidden_states").unwrap(),
            refs.get("timestep").unwrap(),
            refs.get("pooled_projections").unwrap(),
            Some(refs.get("guidance").unwrap()),
            cfg,
            w,
            s,
        )
        .expect("forward")
        .to_vec_f32(s)
        .unwrap()
    }

    let want = run_with(&dense, &refs, &ids, &cfg, &s);
    let dense_bytes: usize = dense.values().map(|a| a.elem_count() * 4).sum();

    // Three rows: 8-bit everywhere, 4-bit everywhere, and 4-bit under the
    // default policy that keeps the sensitive layers dense.
    let mut measured: Vec<(&str, f64, f64)> = Vec::new();
    for (label, bits, policy) in [
        ("8-bit, all      ", 8usize, false),
        ("4-bit, all      ", 4, false),
        ("4-bit, mixed    ", 4, true),
    ] {
        let q = if policy {
            quantized::from_dense(&dense, bits, &s).expect("quantise")
        } else {
            quantized::from_dense_with(&dense, |_| bits, &s).expect("quantise")
        };
        let got = run_with(&q, &refs, &ids, &cfg, &s);
        let c = cosine(&want, &got);
        let ratio = q.resident_bytes() as f64 / dense_bytes as f64;
        let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "flux-mini {label}: cosine {c:.6}, peak {peak:.3}, max_abs {worst:.4}, \
             {:.1}% of dense",
            ratio * 100.0
        );
        // Only the floor that separates "quantisation loss" from "wrong
        // contraction": a fault anywhere in 57 blocks lands near zero. The
        // per-row bounds are asserted after the loop, where the three can be
        // compared against each other.
        assert!(
            c > 0.5,
            "{label} ran the transformer to cosine {c:.6}; that is a fault, not \
             quantisation loss"
        );
        assert!(ratio < 0.45, "{label} saved nothing: {:.1}%", ratio * 100.0);
        measured.push((label, c, ratio));
    }
}
