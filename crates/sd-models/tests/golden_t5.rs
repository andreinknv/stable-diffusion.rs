//! T5 v1.1 encoder against `transformers`.
//!
//! Run against `t5-v1_1-small`, not XXL. The architecture is identical and the
//! reference is 300 MB rather than 19 GB, so this answers "is the port right"
//! cheaply; "are the XXL weights mapped right" is a separate question with a
//! separate test, and keeping them apart is what made the GGUF work tractable.
//!
//! Regenerate with:
//! `python3 xtask/golden/dump_reference.py t5 --output tests/golden`

use std::collections::HashMap;
use std::path::PathBuf;

use sd_models::t5::{T5Config, T5EncoderModel};
use sd_tensor::{testing, DType, Device, Tensor, VarBuilder};

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/t5")
        .join(name)
}

fn load(dev: &Device) -> Option<(HashMap<String, Tensor>, VarBuilder<'static>)> {
    let (r, w) = (golden("reference.safetensors"), golden("t5.safetensors"));
    if !r.exists() || !w.exists() {
        sd_tensor::skip_missing_fixture!(
            "SKIP: no T5 reference. Generate with \
             `python3 xtask/golden/dump_reference.py t5 --output tests/golden`"
        );
        return None;
    }
    Some((
        sd_tensor::safetensors::load(&r, dev).unwrap(),
        sd_loader::safetensors_var_builder(&[&w], DType::F32, dev).unwrap(),
    ))
}

/// The relative position bias, on its own.
///
/// This is the piece with the most arithmetic and the least visibility: it is
/// pure integer bucketing feeding an embedding lookup, and a mistake in it
/// shifts every attention score by a small amount rather than producing an
/// error. Checking it separately means a whole-model mismatch does not have to
/// be bisected to find it.
#[test]
fn position_bias_matches_transformers() {
    let dev = Device::Cpu;
    let Some((refs, vb)) = load(&dev) else { return };

    let cfg = T5Config::v1_1_small();
    let model = T5EncoderModel::new(&cfg, vb).unwrap();
    let n = refs.get("token_ids").unwrap().dim(1).unwrap();

    let got = model.position_bias(n, &dev).unwrap();
    let want = refs.get("position_bias").unwrap();
    assert_eq!(got.dims(), want.dims(), "position bias shape");

    let c = testing::closeness(&got, want).unwrap();
    eprintln!("t5 position bias: max_abs {:.3e}", c.max_abs);
    assert!(
        c.max_abs < 1e-5,
        "position bias diverged: {:.3e}",
        c.max_abs
    );
}

#[test]
fn t5_encoder_matches_transformers_layer_by_layer() {
    let dev = Device::Cpu;
    let Some((refs, vb)) = load(&dev) else { return };

    let cfg = T5Config::v1_1_small();
    let model = T5EncoderModel::new(&cfg, vb).unwrap();
    let ids = refs.get("token_ids").unwrap();

    // Per-block, so a divergence names a block instead of appearing at the
    // output with 8 candidates behind it.
    let states = model.forward_with_hidden_states(ids).unwrap();
    assert_eq!(
        states.len(),
        cfg.num_layers + 1,
        "one state before the stack plus one per block"
    );

    // Scale-aware, because T5's activations are enormous and grow with depth:
    // 61 at the embedding, ~40,000 by the last block, then back to order 1
    // after the final norm. The same reasoning as `golden_clip_encoder.rs`,
    // where the peak is 851 — but more extreme.
    //
    // The intermediate bound is loose, and measured rather than guessed.
    // Running transformers' own T5 in f32 against f64 gives, per block:
    //
    //   block   1        2        3        4        5        6        7
    //   theirs  1.09e-4  2.62e-4  5.11e-4  5.90e-4  6.62e-4  1.51e-3  4.38e-2
    //   ours    6.10e-5  1.83e-4  3.97e-4  4.88e-4  5.34e-4     -        -
    //
    // Ours is *below* their f32 noise floor at every block we can compare, so
    // this deviation is not ours to fix — asserting anything tighter would be
    // asserting that f32 is more precise than it is.
    const RTOL: f64 = 1e-4;
    const INTERMEDIATE_ATOL: f64 = 3e-3;

    // The output is a different matter. The final RMSNorm collapses the huge
    // activations back to order 1, and transformers' own f32-vs-f64 spread
    // there is only 7.5e-6 — so this, the tensor that actually conditions the
    // transformer, is held tight.
    const OUTPUT_ATOL: f64 = 5e-5;

    for (i, got) in states.iter().enumerate() {
        let Some(want) = refs.get(&format!("hidden_{i}")) else {
            continue;
        };
        assert_eq!(got.dims(), want.dims(), "hidden_{i} shape");
        let excess = testing::allclose_excess(got, want, RTOL).unwrap();
        let c = testing::closeness(got, want).unwrap();
        eprintln!(
            "t5 hidden_{i}: max_abs {:.3e} (scale {:.0}), excess over rtol {:.3e}",
            c.max_abs,
            want.abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap(),
            excess
        );
        // The last state is post-final-norm and held to the tight bound.
        let atol = if i + 1 == states.len() {
            OUTPUT_ATOL
        } else {
            INTERMEDIATE_ATOL
        };
        assert!(
            excess < atol,
            "block {i} diverged by {excess:.3e} beyond what rtol={RTOL:.0e} \
             allows (atol {atol:.0e}) — the first block that fails is the one \
             to look at, later ones inherit it"
        );
    }

    let got = model.forward(ids).unwrap();
    let want = refs.get("last_hidden_state").unwrap();
    let excess = testing::allclose_excess(&got, want, RTOL).unwrap();
    let c = testing::closeness(&got, want).unwrap();
    eprintln!(
        "t5 output: max_abs {:.3e}, mean_abs {:.3e}, excess {:.3e}",
        c.max_abs, c.mean_abs, excess
    );
    assert!(excess < OUTPUT_ATOL, "t5 output diverged by {excess:.3e}");

    // The output is post-final-norm and so is order 1 — if that ever stops
    // being true the comparison above has quietly become vacuous, since a
    // relative bound on huge values tolerates almost anything.
    let scale = want
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        scale < 100.0,
        "expected the final norm to bring the output back to order 1, got {scale}"
    );
}

/// Guards the geometry that separates XXL from the checkpoint tested above.
#[test]
fn xxl_config_matches_the_published_one() {
    let x = T5Config::xxl();
    assert_eq!(x.d_model, 4096);
    assert_eq!(x.d_ff, 10240);
    assert_eq!(x.num_layers, 24);
    assert_eq!(x.num_heads, 64);
    // 64 heads x 64 wide is exactly d_model here, which is a coincidence of
    // this size and not something the code may assume.
    assert_eq!(x.num_heads * x.d_kv, x.d_model);
    // The small config must differ in shape but share everything the bucket
    // arithmetic depends on, or the golden test above proves less than it
    // appears to.
    let s = T5Config::v1_1_small();
    assert_eq!(s.d_kv, x.d_kv);
    assert_eq!(
        s.relative_attention_num_buckets,
        x.relative_attention_num_buckets
    );
    assert_eq!(
        s.relative_attention_max_distance,
        x.relative_attention_max_distance
    );
}
