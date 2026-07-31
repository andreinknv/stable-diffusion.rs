//! Writing a weight map back out.
//!
//! New FFI, and the failure mode it has is silent: MLX is lazy, so an
//! unevaluated array has no data behind it and saving one writes whatever the
//! buffer happened to contain — a file of exactly the right shape and size,
//! holding garbage.
#![cfg(feature = "mlx")]

use std::collections::HashMap;

use sd_tensor::mlx::{load_safetensors, save_safetensors, Array, Stream};

fn tmp(name: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("sdrs-save-{name}.safetensors"));
    let _ = std::fs::remove_file(&p);
    p
}

/// Values and shapes survive the round trip.
#[test]
fn a_weight_map_round_trips() {
    let s = Stream::cpu();
    let mut w: HashMap<String, Array> = HashMap::new();
    w.insert(
        "conv.weight".into(),
        Array::from_slice_f32(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), &[2, 3, 4]).unwrap(),
    );
    w.insert(
        "norm.bias".into(),
        Array::from_slice_f32(&[-1.5, 0.0, 2.25], &[3]).unwrap(),
    );

    let path = tmp("roundtrip");
    save_safetensors(&path, &w).expect("save");
    let back = load_safetensors(&path).expect("load");

    assert_eq!(back.len(), w.len(), "every tensor came back");
    for (name, before) in &w {
        let after = back.get(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(after.shape(), before.shape(), "{name}: shape");
        assert_eq!(
            after.to_vec_f32(&s).unwrap(),
            before.to_vec_f32(&s).unwrap(),
            "{name}: values"
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// **An unevaluated array is evaluated before it is written.**
///
/// The one that matters. `a.mul(b)` returns a graph node, not data; saving it
/// without forcing evaluation writes an uninitialised buffer at the right
/// shape. Here the arithmetic is deliberately left lazy, and the file has to
/// hold the *result*.
#[test]
fn a_lazy_array_is_evaluated_before_writing() {
    let s = Stream::cpu();
    let a = Array::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    let scaled = a.mul(&Array::scalar_f32(10.0).unwrap(), &s).unwrap();
    let summed = scaled.add(&a, &s).unwrap();

    let mut w = HashMap::new();
    w.insert("lazy".to_string(), summed);

    let path = tmp("lazy");
    save_safetensors(&path, &w).expect("save");
    let back = load_safetensors(&path).expect("load");
    assert_eq!(
        back["lazy"].to_vec_f32(&s).unwrap(),
        vec![11.0, 22.0, 33.0, 44.0],
        "the file holds an unevaluated buffer rather than the computed values"
    );
    let _ = std::fs::remove_file(&path);
}
