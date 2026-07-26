//! Reading a GGUF header.
//!
//! The files here are written with candle's GGUF *writer* and read back
//! through our wrapper on its reader. That crosses two independent code
//! paths, so it is not the parser agreeing with itself — but it is also not
//! the same as reading a checkpoint someone else produced.
//!
//! **What this does not cover.** GGUF in the wild varies: non-UTF8 strings,
//! null-terminated strings the spec says are not, version-2 layouts beside
//! version-3. candle's reader carries a lossy-UTF8 path that exists precisely
//! because real files violate the spec. None of that is exercised until this
//! is pointed at a genuine quantised checkpoint, which is why GGUF stays
//! marked incomplete on the roadmap.

use sd_loader::GgufInfo;
use sd_tensor::gguf::{GgmlDType, QTensor, Value};
use sd_tensor::{DType, Device, Tensor};

fn write_gguf(
    name: &str,
    metadata: &[(&str, Value)],
    tensors: &[(&str, Vec<usize>, GgmlDType)],
) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("sdrs-gguf-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);

    let dev = Device::Cpu;
    let quantised: Vec<(String, QTensor)> = tensors
        .iter()
        .map(|(n, shape, dtype)| {
            // Quantisation works in blocks along the LAST dimension — Q4_K and
            // Q5_K want it divisible by 256, Q8_0 by 32. The shapes below
            // satisfy that; anything else fails to quantise.
            let t = Tensor::zeros(shape.clone(), DType::F32, &dev).unwrap();
            (n.to_string(), QTensor::quantize(&t, *dtype).unwrap())
        })
        .collect();

    let meta: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    let ts: Vec<(&str, &QTensor)> = quantised.iter().map(|(n, q)| (n.as_str(), q)).collect();

    let mut f = std::fs::File::create(&path).unwrap();
    sd_tensor::gguf::write(&mut f, &meta, &ts).unwrap();
    path
}

#[test]
fn a_header_reports_architecture_shapes_and_quantisation() {
    let path = write_gguf(
        "basic.gguf",
        &[
            ("general.architecture", Value::String("sd".to_string())),
            ("general.name", Value::String("test".to_string())),
        ],
        &[
            ("a.weight", vec![128, 256], GgmlDType::Q4K),
            ("b.weight", vec![128, 256], GgmlDType::Q4K),
            ("norm.weight", vec![256], GgmlDType::F32),
        ],
    );

    let info = GgufInfo::open(&path).expect("reading the header");
    assert_eq!(info.architecture(), Some("sd"));
    assert_eq!(info.get_str("general.name"), Some("test"));

    assert_eq!(info.tensors.len(), 3);
    let (shape, dtype) = info.tensors.get("a.weight").expect("a.weight");
    assert_eq!(shape, &vec![128, 256]);
    assert_eq!(*dtype, GgmlDType::Q4K);
    assert_eq!(info.parameter_count(), 128 * 256 * 2 + 256);
}

#[test]
fn quantisation_is_reported_as_a_spread_not_a_single_type() {
    // Real k-quant checkpoints keep norms and embeddings at higher precision,
    // so "is this Q4_K" has no single answer, and a caller deciding what it
    // can load needs the breakdown rather than a label.
    let path = write_gguf(
        "mixed.gguf",
        &[("general.architecture", Value::String("sd".to_string()))],
        &[
            ("a", vec![128, 256], GgmlDType::Q4K),
            ("b", vec![128, 256], GgmlDType::Q4K),
            ("c", vec![128, 256], GgmlDType::Q8_0),
            ("d", vec![256], GgmlDType::F32),
        ],
    );

    let info = GgufInfo::open(&path).expect("reading the header");
    let spread = info.quantisations();
    assert_eq!(spread.len(), 3, "three distinct types: {spread:?}");
    assert_eq!(spread[0], (GgmlDType::Q4K, 2), "commonest first");
    let total: usize = spread.iter().map(|(_, n)| n).sum();
    assert_eq!(total, info.tensors.len());
}

#[test]
fn the_ordering_of_the_spread_is_stable() {
    // Built from a HashMap and shown to users, so an unstable order would
    // make one file describe itself differently between runs.
    let path = write_gguf(
        "stable.gguf",
        &[("general.architecture", Value::String("sd".to_string()))],
        &[
            ("a", vec![128, 256], GgmlDType::Q4K),
            ("b", vec![128, 256], GgmlDType::Q8_0),
            ("c", vec![128, 256], GgmlDType::Q5K),
        ],
    );
    let info = GgufInfo::open(&path).expect("reading the header");
    let first = info.quantisations();
    for _ in 0..8 {
        assert_eq!(info.quantisations(), first, "ordering must not vary");
    }
}

#[test]
fn a_missing_file_and_a_wrong_extension_are_distinguished() {
    let dir = std::env::temp_dir().join("sdrs-gguf-tests");
    std::fs::create_dir_all(&dir).unwrap();

    let missing = dir.join("definitely-absent.gguf");
    let _ = std::fs::remove_file(&missing);
    assert!(
        matches!(
            GgufInfo::open(&missing),
            Err(sd_loader::LoadError::NotFound(_))
        ),
        "a missing file should say so, not fail as a parse error"
    );

    let wrong = dir.join("model.safetensors");
    std::fs::write(&wrong, b"not a gguf").unwrap();
    assert!(
        matches!(
            GgufInfo::open(&wrong),
            Err(sd_loader::LoadError::Unsupported { .. })
        ),
        "the extension check should reject before parsing"
    );
}

#[test]
fn absent_metadata_is_absent_rather_than_defaulted() {
    // A file with no architecture key is one we cannot identify. Returning a
    // plausible default would send a caller to the wrong loader.
    let path = write_gguf(
        "bare.gguf",
        &[("general.name", Value::String("anonymous".to_string()))],
        &[("a", vec![128, 256], GgmlDType::Q4K)],
    );
    let info = GgufInfo::open(&path).expect("reading the header");
    assert_eq!(info.architecture(), None);
    assert_eq!(info.get_str("general.name"), Some("anonymous"));
    assert_eq!(info.get_str("nothing.here"), None);
}
