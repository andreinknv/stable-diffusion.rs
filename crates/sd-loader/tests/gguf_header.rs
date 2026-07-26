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

// -- real files ------------------------------------------------------------
//
// Everything above round-trips through candle's own writer. These read files
// produced by llama.cpp, which is the coverage that round trip cannot give.
// They skip when the fixtures are absent, like every other golden test here:
//
//   python3 xtask/golden/dump_reference.py gguf --output tests/golden

fn fixture(name: &str) -> Option<std::path::PathBuf> {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/gguf")
        .join(name);
    if !p.exists() {
        eprintln!("SKIP: no {name}; see xtask/golden/README.md");
        return None;
    }
    Some(p)
}

#[test]
fn a_real_llama_cpp_checkpoint_reads() {
    let Some(p) = fixture("moe_shakespeare15M.gguf") else {
        return;
    };
    let info = GgufInfo::open(&p).expect("reading a real GGUF");
    assert_eq!(info.architecture(), Some("llama"));
    assert_eq!(info.tensors.len(), 86);
    // ~7.8 M parameters. Pinned loosely: the assertion is that we counted
    // something sane, not that this file never changes.
    let params = info.parameter_count();
    assert!(
        (7_000_000..9_000_000).contains(&params),
        "parameter count looks wrong: {params}"
    );
    // Real files carry far more metadata than anything synthetic here.
    assert!(
        info.metadata.len() > 10,
        "expected real metadata, got {} keys",
        info.metadata.len()
    );
}

#[test]
fn a_real_quantised_checkpoint_reports_a_mixed_spread() {
    // The design case for `quantisations()`, confirmed against a real file
    // rather than assumed: a Q8_0 checkpoint is not uniformly Q8_0. It keeps
    // 19 tensors at F32, which is why a single label would be a lie.
    let Some(p) = fixture("stories15M_MOE-Q8_0.gguf") else {
        return;
    };
    let info = GgufInfo::open(&p).expect("reading a real quantised GGUF");
    let spread = info.quantisations();
    assert!(
        spread.len() >= 2,
        "a real k-quant file mixes precisions; got {spread:?}"
    );
    assert_eq!(spread[0].0, GgmlDType::Q8_0, "commonest type first");
    let f32_count = spread
        .iter()
        .find(|(d, _)| *d == GgmlDType::F32)
        .map(|(_, n)| *n)
        .unwrap_or(0);
    assert!(
        f32_count > 0,
        "expected some tensors kept at full precision: {spread:?}"
    );
}

#[test]
fn a_big_endian_file_is_named_rather_than_reported_as_corrupt() {
    // HuggingFace hosts big-endian builds for s390x. candle rejects them with
    // "unsupported magic/version Gguf/50331648" — accurate, and useless: that
    // number is version 3 with its bytes reversed. A reader who sees it has
    // no way to know the file is fine and the byte order is not.
    let Some(p) = fixture("ggml-model-f16-big-endian.gguf") else {
        return;
    };
    let err = GgufInfo::open(&p).expect_err("big-endian must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("big-endian"), "should name the cause: {msg}");
    assert!(
        msg.contains("little-endian"),
        "should say what to use instead: {msg}"
    );
}

// -- dequantisation --------------------------------------------------------

#[test]
fn dequantised_size_is_not_the_file_size() {
    // The number a caller needs before loading. A Q8_0 file is ~1 byte per
    // parameter on disk and 4 bytes per parameter once expanded to f32, so
    // sizing a load from the file is wrong by roughly that factor — and for
    // Q4_K it is wrong by eight.
    let Some(p) = fixture("stories15M_MOE-Q8_0.gguf") else {
        return;
    };
    let info = GgufInfo::open(&p).expect("header");
    let on_disk = std::fs::metadata(&p).expect("stat").len();
    let expanded = info.dequantised_bytes(sd_tensor::DType::F32);

    assert!(
        expanded > on_disk * 2,
        "expanded {expanded} should dwarf the {on_disk} byte file"
    );
    assert_eq!(expanded, info.parameter_count() * 4);
}

#[test]
fn a_quantised_checkpoint_dequantises_to_usable_tensors() {
    let Some(p) = fixture("stories15M_MOE-Q8_0.gguf") else {
        return;
    };
    let dev = sd_tensor::Device::Cpu;
    let info = GgufInfo::open(&p).expect("header");
    let vb = sd_loader::gguf_var_builder(&p, sd_tensor::DType::F32, &dev)
        .expect("dequantising a real Q8_0 checkpoint");

    // Every tensor the header advertised must be fetchable at its stated
    // shape. A dequantiser that dropped or reshaped tensors would pass a
    // count check and fail here.
    let mut checked = 0;
    for (name, (shape, _)) in info.tensors.iter().take(12) {
        let t = vb
            .get(shape.clone(), name)
            .unwrap_or_else(|e| panic!("{name} at {shape:?}: {e}"));
        assert_eq!(t.dims(), shape.as_slice(), "{name}");
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "{name} dequantised to NaN or inf"
        );
        // Q8_0 of a real trained weight should not come back all zeros —
        // that is what a mis-scaled block would produce.
        assert!(v.iter().any(|x| *x != 0.0), "{name} is entirely zero");
        checked += 1;
    }
    assert!(checked > 0, "no tensors checked");
}

#[test]
fn an_f16_checkpoint_dequantises_too() {
    // F16 is not quantised, so it exercises the pass-through side of the
    // same path — a dequantiser that only handled block formats would fail.
    let Some(p) = fixture("moe_shakespeare15M.gguf") else {
        return;
    };
    let dev = sd_tensor::Device::Cpu;
    let info = GgufInfo::open(&p).expect("header");
    let vb = sd_loader::gguf_var_builder(&p, sd_tensor::DType::F32, &dev).expect("dequantising");

    let (name, (shape, _)) = info.tensors.iter().next().expect("at least one tensor");
    let t = vb.get(shape.clone(), name).expect("fetching a tensor");
    assert_eq!(t.dims(), shape.as_slice());
    assert_eq!(t.dtype(), sd_tensor::DType::F32, "requested dtype honoured");
}

#[test]
fn a_load_too_large_for_the_machine_is_refused_before_reading() {
    // The guard has to see the *dequantised* figure, not the file size. With
    // the headroom pinned to nothing, even a 16 MB file must be refused —
    // and refused before any tensor data is read.
    let Some(p) = fixture("moe_shakespeare15M.gguf") else {
        return;
    };
    // SAFETY: single-threaded test process; restored below.
    unsafe { std::env::set_var(sd_tensor::sysmem::HEADROOM_ENV, "0.0000001") };
    let result = sd_loader::gguf_var_builder(&p, sd_tensor::DType::F32, &sd_tensor::Device::Cpu);
    unsafe { std::env::remove_var(sd_tensor::sysmem::HEADROOM_ENV) };

    // VarBuilder has no Debug impl, so match on the Result rather than
    // using expect_err.
    let msg = match result {
        Ok(_) => panic!("a tiny headroom must refuse this load"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("refusing to start"),
        "should be the memory guard, not a parse error: {msg}"
    );
}
