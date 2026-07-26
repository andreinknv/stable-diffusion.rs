//! Rewrite a GGUF checkpoint at a different quantisation.
//!
//! This exists because nobody publishes k-quant Stable Diffusion 1.5. The
//! reason is structural rather than an oversight: k-quants operate on blocks
//! of 256 values along the fastest axis, and SD 1.5's UNet is built from 320-
//! and 640-channel blocks. 320 % 256 = 64, so those tensors cannot be
//! k-quantised at all. Without a way to produce the files locally, "how do
//! k-quants do on SD 1.5" is unanswerable.
//!
//! Tensors that do not divide evenly fall back to F16, which is the same
//! policy the shipped Q4_0 file uses for convolution weights (their fastest
//! axis is the 3-wide kernel, so no block quantisation applies).
//!
//! ```text
//! cargo run --release -p sd-cli --example requantise -- in.gguf out.gguf Q4_K
//! ```

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom};
use std::path::PathBuf;

use sd_tensor::gguf::{write, Content, GgmlDType, QTensor};
use sd_tensor::ops::human_bytes;
use sd_tensor::{sysmem, Device, Result};

/// Parse the names people actually type. candle spells these `Q4K`; every
/// tool that writes them spells them `Q4_K`.
fn parse_dtype(s: &str) -> Option<GgmlDType> {
    match s.to_ascii_uppercase().replace('_', "").as_str() {
        "F16" => Some(GgmlDType::F16),
        "F32" => Some(GgmlDType::F32),
        "Q40" => Some(GgmlDType::Q4_0),
        "Q41" => Some(GgmlDType::Q4_1),
        "Q50" => Some(GgmlDType::Q5_0),
        "Q51" => Some(GgmlDType::Q5_1),
        "Q80" => Some(GgmlDType::Q8_0),
        "Q2K" => Some(GgmlDType::Q2K),
        "Q3K" => Some(GgmlDType::Q3K),
        "Q4K" => Some(GgmlDType::Q4K),
        "Q5K" => Some(GgmlDType::Q5K),
        "Q6K" => Some(GgmlDType::Q6K),
        _ => None,
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let [_, src, dst, want] = args.as_slice() else {
        eprintln!("usage: requantise <in.gguf> <out.gguf> <Q4_K|Q5_K|Q6_K|Q8_0|...>");
        std::process::exit(2);
    };
    let (src, dst) = (PathBuf::from(src), PathBuf::from(dst));
    let Some(target) = parse_dtype(want) else {
        eprintln!("unknown quantisation {want:?}");
        std::process::exit(2);
    };

    let dev = Device::Cpu;
    let mut reader = File::open(&src)?;
    let content = Content::read(&mut reader)?;

    // Every quantised tensor is held until the write, so the peak is roughly
    // the output size. Ask before committing to it rather than discovering
    // the limit by thrashing.
    let planned: u64 = content
        .tensor_infos
        .values()
        .map(|i| {
            let n = i.shape.elem_count() as u64;
            let fits = i
                .shape
                .dims()
                .last()
                .is_some_and(|d| d % target.block_size() == 0);
            let dt = if fits { target } else { GgmlDType::F16 };
            n * dt.type_size() as u64 / dt.block_size() as u64
        })
        .sum();
    sysmem::check_headroom(planned, "requantised weights")?;

    let block = target.block_size();
    // BTreeMap keeps the tensor order stable, so requantising twice produces
    // byte-identical files and a diff means a real change.
    let mut out: BTreeMap<String, QTensor> = BTreeMap::new();
    let (mut converted, mut fell_back) = (0usize, 0usize);

    let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    for name in names {
        // Dequantise, requantise, and drop the f32 before the next tensor.
        // Holding them all would need the f32 footprint of the whole model.
        let dequantised = content.tensor(&mut reader, &name, &dev)?.dequantize(&dev)?;
        let divides = dequantised.dims().last().is_some_and(|d| d % block == 0);
        let dtype = if divides {
            converted += 1;
            target
        } else {
            fell_back += 1;
            GgmlDType::F16
        };
        out.insert(name, QTensor::quantize(&dequantised, dtype)?);
    }

    let metadata: Vec<(&str, &sd_tensor::gguf::Value)> = content
        .metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    let tensors: Vec<(&str, &QTensor)> = out.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let mut writer = BufWriter::new(File::create(&dst)?);
    write(&mut writer, &metadata, &tensors)?;
    let written = writer.seek(SeekFrom::End(0))?;
    drop(writer);

    println!(
        "{} -> {}  [{want}]\n  {converted} tensors quantised, {fell_back} kept F16 \
         (fastest axis not a multiple of {block})\n  {}",
        src.display(),
        dst.display(),
        human_bytes(written),
    );
    Ok(())
}
