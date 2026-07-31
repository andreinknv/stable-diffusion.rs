//! A GGUF reader that does not go through candle, for the MLX backend.
//!
//! **Why not MLX's own.** `mlx_load_gguf` exists and works for some files, but
//! it handles only part of the format: measured on this project's checkpoints,
//! Flux's `Q4_K_S` loads and SD 3.5's `Q4_K_M` fails with
//! `gguf_tensor_to_f16 failed`, because that file carries 48 `Q5_K` tensors.
//! Since SD 3.5's default quantisation is exactly the one that fails, relying
//! on it would leave the model this project prefers unloadable.
//!
//! **Why not candle's.** `sd_tensor::gguf` is candle's, and the point of this
//! module is to have a path that does not need it.
//!
//! So the container is parsed here and the blocks dequantised here. The types
//! are those this project's checkpoints actually use, counted across every
//! `.gguf` on hand: F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K. Anything else is an
//! error naming the type rather than a silent zero.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::{Error, Result};

const MAGIC: u32 = 0x4655_4747; // "GGUF"

/// Block layouts, from llama.cpp's `ggml-common.h`.
mod block {
    pub const QK_K: usize = 256;
    /// 32 values: `f16 d`, then 16 bytes of nibbles.
    pub const Q4_0: (usize, usize) = (32, 18);
    /// 32 values: `f16 d`, then 32 signed bytes.
    pub const Q8_0: (usize, usize) = (32, 34);
    /// 256 values: `f16 d`, `f16 dmin`, 12 packed scale bytes, 128 nibble bytes.
    pub const Q4_K: (usize, usize) = (QK_K, 144);
    /// Q4_K plus 32 bytes of high bits.
    pub const Q5_K: (usize, usize) = (QK_K, 176);
    /// 256 values: 128 low, 64 high, 16 signed scales, `f16 d`.
    pub const Q6_K: (usize, usize) = (QK_K, 210);
}

/// GGML type ids, as written in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}

impl GgmlType {
    fn from_id(id: u32) -> Result<Self> {
        Ok(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            other => {
                return Err(Error::Msg(format!(
                    "gguf: type {other} is not one this project's checkpoints use; \
                     the supported set is F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K"
                )))
            }
        })
    }

    /// `(values per block, bytes per block)`.
    fn block(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q4_0 => block::Q4_0,
            Self::Q8_0 => block::Q8_0,
            Self::Q4K => block::Q4_K,
            Self::Q5K => block::Q5_K,
            Self::Q6K => block::Q6_K,
        }
    }
}

/// One tensor's header entry.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// **Reversed from the file.** GGUF writes dimensions fastest-varying
    /// first; every consumer here wants row-major, so they are flipped once at
    /// parse rather than at each use.
    pub shape: Vec<usize>,
    pub kind: GgmlType,
    offset: u64,
}

impl TensorInfo {
    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }
}

struct Reader {
    file: File,
}

impl Reader {
    fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.file.read_exact(&mut b).map_err(io)?;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.file.read_exact(&mut b).map_err(io)?;
        Ok(u64::from_le_bytes(b))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        let mut b = vec![0u8; n];
        self.file.read_exact(&mut b).map_err(io)?;
        String::from_utf8(b).map_err(|e| Error::Msg(format!("gguf: bad utf8 in a name: {e}")))
    }
    /// Skip one metadata value. The models here need none of it, and the graph
    /// shape comes from the caller's config rather than from the file.
    fn skip_value(&mut self, kind: u32) -> Result<()> {
        let n = match kind {
            0 | 1 | 7 => 1u64, // u8, i8, bool
            2 | 3 => 2,        // u16, i16
            4..=6 => 4,        // u32, i32, f32
            10..=12 => 8,      // u64, i64, f64
            8 => {
                let len = self.u64()?;
                self.file.seek(SeekFrom::Current(len as i64)).map_err(io)?;
                return Ok(());
            }
            9 => {
                let inner = self.u32()?;
                let count = self.u64()?;
                for _ in 0..count {
                    self.skip_value(inner)?;
                }
                return Ok(());
            }
            other => return Err(Error::Msg(format!("gguf: unknown metadata type {other}"))),
        };
        self.file.seek(SeekFrom::Current(n as i64)).map_err(io)?;
        Ok(())
    }
}

fn io(e: std::io::Error) -> Error {
    Error::Msg(format!("gguf: {e}"))
}

/// The tensor directory of a GGUF file, and where its data starts.
pub struct Gguf {
    reader: Reader,
    pub tensors: Vec<TensorInfo>,
    data_start: u64,
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(io)?;
        let mut r = Reader { file };

        if r.u32()? != MAGIC {
            return Err(Error::Msg(format!(
                "gguf: {} is not a GGUF file",
                path.display()
            )));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::Msg(format!(
                "gguf: version {version} is not supported"
            )));
        }
        let tensor_count = r.u64()? as usize;
        let kv_count = r.u64()?;

        for _ in 0..kv_count {
            let _key = r.string()?;
            let kind = r.u32()?;
            r.skip_value(kind)?;
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(r.u64()? as usize);
            }
            // GGUF writes dimensions fastest-varying first.
            shape.reverse();
            let kind = GgmlType::from_id(r.u32()?)?;
            let offset = r.u64()?;
            tensors.push(TensorInfo {
                name,
                shape,
                kind,
                offset,
            });
        }

        // Tensor data begins at the next alignment boundary. 32 is the default
        // and every file here uses it; a file declaring otherwise would need
        // `general.alignment`, which is why the value is checked rather than
        // assumed to divide.
        const ALIGN: u64 = 32;
        let here = r.file.stream_position().map_err(io)?;
        let data_start = here.div_ceil(ALIGN) * ALIGN;

        Ok(Self {
            reader: r,
            tensors,
            data_start,
        })
    }

    /// Dequantise one tensor to f32.
    pub fn dequantize(&mut self, info: &TensorInfo) -> Result<Vec<f32>> {
        let n = info.elem_count();
        let (per_block, bytes_per_block) = info.kind.block();
        if n % per_block != 0 {
            return Err(Error::Msg(format!(
                "gguf: {} has {n} elements, not a multiple of {per_block}",
                info.name
            )));
        }
        let blocks = n / per_block;
        let mut raw = vec![0u8; blocks * bytes_per_block];
        self.reader
            .file
            .seek(SeekFrom::Start(self.data_start + info.offset))
            .map_err(io)?;
        self.reader.file.read_exact(&mut raw).map_err(io)?;

        let mut out = vec![0f32; n];
        match info.kind {
            GgmlType::F32 => {
                for (i, c) in raw.chunks_exact(4).enumerate() {
                    out[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
            GgmlType::F16 => {
                for (i, c) in raw.chunks_exact(2).enumerate() {
                    out[i] = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
                }
            }
            GgmlType::Q4_0 => dequant_q4_0(&raw, &mut out),
            GgmlType::Q8_0 => dequant_q8_0(&raw, &mut out),
            GgmlType::Q4K => dequant_q4_k(&raw, &mut out),
            GgmlType::Q5K => dequant_q5_k(&raw, &mut out),
            GgmlType::Q6K => dequant_q6_k(&raw, &mut out),
        }
        Ok(out)
    }
}

/// IEEE half to single.
///
/// Written out rather than pulled from a crate: it is a dozen lines and this
/// module exists to avoid dependencies. **The subnormal branch is the one that
/// matters** — an earlier bit-twiddling version was wrong for all 2046
/// subnormals by exactly a factor of two, which showed up as a 3.05e-5
/// disagreement with candle on Q4_0 weights: small enough to look like
/// precision, structural enough to be a bug.
fn f16_to_f32(h: u16) -> f32 {
    let sign = if h >> 15 == 1 { -1.0f32 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    if exp == 0 {
        // No implicit leading one: the value is frac * 2^-24.
        sign * frac * (-24f32).exp2()
    } else if exp == 0x1f {
        if frac == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + frac / 1024.0) * ((exp - 15) as f32).exp2()
    }
}

fn half(raw: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([raw[at], raw[at + 1]]))
}

fn dequant_q4_0(raw: &[u8], out: &mut [f32]) {
    for (b, blk) in raw.chunks_exact(18).enumerate() {
        let d = half(blk, 0);
        for i in 0..16 {
            let byte = blk[2 + i];
            // Low nibbles fill the first half of the block, high the second —
            // not interleaved pairs.
            out[b * 32 + i] = d * (((byte & 0x0f) as i32) - 8) as f32;
            out[b * 32 + i + 16] = d * (((byte >> 4) as i32) - 8) as f32;
        }
    }
}

fn dequant_q8_0(raw: &[u8], out: &mut [f32]) {
    for (b, blk) in raw.chunks_exact(34).enumerate() {
        let d = half(blk, 0);
        for i in 0..32 {
            out[b * 32 + i] = d * (blk[2 + i] as i8) as f32;
        }
    }
}

/// llama.cpp's `get_scale_min_k4`: eight 6-bit scale/min pairs in twelve bytes.
///
/// The second four are split across bytes, which is the part that is easy to
/// get subtly wrong — a wrong unpack gives plausible weights with the wrong
/// dynamic range in five of every eight sub-blocks.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn dequant_q4_k(raw: &[u8], out: &mut [f32]) {
    for (b, blk) in raw.chunks_exact(144).enumerate() {
        let d = half(blk, 0);
        let dmin = half(blk, 2);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let base = b * 256;
        for j in 0..8 {
            let (sc, m) = scale_min_k4(j, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            // Sub-block j takes 32 values: the low nibbles of one 32-byte half
            // for even j, the high nibbles for odd.
            let half_idx = j / 2;
            let shift = if j % 2 == 0 { 0 } else { 4 };
            for i in 0..32 {
                let q = (qs[half_idx * 32 + i] >> shift) & 0x0f;
                out[base + j * 32 + i] = d1 * q as f32 - m1;
            }
        }
    }
}

fn dequant_q5_k(raw: &[u8], out: &mut [f32]) {
    for (b, blk) in raw.chunks_exact(176).enumerate() {
        let d = half(blk, 0);
        let dmin = half(blk, 2);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let qs = &blk[48..176];
        let base = b * 256;
        for j in 0..8 {
            let (sc, m) = scale_min_k4(j, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            let half_idx = j / 2;
            let shift = if j % 2 == 0 { 0 } else { 4 };
            for i in 0..32 {
                let lo = (qs[half_idx * 32 + i] >> shift) & 0x0f;
                // The fifth bit lives in `qh`, one bit per value per
                // sub-block, indexed by j rather than by the nibble half.
                let hi = (qh[i] >> j) & 1;
                let q = lo as u32 | ((hi as u32) << 4);
                out[base + j * 32 + i] = d1 * q as f32 - m1;
            }
        }
    }
}

fn dequant_q6_k(raw: &[u8], out: &mut [f32]) {
    for (b, blk) in raw.chunks_exact(210).enumerate() {
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        let d = half(blk, 208);
        let base = b * 256;
        // Two halves of 128 values each, each built from 64 low bytes, 32 high
        // bytes and 8 scales.
        for n in 0..2 {
            let ql = &ql[n * 64..];
            let qh = &qh[n * 32..];
            let sc = &scales[n * 8..];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0f) as i32 | ((qh[l] & 3) as i32) << 4) - 32;
                let q2 = ((ql[l + 32] & 0x0f) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) - 32;
                let q3 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) - 32;
                let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) - 32;
                let o = base + n * 128 + l;
                out[o] = d * (sc[is] as i8) as f32 * q1 as f32;
                out[o + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                out[o + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                out[o + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
}

/// One tensor's shape and dequantised values.
pub type Dequantized = (Vec<usize>, Vec<f32>);

/// Every tensor in a GGUF file, dequantised to f32 and keyed by name.
pub fn load(path: &Path) -> Result<HashMap<String, Dequantized>> {
    let mut g = Gguf::open(path)?;
    let infos = g.tensors.clone();
    let mut out = HashMap::with_capacity(infos.len());
    for info in &infos {
        let values = g.dequantize(info)?;
        out.insert(info.name.clone(), (info.shape.clone(), values));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All 65536 half-precision bit patterns against Rust's own conversion.
    ///
    /// Cheap and total, and it would have caught the subnormal bug instantly:
    /// the first wrong value is `0x0001`, the smallest positive subnormal.
    #[test]
    fn f16_conversion_is_exact_for_every_bit_pattern() {
        for bits in 0u32..=0xffff {
            let h = bits as u16;
            let want = half_reference(h);
            let got = f16_to_f32(h);
            if want.is_nan() {
                assert!(got.is_nan(), "{h:#06x}: expected NaN, got {got}");
            } else {
                assert_eq!(got.to_bits(), want.to_bits(), "{h:#06x}: {got} != {want}");
            }
        }
    }

    /// A reference conversion written the other way round — via the exponent
    /// arithmetic rather than the arithmetic form — so the test is not simply
    /// the implementation restated.
    fn half_reference(h: u16) -> f32 {
        let sign = ((h >> 15) & 1) as u32;
        let exp = ((h >> 10) & 0x1f) as u32;
        let frac = (h & 0x3ff) as u32;
        if exp == 0 {
            let v = frac as f32 * 2f32.powi(-24);
            return if sign == 1 { -v } else { v };
        }
        if exp == 0x1f {
            let bits = (sign << 31) | (0xff << 23) | (frac << 13);
            return f32::from_bits(bits);
        }
        let bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
        f32::from_bits(bits)
    }
}
