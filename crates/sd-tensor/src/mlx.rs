//! The MLX backend, behind the `mlx` feature.
//!
//! This is the migration recorded in `docs/handoff.md`: decision (b), a native
//! MLX-shaped surface rather than a candle-compatible shim. It lives in
//! `sd-tensor` and nowhere else, the rule `scripts/check-seam.sh` already
//! enforces for candle.
//!
//! **Lazy, and deliberately visible.** MLX builds a graph and computes nothing
//! until [`eval`]. That is where its speed comes from — the measured 2.53x on
//! SD 1.5 is fusion across ops, and a wrapper that evaluated eagerly to feel
//! like candle would hand it straight back. So operations return unevaluated
//! handles and the caller says when to synchronise. Reading data forces an
//! eval, because there is nothing else it could mean.
//!
//! **Channels last.** MLX convolutions take NHWC with `(out, kh, kw, in)`
//! weights, where candle takes NCHW with `(out, in, kh, kw)`. Nothing here
//! hides that: [`Array::conv2d`] documents the layout it wants and the models
//! carry NHWC rather than paying a transpose per call. This is the largest
//! single consequence of the move for code above the seam.
//!
//! Bindings are hand-written. `bindgen` would put a libclang dependency in
//! every build of this crate to generate the declarations below.

use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr};
use std::fmt;
use std::sync::Once;

use crate::{Error, Result};

// -- raw FFI ---------------------------------------------------------------
//
// `mlx_array` and friends are each one opaque pointer in a struct; the layout
// is `{ void* ctx; }` and must stay `repr(C)` for that to hold.

#[repr(C)]
#[derive(Copy, Clone)]
struct mlx_array {
    ctx: *mut c_void,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct mlx_stream {
    ctx: *mut c_void,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct mlx_vector_array {
    ctx: *mut c_void,
}

/// Positions in `mlx_dtype`, which is declared in this order in
/// `mlx/c/array.h`. These are indices, not arbitrary tags.
const MLX_FLOAT16: i32 = 9;
const MLX_FLOAT32: i32 = 10;

#[link(name = "mlxc")]
unsafe extern "C" {
    // lifecycle and inspection
    fn mlx_array_new_data(data: *const c_void, shape: *const i32, dim: i32, dtype: i32)
        -> mlx_array;
    fn mlx_array_new_float32(val: f32) -> mlx_array;
    fn mlx_array_free(arr: mlx_array) -> i32;
    fn mlx_array_size(arr: mlx_array) -> usize;
    fn mlx_array_ndim(arr: mlx_array) -> usize;
    fn mlx_array_shape(arr: mlx_array) -> *const i32;
    fn mlx_array_dtype(arr: mlx_array) -> i32;
    fn mlx_array_data_float32(arr: mlx_array) -> *const f32;

    // elementwise
    fn mlx_add(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> i32;
    fn mlx_subtract(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> i32;
    fn mlx_multiply(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> i32;
    fn mlx_divide(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> i32;
    fn mlx_sigmoid(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> i32;
    fn mlx_erf(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> i32;
    fn mlx_sqrt(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> i32;

    // shape
    fn mlx_reshape(
        res: *mut mlx_array,
        a: mlx_array,
        shape: *const i32,
        shape_num: usize,
        s: mlx_stream,
    ) -> i32;
    fn mlx_transpose_axes(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const i32,
        axes_num: usize,
        s: mlx_stream,
    ) -> i32;
    fn mlx_astype(res: *mut mlx_array, a: mlx_array, dtype: i32, s: mlx_stream) -> i32;
    fn mlx_contiguous(
        res: *mut mlx_array,
        a: mlx_array,
        allow_col_major: bool,
        s: mlx_stream,
    ) -> i32;
    fn _mlx_array_is_contiguous(res: *mut bool, arr: mlx_array) -> i32;

    // linear algebra and reductions
    fn mlx_matmul(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> i32;
    fn mlx_sum_axes(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const i32,
        axes_num: usize,
        keepdims: bool,
        s: mlx_stream,
    ) -> i32;
    fn mlx_mean_axes(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const i32,
        axes_num: usize,
        keepdims: bool,
        s: mlx_stream,
    ) -> i32;
    fn mlx_softmax_axis(
        res: *mut mlx_array,
        a: mlx_array,
        axis: i32,
        precise: bool,
        s: mlx_stream,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn mlx_conv2d(
        res: *mut mlx_array,
        input: mlx_array,
        weight: mlx_array,
        stride_0: i32,
        stride_1: i32,
        padding_0: i32,
        padding_1: i32,
        dilation_0: i32,
        dilation_1: i32,
        groups: i32,
        s: mlx_stream,
    ) -> i32;

    // streams, evaluation, errors
    fn mlx_default_gpu_stream_new() -> mlx_stream;
    fn mlx_stream_free(s: mlx_stream) -> i32;
    fn mlx_vector_array_new_value(val: mlx_array) -> mlx_vector_array;
    fn mlx_vector_array_free(vec: mlx_vector_array) -> i32;
    fn mlx_eval(outputs: mlx_vector_array) -> i32;
    fn mlx_set_error_handler(
        handler: Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
        data: *mut c_void,
        dtor: Option<unsafe extern "C" fn(*mut c_void)>,
    );
}

// -- errors ----------------------------------------------------------------
//
// `mlx-c` reports failures through a handler installed once per process: the
// call returns a status int and the message goes somewhere else entirely. The
// default handler prints to stderr and keeps going, so without this the text
// is lost and the caller gets a bare number.
//
// The handler runs synchronously on the thread that raised the error — mlx-c
// catches the C++ exception inside the same call the caller made — so a
// thread-local reunites the message with the status. If that ever stops
// holding, the symptom is a `None` here and a status with no text, which is
// exactly what this code produced before the trampoline existed.

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

unsafe extern "C" fn error_handler(msg: *const c_char, _data: *mut c_void) {
    let text = if msg.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(msg) }.to_string_lossy().into_owned()
    };
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(text));
}

/// Install the handler once. Every entry point calls this before touching MLX.
fn init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        mlx_set_error_handler(Some(error_handler), std::ptr::null_mut(), None);
    });
}

fn take_last_error() -> Option<String> {
    LAST_ERROR.with(|slot| slot.borrow_mut().take())
}

fn check(status: i32, what: &str) -> Result<()> {
    if status == 0 {
        // Drop anything the handler recorded for a call that then succeeded,
        // so it cannot be misattributed to a later failure.
        let _ = take_last_error();
        return Ok(());
    }
    Err(match take_last_error() {
        Some(msg) if !msg.is_empty() => Error::Msg(format!("mlx: {what}: {msg}")),
        _ => Error::Msg(format!("mlx: {what} failed with status {status}")),
    })
}

// -- stream ----------------------------------------------------------------

/// The stream ops are submitted to.
///
/// `mlx-c` takes a stream per call rather than having an ambient device, so one
/// is held here instead of threading it through every signature.
pub struct Stream(mlx_stream);

impl Stream {
    /// The default GPU stream.
    pub fn gpu() -> Self {
        init();
        Self(unsafe { mlx_default_gpu_stream_new() })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { mlx_stream_free(self.0) };
    }
}

// -- array -----------------------------------------------------------------

/// An MLX array. **Unevaluated until [`eval`] or a read.**
///
/// The shape is read back from MLX rather than tracked here, so a wrapper bug
/// cannot report a shape the array does not have.
pub struct Array {
    raw: mlx_array,
}

// The handle owns a reference-counted MLX array; it is not tied to a thread.
unsafe impl Send for Array {}

impl Array {
    fn wrap(raw: mlx_array) -> Result<Self> {
        if raw.ctx.is_null() {
            return Err(Error::Msg("mlx: operation returned a null array".into()));
        }
        Ok(Self { raw })
    }

    /// Copy `data` into a new f32 array. `mlx_array_new_data` copies, so the
    /// slice does not need to outlive the call.
    pub fn from_slice_f32(data: &[f32], shape: &[usize]) -> Result<Self> {
        init();
        let n: usize = shape.iter().product();
        if n != data.len() {
            return Err(Error::Msg(format!(
                "mlx: shape {shape:?} needs {n} elements, got {}",
                data.len()
            )));
        }
        let dims: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        Self::wrap(unsafe {
            mlx_array_new_data(
                data.as_ptr().cast(),
                dims.as_ptr(),
                dims.len() as i32,
                MLX_FLOAT32,
            )
        })
    }

    /// The shape as MLX reports it.
    pub fn shape(&self) -> Vec<usize> {
        let ndim = unsafe { mlx_array_ndim(self.raw) };
        let ptr = unsafe { mlx_array_shape(self.raw) };
        if ptr.is_null() || ndim == 0 {
            return Vec::new();
        }
        unsafe { std::slice::from_raw_parts(ptr, ndim) }
            .iter()
            .map(|&d| d as usize)
            .collect()
    }

    pub fn elem_count(&self) -> usize {
        unsafe { mlx_array_size(self.raw) }
    }

    pub fn is_f32(&self) -> bool {
        (unsafe { mlx_array_dtype(self.raw) }) == MLX_FLOAT32
    }

    fn binary(
        &self,
        rhs: &Self,
        stream: &Stream,
        what: &str,
        f: unsafe extern "C" fn(*mut mlx_array, mlx_array, mlx_array, mlx_stream) -> i32,
    ) -> Result<Self> {
        let mut out = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        check(unsafe { f(&mut out, self.raw, rhs.raw, stream.0) }, what)?;
        Self::wrap(out)
    }

    fn unary(
        &self,
        stream: &Stream,
        what: &str,
        f: unsafe extern "C" fn(*mut mlx_array, mlx_array, mlx_stream) -> i32,
    ) -> Result<Self> {
        let mut out = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        check(unsafe { f(&mut out, self.raw, stream.0) }, what)?;
        Self::wrap(out)
    }

    /// Elementwise sum. Broadcasting follows MLX's rules, not candle's.
    pub fn add(&self, rhs: &Self, stream: &Stream) -> Result<Self> {
        self.binary(rhs, stream, "add", mlx_add)
    }

    pub fn sub(&self, rhs: &Self, stream: &Stream) -> Result<Self> {
        self.binary(rhs, stream, "sub", mlx_subtract)
    }

    pub fn mul(&self, rhs: &Self, stream: &Stream) -> Result<Self> {
        self.binary(rhs, stream, "mul", mlx_multiply)
    }

    pub fn div(&self, rhs: &Self, stream: &Stream) -> Result<Self> {
        self.binary(rhs, stream, "div", mlx_divide)
    }

    pub fn matmul(&self, rhs: &Self, stream: &Stream) -> Result<Self> {
        self.binary(rhs, stream, "matmul", mlx_matmul)
    }

    pub fn sigmoid(&self, stream: &Stream) -> Result<Self> {
        self.unary(stream, "sigmoid", mlx_sigmoid)
    }

    pub fn erf(&self, stream: &Stream) -> Result<Self> {
        self.unary(stream, "erf", mlx_erf)
    }

    pub fn sqrt(&self, stream: &Stream) -> Result<Self> {
        self.unary(stream, "sqrt", mlx_sqrt)
    }

    /// SiLU / swish, `x * sigmoid(x)`.
    ///
    /// Composed rather than called as one op: MLX fuses it, which is the
    /// premise of the whole move, and `docs/roadmap.md` records that hand-fusing
    /// what the backend already fuses is where three sessions went.
    pub fn silu(&self, stream: &Stream) -> Result<Self> {
        let s = self.sigmoid(stream)?;
        self.mul(&s, stream)
    }

    pub fn reshape(&self, shape: &[usize], stream: &Stream) -> Result<Self> {
        let dims: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let mut out = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        check(
            unsafe { mlx_reshape(&mut out, self.raw, dims.as_ptr(), dims.len(), stream.0) },
            "reshape",
        )?;
        Self::wrap(out)
    }

    /// Permute axes. `axes` is a permutation of `0..ndim`.
    pub fn transpose(&self, axes: &[usize], stream: &Stream) -> Result<Self> {
        let ax: Vec<i32> = axes.iter().map(|&d| d as i32).collect();
        let mut out = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        check(
            unsafe { mlx_transpose_axes(&mut out, self.raw, ax.as_ptr(), ax.len(), stream.0) },
            "transpose",
        )?;
        Self::wrap(out)
    }

    fn reduce(
        &self,
        axes: &[usize],
        keepdims: bool,
        stream: &Stream,
        what: &str,
        f: unsafe extern "C" fn(*mut mlx_array, mlx_array, *const i32, usize, bool, mlx_stream)
            -> i32,
    ) -> Result<Self> {
        let ax: Vec<i32> = axes.iter().map(|&d| d as i32).collect();
        let mut out = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        check(
            unsafe { f(&mut out, self.raw, ax.as_ptr(), ax.len(), keepdims, stream.0) },
            what,
        )?;
        Self::wrap(out)
    }

    pub fn sum(&self, axes: &[usize], keepdims: bool, stream: &Stream) -> Result<Self> {
        self.reduce(axes, keepdims, stream, "sum", mlx_sum_axes)
    }

    pub fn mean(&self, axes: &[usize], keepdims: bool, stream: &Stream) -> Result<Self> {
        self.reduce(axes, keepdims, stream, "mean", mlx_mean_axes)
    }

    /// Softmax along `axis`.
    ///
    /// `precise` accumulates in f32 for f16 inputs. It is on: this project's
    /// golden tests are held to bounds derived from diffusers' own f32-vs-f64
    /// floor, and the cheap softmax is not worth re-deriving them for.
    pub fn softmax(&self, axis: isize, stream: &Stream) -> Result<Self> {
        let mut out = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        check(
            unsafe { mlx_softmax_axis(&mut out, self.raw, axis as i32, true, stream.0) },
            "softmax",
        )?;
        Self::wrap(out)
    }

    /// 2D convolution, **NHWC input and `(out, kh, kw, in)` weights**.
    ///
    /// This is MLX's native layout and the reason convolution measured 4.1x
    /// against candle's im2col path. Feeding NCHW here does not error; it
    /// convolves the wrong axes and returns a plausible tensor, so the layout
    /// is the caller's responsibility and the golden tests are what catch it.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &self,
        weight: &Self,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        stream: &Stream,
    ) -> Result<Self> {
        let mut out = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        check(
            unsafe {
                mlx_conv2d(
                    &mut out,
                    self.raw,
                    weight.raw,
                    stride.0 as i32,
                    stride.1 as i32,
                    padding.0 as i32,
                    padding.1 as i32,
                    dilation.0 as i32,
                    dilation.1 as i32,
                    groups as i32,
                    stream.0,
                )
            },
            "conv2d",
        )?;
        Self::wrap(out)
    }

    /// A 0-d f32 array, for broadcasting a constant into an expression.
    pub fn scalar_f32(value: f32) -> Result<Self> {
        init();
        Self::wrap(unsafe { mlx_array_new_float32(value) })
    }

    /// Group normalisation over NHWC, in the input's own dtype.
    ///
    /// `weight` and `bias` are per-channel, length `c`, or `None` for neither.
    /// The reduction is over space and within-group channels, which is what
    /// PyTorch's `GroupNorm` does and what `Transformer2DModel` and the resnets
    /// both expect.
    ///
    /// **Do not add an f32 upcast for the statistics. It was measured and it
    /// buys nothing** — recorded here so it is not tried a third time, the way
    /// `docs/roadmap.md` records the rejected `GroupNorm -> SiLU` fusion.
    ///
    /// `ml-explore/mlx-examples#404` (open since 2024-02-03) blames f16
    /// GroupNorm for all-black output, and f16 does look catastrophic next to
    /// f32: 2.9e-3 against 6.9e-7 at this project's UNet shapes, some 4000x.
    /// That comparison is misleading, because f32's accuracy is not reachable
    /// in f16 at all. Against the f16 representation floor — the float64
    /// reference rounded to f16, which is the best any f16 result could be —
    /// the picture is:
    ///
    /// ```text
    ///   shape             f16 floor   MLX f16   stats in f32
    ///   [1,64,64,320]     1.930e-3    2.938e-3    2.938e-3
    ///   [1,16,16,1280]    1.739e-3    2.979e-3    2.979e-3
    /// ```
    ///
    /// Taking the statistics in f32 changes the result by nothing at all, and
    /// MLX's own reduction already sits about 1.5x off a floor it cannot beat.
    /// Whatever #404 was in 2024, it is not a narrow accumulator in 0.29.3, and
    /// an upcast here would cost two casts and the memory an f16 model exists
    /// to save.
    pub fn group_norm(
        &self,
        groups: usize,
        eps: f32,
        weight: Option<&Self>,
        bias: Option<&Self>,
        stream: &Stream,
    ) -> Result<Self> {
        let shape = self.shape();
        let [n, h, w, c] = shape[..] else {
            return Err(Error::Msg(format!(
                "mlx: group_norm expects NHWC, got {shape:?}"
            )));
        };
        if groups == 0 || c % groups != 0 {
            return Err(Error::Msg(format!(
                "mlx: {c} channels do not divide into {groups} groups"
            )));
        }

        let grouped = self.reshape(&[n, h * w, groups, c / groups], stream)?;
        let mean = grouped.mean(&[1, 3], true, stream)?;
        let centred = grouped.sub(&mean, stream)?;
        let var = centred.mul(&centred, stream)?.mean(&[1, 3], true, stream)?;
        let denom = var
            .add(&Self::scalar_f32(eps)?, stream)?
            .sqrt(stream)?;
        let normed = centred.div(&denom, stream)?.reshape(&[n, h, w, c], stream)?;

        let scaled = match weight {
            Some(g) => normed.mul(g, stream)?,
            None => normed,
        };
        match bias {
            Some(b) => scaled.add(b, stream),
            None => Ok(scaled),
        }
    }

    /// Cast to f16.
    pub fn to_f16(&self, stream: &Stream) -> Result<Self> {
        let mut out = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        check(
            unsafe { mlx_astype(&mut out, self.raw, MLX_FLOAT16, stream.0) },
            "astype f16",
        )?;
        Self::wrap(out)
    }

    /// Cast to f32.
    pub fn to_f32(&self, stream: &Stream) -> Result<Self> {
        let mut out = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        check(
            unsafe { mlx_astype(&mut out, self.raw, MLX_FLOAT32, stream.0) },
            "astype f32",
        )?;
        Self::wrap(out)
    }

    /// True when the buffer is laid out row-major with no gaps.
    pub fn is_contiguous(&self) -> bool {
        let mut flag = false;
        let status = unsafe { _mlx_array_is_contiguous(&mut flag, self.raw) };
        status == 0 && flag
    }

    /// A row-major copy, or the same array when it is already one.
    ///
    /// Needed because `transpose` and friends return strided views.
    pub fn contiguous(&self, stream: &Stream) -> Result<Self> {
        let mut out = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        check(
            unsafe { mlx_contiguous(&mut out, self.raw, false, stream.0) },
            "contiguous",
        )?;
        Self::wrap(out)
    }

    /// Force evaluation, then copy the contents out as f32.
    ///
    /// The eval is not optional: MLX has computed nothing until asked, so
    /// reading the data pointer of an unevaluated array would read a buffer
    /// that does not exist yet.
    /// **Takes a stream because it may have to make the array contiguous.**
    /// `transpose` returns a strided view, and MLX's data pointer walks the
    /// underlying buffer rather than the logical order — so reading a
    /// transposed array without this returns the *untransposed* values, with
    /// the right shape and no error. That is a silent wrong answer of exactly
    /// the kind `docs/roadmap.md` records for candle's quantised matmul, and it
    /// is handled here rather than left to every caller to remember.
    pub fn to_vec_f32(&self, stream: &Stream) -> Result<Vec<f32>> {
        // `contiguous` before `eval`, and unconditionally. Both are lazy, so
        // asking an unevaluated array whether it is contiguous answers about a
        // buffer that does not exist yet — the check silently said "yes" and
        // returned untransposed data. MLX makes this a no-op when the array is
        // already row-major, so the branch bought nothing but the bug.
        let src = self.contiguous(stream)?;
        eval(&[&src])?;
        if !src.is_f32() {
            return Err(Error::Msg(
                "mlx: to_vec_f32 on an array that is not f32; cast with to_f32 first".into(),
            ));
        }
        let n = unsafe { mlx_array_size(src.raw) };
        let ptr = unsafe { mlx_array_data_float32(src.raw) };
        if ptr.is_null() {
            return Err(Error::Msg("mlx: array has no f32 data".into()));
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec())
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        unsafe { mlx_array_free(self.raw) };
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mlx::Array{:?}", self.shape())
    }
}

/// Evaluate `arrays`, running everything their graphs depend on.
///
/// This is the synchronisation point, and placing it is a real decision: too
/// often and the graph cannot fuse, which is the whole reason for the move.
pub fn eval(arrays: &[&Array]) -> Result<()> {
    init();
    for a in arrays {
        let vec = unsafe { mlx_vector_array_new_value(a.raw) };
        let status = unsafe { mlx_eval(vec) };
        unsafe { mlx_vector_array_free(vec) };
        check(status, "eval")?;
    }
    Ok(())
}
