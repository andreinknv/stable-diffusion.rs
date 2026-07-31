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
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt;
use std::path::Path;
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

/// `mlx_optional_int`. Passed by value, so the layout has to match exactly:
/// an `int` then a `bool`, not a tagged union.
#[repr(C)]
#[derive(Copy, Clone)]
struct mlx_optional_int {
    value: i32,
    has_value: bool,
}

impl mlx_optional_int {
    fn some(v: i32) -> Self {
        Self {
            value: v,
            has_value: true,
        }
    }
}

/// `mlx_optional_dtype`, which quantisation takes for the output type.
#[repr(C)]
#[derive(Copy, Clone)]
struct mlx_optional_dtype {
    value: i32,
    has_value: bool,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct mlx_map_string_to_array {
    ctx: *mut c_void,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct mlx_map_string_to_string {
    ctx: *mut c_void,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct mlx_map_string_to_array_iterator {
    ctx: *mut c_void,
}

/// Positions in `mlx_dtype`, which is declared in this order in
/// `mlx/c/array.h`. These are indices, not arbitrary tags.
const MLX_INT32: i32 = 7;
const MLX_FLOAT16: i32 = 9;
const MLX_FLOAT32: i32 = 10;

#[link(name = "mlxc")]
unsafe extern "C" {
    // lifecycle and inspection
    fn mlx_array_new_data(
        data: *const c_void,
        shape: *const i32,
        dim: i32,
        dtype: i32,
    ) -> mlx_array;
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
    fn mlx_abs(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> i32;
    fn mlx_maximum(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> i32;
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
    fn mlx_concatenate_axis(
        res: *mut mlx_array,
        arrays: mlx_vector_array,
        axis: i32,
        s: mlx_stream,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn mlx_slice(
        res: *mut mlx_array,
        a: mlx_array,
        start: *const i32,
        start_num: usize,
        stop: *const i32,
        stop_num: usize,
        strides: *const i32,
        strides_num: usize,
        s: mlx_stream,
    ) -> i32;
    fn mlx_vector_array_new_data(data: *const mlx_array, size: usize) -> mlx_vector_array;
    fn mlx_take_axis(
        res: *mut mlx_array,
        a: mlx_array,
        indices: mlx_array,
        axis: i32,
        s: mlx_stream,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn mlx_pad(
        res: *mut mlx_array,
        a: mlx_array,
        axes: *const i32,
        axes_num: usize,
        low_pad_size: *const i32,
        low_pad_size_num: usize,
        high_pad_size: *const i32,
        high_pad_size_num: usize,
        pad_value: mlx_array,
        mode: *const c_char,
        s: mlx_stream,
    ) -> i32;
    fn mlx_broadcast_to(
        res: *mut mlx_array,
        a: mlx_array,
        shape: *const i32,
        shape_num: usize,
        s: mlx_stream,
    ) -> i32;

    fn mlx_tanh(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> i32;
    fn mlx_exp(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> i32;
    fn mlx_cos(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> i32;
    fn mlx_sin(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> i32;
    fn mlx_log(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> i32;
    fn mlx_rsqrt(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> i32;

    fn mlx_fast_rms_norm(
        res: *mut mlx_array,
        x: mlx_array,
        weight: mlx_array,
        eps: f32,
        s: mlx_stream,
    ) -> i32;
    fn mlx_fast_layer_norm(
        res: *mut mlx_array,
        x: mlx_array,
        weight: mlx_array,
        bias: mlx_array,
        eps: f32,
        s: mlx_stream,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn mlx_fast_scaled_dot_product_attention(
        res: *mut mlx_array,
        queries: mlx_array,
        keys: mlx_array,
        values: mlx_array,
        scale: f32,
        mask_mode: *const c_char,
        mask_arr: mlx_array,
        sinks: mlx_array,
        s: mlx_stream,
    ) -> i32;

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
    fn mlx_max_axes(
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

    // weight loading
    fn mlx_load_safetensors(
        res_0: *mut mlx_map_string_to_array,
        res_1: *mut mlx_map_string_to_string,
        file: *const c_char,
        s: mlx_stream,
    ) -> i32;
    fn mlx_map_string_to_array_new() -> mlx_map_string_to_array;
    fn mlx_map_string_to_array_insert(
        map: mlx_map_string_to_array,
        key: *const c_char,
        value: mlx_array,
    ) -> i32;
    fn mlx_save_safetensors(
        file: *const c_char,
        param: mlx_map_string_to_array,
        metadata: mlx_map_string_to_string,
    ) -> i32;
    fn mlx_map_string_to_array_free(map: mlx_map_string_to_array) -> i32;
    fn mlx_map_string_to_string_new() -> mlx_map_string_to_string;
    fn mlx_map_string_to_string_free(map: mlx_map_string_to_string) -> i32;
    fn mlx_map_string_to_array_iterator_new(
        map: mlx_map_string_to_array,
    ) -> mlx_map_string_to_array_iterator;
    fn mlx_map_string_to_array_iterator_free(it: mlx_map_string_to_array_iterator) -> i32;
    fn mlx_map_string_to_array_iterator_next(
        key: *mut *const c_char,
        value: *mut mlx_array,
        it: mlx_map_string_to_array_iterator,
    ) -> i32;

    // streams, evaluation, errors
    fn mlx_default_gpu_stream_new() -> mlx_stream;
    fn mlx_default_cpu_stream_new() -> mlx_stream;
    fn mlx_stream_free(s: mlx_stream) -> i32;
    fn mlx_vector_array_new_value(val: mlx_array) -> mlx_vector_array;
    fn mlx_vector_array_size(vec: mlx_vector_array) -> usize;
    fn mlx_vector_array_get(res: *mut mlx_array, vec: mlx_vector_array, index: usize) -> i32;

    // quantisation
    #[allow(clippy::too_many_arguments)]
    fn mlx_quantize(
        res: *mut mlx_vector_array,
        w: mlx_array,
        group_size: mlx_optional_int,
        bits: mlx_optional_int,
        mode: *const c_char,
        global_scale: mlx_array,
        s: mlx_stream,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn mlx_dequantize(
        res: *mut mlx_array,
        w: mlx_array,
        scales: mlx_array,
        biases: mlx_array,
        group_size: mlx_optional_int,
        bits: mlx_optional_int,
        mode: *const c_char,
        global_scale: mlx_array,
        dtype: mlx_optional_dtype,
        s: mlx_stream,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn mlx_quantized_matmul(
        res: *mut mlx_array,
        x: mlx_array,
        w: mlx_array,
        scales: mlx_array,
        biases: mlx_array,
        transpose: bool,
        group_size: mlx_optional_int,
        bits: mlx_optional_int,
        mode: *const c_char,
        s: mlx_stream,
    ) -> i32;
    fn mlx_vector_array_free(vec: mlx_vector_array) -> i32;
    fn mlx_eval(outputs: mlx_vector_array) -> i32;

    // Memory accounting. MLX tracks its own allocations, and on unified memory
    // those are not visible as resident set size in any way you can attribute
    // — which is why measuring a run with `/usr/bin/time -l` answers a
    // different question from the one being asked.
    fn mlx_get_active_memory(res: *mut usize) -> i32;
    fn mlx_get_peak_memory(res: *mut usize) -> i32;
    fn mlx_get_cache_memory(res: *mut usize) -> i32;
    fn mlx_reset_peak_memory() -> i32;
    fn mlx_set_wired_limit(res: *mut usize, limit: usize) -> i32;
    fn mlx_set_cache_limit(res: *mut usize, limit: usize) -> i32;
    fn mlx_clear_cache() -> i32;
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
        unsafe { CStr::from_ptr(msg) }
            .to_string_lossy()
            .into_owned()
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

    /// The default CPU stream.
    ///
    /// Reading a `.safetensors` file needs this: MLX's `Load` primitive has no
    /// GPU implementation, and submitting it to the GPU stream fails at eval
    /// with `[Load::eval_gpu] Not implemented`, long after the call that
    /// chose the stream.
    pub fn cpu() -> Self {
        init();
        Self(unsafe { mlx_default_cpu_stream_new() })
    }

    /// The stream for a named device.
    ///
    /// The reason this exists rather than callers writing `Stream::gpu()`: a
    /// device that can only be chosen at the call site cannot be chosen at all
    /// by whoever is running the program. Every pipeline takes a [`Device`] and
    /// passes it here, so `--cpu` is a flag rather than a rebuild.
    pub fn for_device(device: Device) -> Self {
        match device {
            Device::Gpu => Self::gpu(),
            Device::Cpu => Self::cpu(),
        }
    }
}

/// Which device to run on.
///
/// **MLX's CPU backend is real, not a stub** — it computes, and on this
/// machine it is what reads every `.safetensors` file, because MLX's `Load`
/// primitive has no GPU implementation. What it is not is *fast*: a diffusion
/// step is thousands of matmuls and the GPU is the point.
///
/// It is exposed because "GPU or nothing" should be the user's choice to make,
/// not a constant compiled into 105 call sites — and because a machine whose
/// GPU is busy or absent should still be able to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Gpu,
    Cpu,
}

impl std::str::FromStr for Device {
    type Err = crate::Error;

    fn from_str(s: &str) -> crate::Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "gpu" | "metal" => Ok(Self::Gpu),
            "cpu" => Ok(Self::Cpu),
            other => Err(crate::Error::Msg(format!(
                "unknown device {other:?}; try `gpu` or `cpu`"
            ))),
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
        })
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

    /// Elementwise absolute value.
    pub fn abs(&self, stream: &Stream) -> Result<Self> {
        self.unary(stream, "abs", mlx_abs)
    }

    pub fn sigmoid(&self, stream: &Stream) -> Result<Self> {
        self.unary(stream, "sigmoid", mlx_sigmoid)
    }

    /// Elementwise `max(self, rhs)`.
    pub fn maximum(&self, rhs: &Self, stream: &Stream) -> Result<Self> {
        self.binary(rhs, stream, "maximum", mlx_maximum)
    }

    /// ReLU, as `max(x, 0)`. `mlx-c` exposes no relu of its own.
    pub fn relu(&self, stream: &Stream) -> Result<Self> {
        self.maximum(&Self::scalar_f32(0.0)?, stream)
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
        f: unsafe extern "C" fn(
            *mut mlx_array,
            mlx_array,
            *const i32,
            usize,
            bool,
            mlx_stream,
        ) -> i32,
    ) -> Result<Self> {
        let ax: Vec<i32> = axes.iter().map(|&d| d as i32).collect();
        let mut out = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        check(
            unsafe {
                f(
                    &mut out,
                    self.raw,
                    ax.as_ptr(),
                    ax.len(),
                    keepdims,
                    stream.0,
                )
            },
            what,
        )?;
        Self::wrap(out)
    }

    pub fn sum(&self, axes: &[usize], keepdims: bool, stream: &Stream) -> Result<Self> {
        self.reduce(axes, keepdims, stream, "sum", mlx_sum_axes)
    }

    /// Reduce by maximum over `axes`. An inpainting mask is downsampled with
    /// this rather than with [`Self::mean`] — see `mlx::sample::latent_mask`
    /// for why the two are not interchangeable.
    pub fn max(&self, axes: &[usize], keepdims: bool, stream: &Stream) -> Result<Self> {
        self.reduce(axes, keepdims, stream, "max", mlx_max_axes)
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
        let denom = var.add(&Self::scalar_f32(eps)?, stream)?.sqrt(stream)?;
        let normed = centred
            .div(&denom, stream)?
            .reshape(&[n, h, w, c], stream)?;

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

    pub fn tanh(&self, stream: &Stream) -> Result<Self> {
        self.unary(stream, "tanh", mlx_tanh)
    }

    pub fn exp(&self, stream: &Stream) -> Result<Self> {
        self.unary(stream, "exp", mlx_exp)
    }

    pub fn cos(&self, stream: &Stream) -> Result<Self> {
        self.unary(stream, "cos", mlx_cos)
    }

    pub fn sin(&self, stream: &Stream) -> Result<Self> {
        self.unary(stream, "sin", mlx_sin)
    }

    pub fn log(&self, stream: &Stream) -> Result<Self> {
        self.unary(stream, "log", mlx_log)
    }

    pub fn rsqrt(&self, stream: &Stream) -> Result<Self> {
        self.unary(stream, "rsqrt", mlx_rsqrt)
    }

    /// Exact GELU, `0.5 * x * (1 + erf(x / sqrt(2)))`.
    ///
    /// The erf form, not the tanh approximation, because diffusers' `GEGLU`
    /// uses `F.gelu` and the golden tests are held to diffusers.
    ///
    /// **Watch the left tail.** `docs/handoff.md` records candle's `gelu_erf`
    /// returning *exactly zero* below about -6, because forming `1 + erf(u)` by
    /// subtraction rounds the tail away where the truth is -5.9e-9. This has
    /// the same shape and the same exposure; `gelu_left_tail_is_not_flat`
    /// measures it rather than assuming either way.
    pub fn gelu(&self, stream: &Stream) -> Result<Self> {
        let inv_sqrt2 = Self::scalar_f32(std::f32::consts::FRAC_1_SQRT_2)?;
        let one = Self::scalar_f32(1.0)?;
        let half = Self::scalar_f32(0.5)?;
        let e = self.mul(&inv_sqrt2, stream)?.erf(stream)?;
        self.mul(&half, stream)?.mul(&e.add(&one, stream)?, stream)
    }

    /// Tanh-approximate GELU — `gelu_new` in transformers.
    ///
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    ///
    /// **Not interchangeable with [`Array::gelu`].** The two differ by ~1e-3,
    /// which is far above T5's noise floor and shows up as a systematic drift
    /// through the stack rather than as a visible break. T5 v1.1's gated
    /// feed-forward wants this one; the UNet's GEGLU wants the erf form.
    pub fn gelu_approx(&self, stream: &Stream) -> Result<Self> {
        let half = Self::scalar_f32(0.5)?;
        let one = Self::scalar_f32(1.0)?;
        let c = Self::scalar_f32((2.0f32 / std::f32::consts::PI).sqrt())?;
        let k = Self::scalar_f32(0.044715)?;

        let cubed = self.mul(self, stream)?.mul(self, stream)?;
        let inner = self
            .add(&cubed.mul(&k, stream)?, stream)?
            .mul(&c, stream)?
            .tanh(stream)?;
        self.mul(&half, stream)?
            .mul(&inner.add(&one, stream)?, stream)
    }

    /// Layer normalisation over the last axis, via MLX's fused kernel.
    ///
    /// `weight` and `bias` are optional; `None` means no affine term.
    pub fn layer_norm(
        &self,
        weight: Option<&Self>,
        bias: Option<&Self>,
        eps: f32,
        stream: &Stream,
    ) -> Result<Self> {
        let null = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        let mut out = null;
        check(
            unsafe {
                mlx_fast_layer_norm(
                    &mut out,
                    self.raw,
                    weight.map_or(null, |w| w.raw),
                    bias.map_or(null, |b| b.raw),
                    eps,
                    stream.0,
                )
            },
            "layer_norm",
        )?;
        Self::wrap(out)
    }

    /// Scaled dot-product attention over `[batch, heads, seq, head_dim]`.
    ///
    /// `scale` is applied to the query before the product, as diffusers does.
    /// Unmasked — the UNet's transformer attends over everything, which is the
    /// same assumption `unet/attention.rs` makes on the candle path.
    /// RMS normalisation over the last axis: no mean subtraction, no bias.
    ///
    /// **Not LayerNorm.** T5 uses this, and substituting LayerNorm gives
    /// plausible activations and a wrong result. `weight` is optional.
    pub fn rms_norm(&self, weight: Option<&Self>, eps: f32, stream: &Stream) -> Result<Self> {
        let null = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        let mut out = null;
        check(
            unsafe {
                mlx_fast_rms_norm(
                    &mut out,
                    self.raw,
                    weight.map_or(null, |w| w.raw),
                    eps,
                    stream.0,
                )
            },
            "rms_norm",
        )?;
        Self::wrap(out)
    }

    /// Attention with an **additive** mask, broadcast over the score matrix.
    ///
    /// T5's relative position bias arrives this way rather than as a boolean
    /// mask.
    pub fn sdpa_masked(
        &self,
        keys: &Self,
        values: &Self,
        scale: f32,
        mask: &Self,
        stream: &Stream,
    ) -> Result<Self> {
        let null = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        let mut out = null;
        check(
            unsafe {
                mlx_fast_scaled_dot_product_attention(
                    &mut out,
                    self.raw,
                    keys.raw,
                    values.raw,
                    scale,
                    c"array".as_ptr(),
                    mask.raw,
                    null,
                    stream.0,
                )
            },
            "sdpa_masked",
        )?;
        Self::wrap(out)
    }

    /// Causal scaled dot-product attention, for CLIP's text tower.
    pub fn sdpa_causal(
        &self,
        keys: &Self,
        values: &Self,
        scale: f32,
        stream: &Stream,
    ) -> Result<Self> {
        self.sdpa_with_mode(keys, values, scale, c"causal", stream)
    }

    pub fn sdpa(&self, keys: &Self, values: &Self, scale: f32, stream: &Stream) -> Result<Self> {
        self.sdpa_with_mode(keys, values, scale, c"", stream)
    }

    fn sdpa_with_mode(
        &self,
        keys: &Self,
        values: &Self,
        scale: f32,
        mode: &std::ffi::CStr,
        stream: &Stream,
    ) -> Result<Self> {
        let null = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        let mut out = null;
        check(
            unsafe {
                mlx_fast_scaled_dot_product_attention(
                    &mut out,
                    self.raw,
                    keys.raw,
                    values.raw,
                    scale,
                    // Not NULL — mlx-c reads this pointer and a null one
                    // segfaults rather than meaning no mask. The accepted
                    // values are 'causal', 'array' or ''; "none" is rejected.
                    mode.as_ptr(),
                    null,
                    null,
                    stream.0,
                )
            },
            "sdpa",
        )?;
        Self::wrap(out)
    }

    /// Take `len` entries along `axis`, starting at `start` — candle's `narrow`.
    pub fn narrow(&self, axis: usize, start: usize, len: usize, stream: &Stream) -> Result<Self> {
        let shape = self.shape();
        if axis >= shape.len() {
            return Err(Error::Msg(format!(
                "mlx: narrow axis {axis} out of range for {shape:?}"
            )));
        }
        if start + len > shape[axis] {
            return Err(Error::Msg(format!(
                "mlx: narrow {start}..{} exceeds dim {axis} of {shape:?}",
                start + len
            )));
        }
        let starts: Vec<i32> = shape
            .iter()
            .enumerate()
            .map(|(i, _)| if i == axis { start as i32 } else { 0 })
            .collect();
        let stops: Vec<i32> = shape
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if i == axis {
                    (start + len) as i32
                } else {
                    d as i32
                }
            })
            .collect();
        let strides: Vec<i32> = vec![1; shape.len()];
        let mut out = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        check(
            unsafe {
                mlx_slice(
                    &mut out,
                    self.raw,
                    starts.as_ptr(),
                    starts.len(),
                    stops.as_ptr(),
                    stops.len(),
                    strides.as_ptr(),
                    strides.len(),
                    stream.0,
                )
            },
            "narrow",
        )?;
        Self::wrap(out)
    }

    /// An i32 array, for indices.
    pub fn from_slice_i32(data: &[i32], shape: &[usize]) -> Result<Self> {
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
                MLX_INT32,
            )
        })
    }

    /// Gather rows of `self` along `axis` at `indices` — an embedding lookup
    /// when `axis` is 0 and `self` is a `[vocab, dim]` table.
    pub fn take(&self, indices: &Self, axis: usize, stream: &Stream) -> Result<Self> {
        let mut out = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        check(
            unsafe { mlx_take_axis(&mut out, self.raw, indices.raw, axis as i32, stream.0) },
            "take",
        )?;
        Self::wrap(out)
    }

    /// CLIP's activation, `x * sigmoid(1.702 * x)`.
    ///
    /// Not the erf GELU: `clip/text_encoder.rs` selects `QuickGelu` for SD 1.5
    /// and the two differ by enough to move the encoder output.
    pub fn quick_gelu(&self, stream: &Stream) -> Result<Self> {
        let k = Self::scalar_f32(1.702)?;
        let gate = self.mul(&k, stream)?.sigmoid(stream)?;
        self.mul(&gate, stream)
    }

    /// Wrap `axes` by `amount`, taking the padding from the opposite edge.
    ///
    /// Circular padding, which is what makes an image tile: a convolution at
    /// the left edge sees the right edge as its neighbour, so the two agree
    /// where they meet. Zero padding instead makes every edge believe it
    /// borders black, which is why an untreated image has a visible seam and a
    /// darker border.
    ///
    /// Built from `narrow` and `concat` rather than a padding mode because MLX
    /// exposes only zero padding in its convolution. **The convolution must
    /// then be called with padding 0** — padding here *and* there pads twice.
    pub fn pad_circular(&self, axes: &[usize], amount: usize, stream: &Stream) -> Result<Self> {
        if amount == 0 {
            return self.contiguous(stream);
        }
        let mut out = self.contiguous(stream)?;
        for &axis in axes {
            let len = *out
                .shape()
                .get(axis)
                .ok_or_else(|| Error::Msg(format!("mlx: pad_circular axis {axis} out of range")))?;
            // Wrapping by more than the axis is long would need the input
            // repeated, which no convolution in this project asks for and
            // which would silently produce a tiled input rather than a padded
            // one.
            if amount > len {
                return Err(Error::Msg(format!(
                    "mlx: cannot wrap axis {axis} of length {len} by {amount}"
                )));
            }
            let tail = out.narrow(axis, len - amount, amount, stream)?;
            let head = out.narrow(axis, 0, amount, stream)?;
            out = concat(&[&tail, &out, &head], axis, stream)?;
        }
        Ok(out)
    }

    /// Zero-pad `axes` by `(low, high)` each.
    ///
    /// Needed because MLX convolutions take one symmetric padding per spatial
    /// axis, and the VAE encoder's downsample is **asymmetric** — one row at
    /// the bottom and one column at the right, none at the top or left, which
    /// is what diffusers does. A symmetric pad runs, produces the right shape,
    /// and shifts the image half a pixel per downsample.
    pub fn pad(
        &self,
        axes: &[usize],
        low: &[usize],
        high: &[usize],
        stream: &Stream,
    ) -> Result<Self> {
        if axes.len() != low.len() || axes.len() != high.len() {
            return Err(Error::Msg(
                "mlx: pad needs one low and one high per axis".into(),
            ));
        }
        let ax: Vec<i32> = axes.iter().map(|&a| a as i32).collect();
        let lo: Vec<i32> = low.iter().map(|&a| a as i32).collect();
        let hi: Vec<i32> = high.iter().map(|&a| a as i32).collect();
        let zero = Self::scalar_f32(0.0)?;
        let mut out = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        check(
            unsafe {
                mlx_pad(
                    &mut out,
                    self.raw,
                    ax.as_ptr(),
                    ax.len(),
                    lo.as_ptr(),
                    lo.len(),
                    hi.as_ptr(),
                    hi.len(),
                    zero.raw,
                    c"constant".as_ptr(),
                    stream.0,
                )
            },
            "pad",
        )?;
        Self::wrap(out)
    }

    /// Broadcast to `shape`, which must be compatible with the current one.
    pub fn broadcast_to(&self, shape: &[usize], stream: &Stream) -> Result<Self> {
        let dims: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let mut out = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        check(
            unsafe { mlx_broadcast_to(&mut out, self.raw, dims.as_ptr(), dims.len(), stream.0) },
            "broadcast_to",
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

/// Every tensor in a `.safetensors` file, by name.
///
/// MLX reads the container itself, so this replaces `VarBuilder`'s job of
/// walking a prefix tree — the models name what they want and get it, and a
/// missing key is an error at load rather than a silently zero weight.
pub fn load_safetensors(path: &Path) -> Result<HashMap<String, Array>> {
    init();
    let c_path = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|e| Error::Msg(format!("mlx: path is not a valid C string: {e}")))?;
    // CPU, not GPU: see `Stream::cpu`. Later ops on these arrays may use the
    // GPU stream as usual; MLX carries the dependency across.
    let stream = Stream::cpu();

    let mut arrays = unsafe { mlx_map_string_to_array_new() };
    let mut meta = unsafe { mlx_map_string_to_string_new() };
    let status = unsafe { mlx_load_safetensors(&mut arrays, &mut meta, c_path.as_ptr(), stream.0) };
    unsafe { mlx_map_string_to_string_free(meta) };
    if let Err(e) = check(status, "load_safetensors") {
        unsafe { mlx_map_string_to_array_free(arrays) };
        return Err(e);
    }

    let mut out = HashMap::new();
    let it = unsafe { mlx_map_string_to_array_iterator_new(arrays) };
    loop {
        let mut key: *const c_char = std::ptr::null();
        let mut value = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        // Returns non-zero once the iterator is exhausted.
        if unsafe { mlx_map_string_to_array_iterator_next(&mut key, &mut value, it) } != 0 {
            break;
        }
        if key.is_null() || value.ctx.is_null() {
            break;
        }
        let name = unsafe { CStr::from_ptr(key) }
            .to_string_lossy()
            .into_owned();
        out.insert(name, Array { raw: value });
    }
    unsafe { mlx_map_string_to_array_iterator_free(it) };
    unsafe { mlx_map_string_to_array_free(arrays) };

    if out.is_empty() {
        return Err(Error::Msg(format!(
            "mlx: {} contained no tensors",
            path.display()
        )));
    }
    Ok(out)
}

/// Join `arrays` along `axis` — candle's `Tensor::cat`, and how the UNet's up
/// pass rejoins its skip connections.
pub fn concat(arrays: &[&Array], axis: usize, stream: &Stream) -> Result<Array> {
    if arrays.is_empty() {
        return Err(Error::Msg("mlx: concat needs at least one array".into()));
    }
    init();
    let raws: Vec<mlx_array> = arrays.iter().map(|a| a.raw).collect();
    let vec = unsafe { mlx_vector_array_new_data(raws.as_ptr(), raws.len()) };
    let mut out = mlx_array {
        ctx: std::ptr::null_mut(),
    };
    let status = unsafe { mlx_concatenate_axis(&mut out, vec, axis as i32, stream.0) };
    unsafe { mlx_vector_array_free(vec) };
    check(status, "concat")?;
    Array::wrap(out)
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

/// Write a weight map as a safetensors file.
///
/// **Every array is evaluated first.** MLX is lazy, so an unevaluated array has
/// no data to write, and saving one would produce a file of the right shape
/// holding whatever the buffer happened to contain.
pub fn save_safetensors(path: &Path, weights: &HashMap<String, Array>) -> Result<()> {
    init();
    let refs: Vec<&Array> = weights.values().collect();
    eval(&refs)?;

    let c_path = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|e| Error::Msg(format!("mlx: path is not a valid C string: {e}")))?;
    let map = unsafe { mlx_map_string_to_array_new() };
    let meta = unsafe { mlx_map_string_to_string_new() };

    let mut result = Ok(());
    for (name, array) in weights {
        let key = match CString::new(name.as_str()) {
            Ok(k) => k,
            Err(e) => {
                result = Err(Error::Msg(format!("mlx: tensor name {name:?}: {e}")));
                break;
            }
        };
        let status = unsafe { mlx_map_string_to_array_insert(map, key.as_ptr(), array.raw) };
        if let Err(e) = check(status, "map insert") {
            result = Err(e);
            break;
        }
    }
    if result.is_ok() {
        let status = unsafe { mlx_save_safetensors(c_path.as_ptr(), map, meta) };
        result = check(status, "save_safetensors");
    }
    unsafe { mlx_map_string_to_string_free(meta) };
    unsafe { mlx_map_string_to_array_free(map) };
    result
}

// -- memory -----------------------------------------------------------------

/// What MLX currently holds in live arrays.
///
/// **Not resident set size, and the difference is the point.** On unified
/// memory a GPU allocation is not distinguishable from any other page of the
/// process, so `maximum resident set size` includes the memory-mapped
/// checkpoint and MLX's allocator reserve alongside the tensors, and moves for
/// reasons that have nothing to do with the model. This is MLX's own count.
pub fn active_memory() -> Result<usize> {
    init();
    let mut out = 0usize;
    check(
        unsafe { mlx_get_active_memory(&mut out) },
        "get_active_memory",
    )?;
    Ok(out)
}

/// The high-water mark since the process started or [`reset_peak_memory`].
///
/// The number worth reporting for "will this model fit": a run's peak is what
/// has to be available, not its steady state.
pub fn peak_memory() -> Result<usize> {
    init();
    let mut out = 0usize;
    check(unsafe { mlx_get_peak_memory(&mut out) }, "get_peak_memory")?;
    Ok(out)
}

/// Memory MLX is holding for reuse rather than because anything needs it.
///
/// Counted against the process by the OS but available to MLX immediately, so
/// a peak that is mostly cache is not a peak that will fail on a smaller
/// machine.
pub fn cache_memory() -> Result<usize> {
    init();
    let mut out = 0usize;
    check(
        unsafe { mlx_get_cache_memory(&mut out) },
        "get_cache_memory",
    )?;
    Ok(out)
}

/// Start the high-water mark again from here.
///
/// For measuring one phase of a run — a load, or a single step — without the
/// phases before it setting the number.
pub fn reset_peak_memory() -> Result<()> {
    init();
    check(unsafe { mlx_reset_peak_memory() }, "reset_peak_memory")
}

/// Ask the OS to keep this many bytes of MLX's memory resident.
///
/// **The difference between a large model running and the machine swapping.**
/// On Apple silicon the GPU and CPU share one pool, so a model that fits in
/// physical memory can still be paged out under pressure from anything else on
/// the machine — and paging a weight back in mid-step costs more than the step.
/// Returns the previous limit.
///
/// Setting this above what the machine has is refused by the OS rather than by
/// MLX, so the caller should size it from
/// [`crate::sysmem::available_bytes`] rather than from the model.
pub fn set_wired_limit(bytes: usize) -> Result<usize> {
    init();
    let mut previous = 0usize;
    check(
        unsafe { mlx_set_wired_limit(&mut previous, bytes) },
        "set_wired_limit",
    )?;
    Ok(previous)
}

/// Cap the memory MLX holds for reuse. Returns the previous limit.
///
/// Zero makes every free return to the OS immediately, which trades throughput
/// for a smaller footprint — worth it when something else on the machine needs
/// the room more than this process needs the speed.
pub fn set_cache_limit(bytes: usize) -> Result<usize> {
    init();
    let mut previous = 0usize;
    check(
        unsafe { mlx_set_cache_limit(&mut previous, bytes) },
        "set_cache_limit",
    )?;
    Ok(previous)
}

/// Return MLX's reuse cache to the OS now.
pub fn clear_cache() -> Result<()> {
    init();
    check(unsafe { mlx_clear_cache() }, "clear_cache")
}

// -- quantisation -----------------------------------------------------------

/// MLX's default quantisation mode. `affine` is scale-and-bias per group,
/// which is what `mlx.quantize` uses when you do not ask for anything else.
const QUANT_MODE: &str = "affine";

/// A weight held **quantised at rest**, with the scales and biases that
/// reconstruct it.
///
/// The whole point is that this is never fully materialised. A 12B-parameter
/// transformer is 47.6 GB dense in f32 and does not fit on a 36 GB machine; at
/// 4 bits with group-64 scales it is about 6.7 GB, and the dequantisation
/// happens inside the matmul kernel one tile at a time.
///
/// **`bits` and `group_size` are not free parameters.** They have to match what
/// the weight was quantised with, and nothing in the tensors themselves records
/// it — a mismatched pair reconstructs plausible numbers from the right bits in
/// the wrong places.
#[derive(Debug)]
pub struct QuantizedArray {
    /// The packed weights, `uint32` with `32 / bits` values per element.
    pub weight: Array,
    pub scales: Array,
    pub biases: Array,
    pub group_size: usize,
    pub bits: usize,
}

impl QuantizedArray {
    /// Quantise a dense array. `[out, in]` in, packed `[out, in * bits / 32]`
    /// out.
    ///
    /// `group_size` must divide the *input* width, and 64 is MLX's default.
    pub fn quantize(w: &Array, group_size: usize, bits: usize, stream: &Stream) -> Result<Self> {
        let mode = CString::new(QUANT_MODE).expect("literal");
        let null = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        let mut out = mlx_vector_array {
            ctx: std::ptr::null_mut(),
        };
        check(
            unsafe {
                mlx_quantize(
                    &mut out,
                    w.raw,
                    mlx_optional_int::some(group_size as i32),
                    mlx_optional_int::some(bits as i32),
                    mode.as_ptr(),
                    null,
                    stream.0,
                )
            },
            "quantize",
        )?;

        // Three arrays out: weight, scales, biases.
        let n = unsafe { mlx_vector_array_size(out) };
        if n != 3 {
            unsafe { mlx_vector_array_free(out) };
            return Err(Error::Msg(format!(
                "mlx: quantize returned {n} arrays, expected weight, scales and biases"
            )));
        }
        let mut parts = Vec::with_capacity(3);
        for i in 0..3 {
            let mut a = null;
            let rc = unsafe { mlx_vector_array_get(&mut a, out, i) };
            if rc != 0 {
                unsafe { mlx_vector_array_free(out) };
                return Err(Error::Msg(format!("mlx: quantize output {i}")));
            }
            parts.push(Array { raw: a });
        }
        unsafe { mlx_vector_array_free(out) };
        let mut it = parts.into_iter();
        Ok(Self {
            weight: it.next().expect("three"),
            scales: it.next().expect("three"),
            biases: it.next().expect("three"),
            group_size,
            bits,
        })
    }

    /// Reconstruct the dense array.
    ///
    /// **For testing and for layers that have no quantised kernel**, not for
    /// the hot path — materialising every weight is exactly what this type
    /// exists to avoid.
    pub fn dequantize(&self, stream: &Stream) -> Result<Array> {
        let mode = CString::new(QUANT_MODE).expect("literal");
        let null = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        let mut out = null;
        check(
            unsafe {
                mlx_dequantize(
                    &mut out,
                    self.weight.raw,
                    self.scales.raw,
                    self.biases.raw,
                    mlx_optional_int::some(self.group_size as i32),
                    mlx_optional_int::some(self.bits as i32),
                    mode.as_ptr(),
                    null,
                    mlx_optional_dtype {
                        value: 0,
                        has_value: false,
                    },
                    stream.0,
                )
            },
            "dequantize",
        )?;
        Ok(Array { raw: out })
    }

    /// `x @ weight.T`, dequantising inside the kernel.
    ///
    /// `transpose` is true because a `diffusers` linear stores `[out, in]` and
    /// this multiplies by its transpose — the same convention the dense
    /// `linear` here uses. Passing false runs and contracts the wrong axis.
    pub fn matmul(&self, x: &Array, stream: &Stream) -> Result<Array> {
        let mode = CString::new(QUANT_MODE).expect("literal");
        let mut out = mlx_array {
            ctx: std::ptr::null_mut(),
        };
        check(
            unsafe {
                mlx_quantized_matmul(
                    &mut out,
                    x.raw,
                    self.weight.raw,
                    self.scales.raw,
                    self.biases.raw,
                    true,
                    mlx_optional_int::some(self.group_size as i32),
                    mlx_optional_int::some(self.bits as i32),
                    mode.as_ptr(),
                    stream.0,
                )
            },
            "quantized_matmul",
        )?;
        Ok(Array { raw: out })
    }

    /// Bytes this weight occupies, for a residency report.
    ///
    /// The packed weights plus both scale tensors — the honest figure, not just
    /// the bits, because at group 64 the scales are a real 6 % on top.
    pub fn resident_bytes(&self) -> usize {
        let packed = self.weight.elem_count() * 4;
        let scales = (self.scales.elem_count() + self.biases.elem_count()) * 4;
        packed + scales
    }
}
