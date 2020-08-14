#![warn(missing_docs)]

//! ONNX Runtime
//!
//! This crate is a (safe) wrapper around Microsoft's [ONNX Runtime](https://github.com/microsoft/onnxruntime/)
//! through its C API.
//!
//! From its [GitHub page](https://github.com/microsoft/onnxruntime/):
//!
//! > ONNX Runtime is a cross-platform, high performance ML inferencing and training accelerator.
//!
//! The (highly) unsafe [C API](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_c_api.h)
//! is wrapped using bindgen as [`onnxruntime-sys`](https://crates.io/crates/onnxruntime-sys).
//!
//! The unsafe bindings are wrapped in this crate to expose a safe API.
//!
//! For now, efforts are concentrated on the inference API. Training is _not_ supported.
//!
//! # Example
//!
//! The C++ example that uses the C API
//! ([`C_Api_Sample.cpp`](https://github.com/microsoft/onnxruntime/blob/v1.3.1/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp))
//! was ported to
//! [`onnxruntime`](https://github.com/nbigaouette/onnxruntime-rs/blob/master/onnxruntime/examples/sample.rs).
//!
//! First, an environment must be created using and [`EnvBuilder`](environment/struct.EnvBuilder.html):
//!
//! ```no_run
//! # use std::error::Error;
//! # use onnxruntime::{environment::Environment, LoggingLevel};
//! # fn main() -> Result<(), Box<dyn Error>> {
//! let environment = Environment::builder()
//!     .with_name("test")
//!     .with_log_level(LoggingLevel::Verbose)
//!     .build()?;
//! # Ok(())
//! # }
//! ```
//!
//! Then a [`Session`](session/struct.Session.html) is created from the environment, some options and an ONNX archive:
//!
//! ```no_run
//! # use std::error::Error;
//! # use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel};
//! # fn main() -> Result<(), Box<dyn Error>> {
//! # let environment = Environment::builder()
//! #     .with_name("test")
//! #     .with_log_level(LoggingLevel::Verbose)
//! #     .build()?;
//! let mut session = environment
//!     .new_session_builder()?
//!     .with_optimization_level(GraphOptimizationLevel::Basic)?
//!     .with_number_threads(1)?
//!     .load_model_from_file("squeezenet.onnx")?;
//! # Ok(())
//! # }
//! ```
//!
//! Instead of loading a model from file using [`load_model_from_file()`](session/struct.SessionBuilder.html#method.load_model_from_file),
//! a model can be fetched directly from the [ONNX Model Zoo](https://github.com/onnx/models) using
//! [`with_downloaded_model()`](session/struct.SessionBuilder.html#method.with_downloaded_model) method
//! (requires the `model-fetching` feature).
//!
//! ```no_run
//! # use std::error::Error;
//! # use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel};
//! # fn main() -> Result<(), Box<dyn Error>> {
//! # let environment = Environment::builder()
//! #     .with_name("test")
//! #     .with_log_level(LoggingLevel::Verbose)
//! #     .build()?;
//! let mut session = env
//!     .new_session_builder()?
//!     .with_optimization_level(GraphOptimizationLevel::Basic)?
//!     .with_number_threads(1)?
//!     .with_downloaded_model(ImageClassificationModel::SqueezeNet)?;
//! # Ok(())
//! # }
//! ```
//!
//! See [`AvailableOnnxModel`](download/enum.AvailableOnnxModel.html) for the different models available
//! to download.
//!
//!
//! Inference will be run on data passed as an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html).
//!
//! ```no_run
//! # use std::error::Error;
//! # use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel, tensor::TensorFromOrt};
//! # fn main() -> Result<(), Box<dyn Error>> {
//! # let environment = Environment::builder()
//! #     .with_name("test")
//! #     .with_log_level(LoggingLevel::Verbose)
//! #     .build()?;
//! # let mut session = environment
//! #     .new_session_builder()?
//! #     .with_optimization_level(GraphOptimizationLevel::Basic)?
//! #     .with_number_threads(1)?
//! #     .load_model_from_file("squeezenet.onnx")?;
//! let array = ndarray::Array::linspace(0.0_f32, 1.0, 100);
//! // Multiple inputs and outputs are possible
//! let input_tensor = vec![array];
//! let outputs: Vec<TensorFromOrt<f32,_>> = session.run(input_tensor)?;
//! # Ok(())
//! # }
//! ```
//!
//! The outputs are of type [`TensorFromOrt`](tensor/struct.TensorFromOrt.html)s inside a vector,
//! with the same length as the inputs.
//!
//! See the [`sample.rs`](https://github.com/nbigaouette/onnxruntime-rs/blob/master/onnxruntime/examples/sample.rs)
//! example for more details.

use std::sync::{atomic::AtomicPtr, Arc, Mutex};

use lazy_static::lazy_static;

use onnxruntime_sys as sys;

#[cfg(feature = "model-fetching")]
pub mod download;
pub mod environment;
pub mod error;
mod memory;
pub mod session;
pub mod tensor;

// Re-export
pub use error::{OrtApiError, OrtError, Result};

lazy_static! {
    static ref G_ORT: Arc<Mutex<AtomicPtr<sys::OrtApi>>> =
        Arc::new(Mutex::new(AtomicPtr::new(unsafe {
            sys::OrtGetApiBase().as_ref().unwrap().GetApi.unwrap()(sys::ORT_API_VERSION)
        } as *mut sys::OrtApi)));
}

fn g_ort() -> *mut sys::OrtApi {
    *G_ORT.lock().unwrap().get_mut()
}

fn char_p_to_string(raw: *const i8) -> Result<String> {
    let c_string = unsafe { std::ffi::CString::from_raw(raw as *mut i8) };

    match c_string.into_string() {
        Ok(string) => Ok(string),
        Err(e) => Err(OrtApiError::IntoStringError(e)),
    }
    .map_err(OrtError::StringConversion)
}

/// Logging level of the ONNX Runtime C API
#[derive(Debug)]
#[repr(u32)]
pub enum LoggingLevel {
    /// Verbose log level
    Verbose = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE,
    /// Info log level
    Info = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO,
    /// Warning log level
    Warning = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
    /// Error log level
    Error = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_ERROR,
    /// Fatal log level
    Fatal = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_FATAL,
}

/// Optimization level performed by ONNX Runtime of the loaded graph
///
/// See the [official documentation](https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Graph_Optimizations.md)
/// for more information on the different optimization levels.
#[derive(Debug)]
#[repr(u32)]
pub enum GraphOptimizationLevel {
    /// Disable optimization
    DisableAll = sys::GraphOptimizationLevel_ORT_DISABLE_ALL,
    /// Basic optimization
    Basic = sys::GraphOptimizationLevel_ORT_ENABLE_BASIC,
    /// Extended optimization
    Extended = sys::GraphOptimizationLevel_ORT_ENABLE_EXTENDED,
    /// Add optimization
    All = sys::GraphOptimizationLevel_ORT_ENABLE_ALL,
}

// FIXME: Use https://docs.rs/bindgen/0.54.1/bindgen/struct.Builder.html#method.rustified_enum
// FIXME: Add tests to cover the commented out types
/// Enum mapping ONNX Runtime's supported tensor types
#[derive(Debug)]
#[repr(u32)]
pub enum TensorElementDataType {
    /// 32-bit floating point, equivalent to Rust's `f32`
    Float = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    /// Unsigned 8-bit int, equivalent to Rust's `u8`
    Uint8 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    /// Signed 8-bit int, equivalent to Rust's `i8`
    Int8 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
    /// Unsigned 16-bit int, equivalent to Rust's `u16`
    Uint16 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
    /// Signed 16-bit int, equivalent to Rust's `i16`
    Int16 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
    /// Signed 32-bit int, equivalent to Rust's `i32`
    Int32 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    /// Signed 64-bit int, equivalent to Rust's `i64`
    Int64 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    // /// String, equivalent to Rust's `String`
    // String = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
    // /// Boolean, equivalent to Rust's `bool`
    // Bool = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
    // /// 16-bit floating point, equivalent to Rust's `f16`
    // Float16 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
    /// 64-bit floating point, equivalent to Rust's `f64`
    Double = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
    /// Unsigned 32-bit int, equivalent to Rust's `u32`
    Uint32 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
    /// Unsigned 64-bit int, equivalent to Rust's `u64`
    Uint64 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
    // /// Complex 64-bit floating point, equivalent to Rust's `???`
    // Complex64 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
    // /// Complex 128-bit floating point, equivalent to Rust's `???`
    // Complex128 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
    // /// Brain 16-bit floating point
    // Bfloat16 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
}

/// Trait used to map Rust types (for example `f32`) to ONNX types (for example `Float`)
pub trait TypeToTensorElementDataType {
    /// Return the ONNX type for a Rust type
    fn tensor_element_data_type() -> TensorElementDataType;
}

macro_rules! impl_type_trait {
    ($type_:ty, $variant:ident) => {
        impl TypeToTensorElementDataType for $type_ {
            fn tensor_element_data_type() -> TensorElementDataType {
                TensorElementDataType::$variant
            }
        }
    };
}

impl_type_trait!(f32, Float);
impl_type_trait!(u8, Uint8);
impl_type_trait!(i8, Int8);
impl_type_trait!(u16, Uint16);
impl_type_trait!(i16, Int16);
impl_type_trait!(i32, Int32);
impl_type_trait!(i64, Int64);
// impl_type_trait!(String, String);
// impl_type_trait!(bool, Bool);
// impl_type_trait!(f16, Float16);
impl_type_trait!(f64, Double);
impl_type_trait!(u32, Uint32);
impl_type_trait!(u64, Uint64);
// impl_type_trait!(, Complex64);
// impl_type_trait!(, Complex128);
// impl_type_trait!(, Bfloat16);

/// Allocator type
#[derive(Debug, Clone)]
#[repr(i32)]
pub enum AllocatorType {
    // Invalid = sys::OrtAllocatorType_Invalid,
    /// Device allocator
    Device = sys::OrtAllocatorType_OrtDeviceAllocator,
    /// Arena allocator
    Arena = sys::OrtAllocatorType_OrtArenaAllocator,
}

/// Memory type
///
/// Only support ONNX's default type for now.
#[derive(Debug, Clone)]
#[repr(i32)]
pub enum MemType {
    // FIXME: C API's `OrtMemType_OrtMemTypeCPU` defines it equal to `OrtMemType_OrtMemTypeCPUOutput`. How to handle this??
    // CPUInput = sys::OrtMemType_OrtMemTypeCPUInput,
    // CPUOutput = sys::OrtMemType_OrtMemTypeCPUOutput,
    // CPU = sys::OrtMemType_OrtMemTypeCPU,
    /// Default memory type
    Default = sys::OrtMemType_OrtMemTypeDefault,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_base_non_null() {
        assert_ne!(*G_ORT.lock().unwrap().get_mut(), std::ptr::null_mut());
    }
}
