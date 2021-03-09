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
//!     .with_model_from_file("squeezenet.onnx")?;
//! # Ok(())
//! # }
//! ```
//!
#![cfg_attr(
    feature = "model-fetching",
    doc = r##"
Instead of loading a model from file using [`with_model_from_file()`](session/struct.SessionBuilder.html#method.with_model_from_file),
a model can be fetched directly from the [ONNX Model Zoo](https://github.com/onnx/models) using
[`with_model_downloaded()`](session/struct.SessionBuilder.html#method.with_model_downloaded) method
(requires the `model-fetching` feature).

```no_run
# use std::error::Error;
# use onnxruntime::{environment::Environment, download::vision::ImageClassification, LoggingLevel, GraphOptimizationLevel};
# fn main() -> Result<(), Box<dyn Error>> {
# let environment = Environment::builder()
#     .with_name("test")
#     .with_log_level(LoggingLevel::Verbose)
#     .build()?;
let mut session = environment
    .new_session_builder()?
    .with_optimization_level(GraphOptimizationLevel::Basic)?
    .with_number_threads(1)?
    .with_model_downloaded(ImageClassification::SqueezeNet)?;
# Ok(())
# }
```

See [`AvailableOnnxModel`](download/enum.AvailableOnnxModel.html) for the different models available
to download.
"##
)]
//!
//! Inference will be run on data passed as an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html).
//!
//! ```no_run
//! # use std::error::Error;
//! # use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel, tensor::OrtOwnedTensor};
//! # fn main() -> Result<(), Box<dyn Error>> {
//! # let environment = Environment::builder()
//! #     .with_name("test")
//! #     .with_log_level(LoggingLevel::Verbose)
//! #     .build()?;
//! # let mut session = environment
//! #     .new_session_builder()?
//! #     .with_optimization_level(GraphOptimizationLevel::Basic)?
//! #     .with_number_threads(1)?
//! #     .with_model_from_file("squeezenet.onnx")?;
//! let array = ndarray::Array::linspace(0.0_f32, 1.0, 100);
//! // Multiple inputs and outputs are possible
//! let input_tensor = vec![array];
//! let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor)?
//!     .into_iter()
//!     .map(|dyn_tensor| dyn_tensor.try_extract())
//!     .collect::<Result<_, _>>()?;
//! # Ok(())
//! # }
//! ```
//!
//! The outputs are of type [`OrtOwnedTensor`](tensor/struct.OrtOwnedTensor.html)s inside a vector,
//! with the same length as the inputs.
//!
//! See the [`sample.rs`](https://github.com/nbigaouette/onnxruntime-rs/blob/master/onnxruntime/examples/sample.rs)
//! example for more details.

use std::{
    ffi, ptr,
    sync::{atomic::AtomicPtr, Arc, Mutex},
};

use lazy_static::lazy_static;

use onnxruntime_sys as sys;

pub mod download;
pub mod environment;
pub mod error;
mod memory;
pub mod session;
pub mod tensor;

// Re-export
pub use error::{OrtApiError, OrtError, Result};
use sys::OnnxEnumInt;

// Re-export ndarray as it's part of the public API anyway
pub use ndarray;

lazy_static! {
    // static ref G_ORT: Arc<Mutex<AtomicPtr<sys::OrtApi>>> =
    //     Arc::new(Mutex::new(AtomicPtr::new(unsafe {
    //         sys::OrtGetApiBase().as_ref().unwrap().GetApi.unwrap()(sys::ORT_API_VERSION)
    //     } as *mut sys::OrtApi)));
    static ref G_ORT_API: Arc<Mutex<AtomicPtr<sys::OrtApi>>> = {
        let base: *const sys::OrtApiBase = unsafe { sys::OrtGetApiBase() };
        assert_ne!(base, ptr::null());
        let get_api: unsafe extern "C" fn(u32) -> *const onnxruntime_sys::OrtApi =
            unsafe { (*base).GetApi.unwrap() };
        let api: *const sys::OrtApi = unsafe { get_api(sys::ORT_API_VERSION) };
        Arc::new(Mutex::new(AtomicPtr::new(api as *mut sys::OrtApi)))
    };
}

fn g_ort() -> sys::OrtApi {
    let mut api_ref = G_ORT_API
        .lock()
        .expect("Failed to acquire lock: another thread panicked?");
    let api_ref_mut: &mut *mut sys::OrtApi = api_ref.get_mut();
    let api_ptr_mut: *mut sys::OrtApi = *api_ref_mut;

    assert_ne!(api_ptr_mut, ptr::null_mut());

    unsafe { *api_ptr_mut }
}

fn char_p_to_string(raw: *const i8) -> Result<String> {
    let c_string = unsafe { ffi::CStr::from_ptr(raw as *mut i8).to_owned() };

    match c_string.into_string() {
        Ok(string) => Ok(string),
        Err(e) => Err(OrtApiError::IntoStringError(e)),
    }
    .map_err(OrtError::StringConversion)
}

mod onnxruntime {
    //! Module containing a custom logger, used to catch the runtime's own logging and send it
    //! to Rust's tracing logging instead.

    use std::{ffi, ffi::CStr, ptr};
    use tracing::{debug, error, info, span, trace, warn, Level};

    use onnxruntime_sys as sys;

    /// Runtime's logging sends the code location where the log happened, will be parsed to this struct.
    #[derive(Debug)]
    struct CodeLocation<'a> {
        file: &'a str,
        line_number: &'a str,
        function: &'a str,
    }

    impl<'a> From<&'a str> for CodeLocation<'a> {
        fn from(code_location: &'a str) -> Self {
            let mut splitter = code_location.split(' ');
            let file_and_line_number = splitter.next().unwrap_or("<unknown file:line>");
            let function = splitter.next().unwrap_or("<unknown module>");
            let mut file_and_line_number_splitter = file_and_line_number.split(':');
            let file = file_and_line_number_splitter
                .next()
                .unwrap_or("<unknown file>");
            let line_number = file_and_line_number_splitter
                .next()
                .unwrap_or("<unknown line number>");

            CodeLocation {
                file,
                line_number,
                function,
            }
        }
    }

    /// Callback from C that will handle the logging, forwarding the runtime's logs to the tracing crate.
    pub(crate) extern "C" fn custom_logger(
        _params: *mut ffi::c_void,
        severity: sys::OrtLoggingLevel,
        category: *const i8,
        logid: *const i8,
        code_location: *const i8,
        message: *const i8,
    ) {
        let log_level = match severity {
            sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE => Level::TRACE,
            sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO => Level::DEBUG,
            sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING => Level::INFO,
            sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR => Level::WARN,
            sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL => Level::ERROR,
        };

        assert_ne!(category, ptr::null());
        let category = unsafe { CStr::from_ptr(category) };
        assert_ne!(code_location, ptr::null());
        let code_location = unsafe { CStr::from_ptr(code_location) }
            .to_str()
            .unwrap_or("unknown");
        assert_ne!(message, ptr::null());
        let message = unsafe { CStr::from_ptr(message) };

        assert_ne!(logid, ptr::null());
        let logid = unsafe { CStr::from_ptr(logid) };

        // Parse the code location
        let code_location: CodeLocation = code_location.into();

        let span = span!(
            Level::TRACE,
            "onnxruntime",
            category = category.to_str().unwrap_or("<unknown>"),
            file = code_location.file,
            line_number = code_location.line_number,
            function = code_location.function,
            logid = logid.to_str().unwrap_or("<unknown>"),
        );
        let _enter = span.enter();

        match log_level {
            Level::TRACE => trace!("{:?}", message),
            Level::DEBUG => debug!("{:?}", message),
            Level::INFO => info!("{:?}", message),
            Level::WARN => warn!("{:?}", message),
            Level::ERROR => error!("{:?}", message),
        }
    }
}

/// Logging level of the ONNX Runtime C API
#[derive(Debug)]
#[cfg_attr(not(windows), repr(u32))]
#[cfg_attr(windows, repr(i32))]
pub enum LoggingLevel {
    /// Verbose log level
    Verbose = sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE as OnnxEnumInt,
    /// Info log level
    Info = sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO as OnnxEnumInt,
    /// Warning log level
    Warning = sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING as OnnxEnumInt,
    /// Error log level
    Error = sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR as OnnxEnumInt,
    /// Fatal log level
    Fatal = sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL as OnnxEnumInt,
}

impl Into<sys::OrtLoggingLevel> for LoggingLevel {
    fn into(self) -> sys::OrtLoggingLevel {
        match self {
            LoggingLevel::Verbose => sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
            LoggingLevel::Info => sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
            LoggingLevel::Warning => sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            LoggingLevel::Error => sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
            LoggingLevel::Fatal => sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL,
        }
    }
}

/// Optimization level performed by ONNX Runtime of the loaded graph
///
/// See the [official documentation](https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Graph_Optimizations.md)
/// for more information on the different optimization levels.
#[derive(Debug)]
#[cfg_attr(not(windows), repr(u32))]
#[cfg_attr(windows, repr(i32))]
pub enum GraphOptimizationLevel {
    /// Disable optimization
    DisableAll = sys::GraphOptimizationLevel::ORT_DISABLE_ALL as OnnxEnumInt,
    /// Basic optimization
    Basic = sys::GraphOptimizationLevel::ORT_ENABLE_BASIC as OnnxEnumInt,
    /// Extended optimization
    Extended = sys::GraphOptimizationLevel::ORT_ENABLE_EXTENDED as OnnxEnumInt,
    /// Add optimization
    All = sys::GraphOptimizationLevel::ORT_ENABLE_ALL as OnnxEnumInt,
}

impl Into<sys::GraphOptimizationLevel> for GraphOptimizationLevel {
    fn into(self) -> sys::GraphOptimizationLevel {
        use GraphOptimizationLevel::*;
        match self {
            DisableAll => sys::GraphOptimizationLevel::ORT_DISABLE_ALL,
            Basic => sys::GraphOptimizationLevel::ORT_ENABLE_BASIC,
            Extended => sys::GraphOptimizationLevel::ORT_ENABLE_EXTENDED,
            All => sys::GraphOptimizationLevel::ORT_ENABLE_ALL,
        }
    }
}

/// Allocator type
#[derive(Debug, Clone)]
#[repr(i32)]
pub enum AllocatorType {
    // Invalid = sys::OrtAllocatorType::Invalid as i32,
    /// Device allocator
    Device = sys::OrtAllocatorType::OrtDeviceAllocator as i32,
    /// Arena allocator
    Arena = sys::OrtAllocatorType::OrtArenaAllocator as i32,
}

impl Into<sys::OrtAllocatorType> for AllocatorType {
    fn into(self) -> sys::OrtAllocatorType {
        use AllocatorType::*;
        match self {
            // Invalid => sys::OrtAllocatorType::Invalid,
            Device => sys::OrtAllocatorType::OrtDeviceAllocator,
            Arena => sys::OrtAllocatorType::OrtArenaAllocator,
        }
    }
}

/// Memory type
///
/// Only support ONNX's default type for now.
#[derive(Debug, Clone)]
#[repr(i32)]
pub enum MemType {
    // FIXME: C API's `OrtMemType_OrtMemTypeCPU` defines it equal to `OrtMemType_OrtMemTypeCPUOutput`. How to handle this??
    // CPUInput = sys::OrtMemType::OrtMemTypeCPUInput as i32,
    // CPUOutput = sys::OrtMemType::OrtMemTypeCPUOutput as i32,
    // CPU = sys::OrtMemType::OrtMemTypeCPU as i32,
    /// Default memory type
    Default = sys::OrtMemType::OrtMemTypeDefault as i32,
}

impl Into<sys::OrtMemType> for MemType {
    fn into(self) -> sys::OrtMemType {
        use MemType::*;
        match self {
            // CPUInput => sys::OrtMemType::OrtMemTypeCPUInput,
            // CPUOutput => sys::OrtMemType::OrtMemTypeCPUOutput,
            // CPU => sys::OrtMemType::OrtMemTypeCPU,
            Default => sys::OrtMemType::OrtMemTypeDefault,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_char_p_to_string() {
        let s = ffi::CString::new("foo").unwrap();
        let ptr = s.as_c_str().as_ptr();
        assert_eq!("foo", char_p_to_string(ptr).unwrap());
    }
}
