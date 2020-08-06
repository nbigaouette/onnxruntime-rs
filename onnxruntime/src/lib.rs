use std::sync::{atomic::AtomicPtr, Arc, Mutex};

use lazy_static::lazy_static;

use onnxruntime_sys as sys;

mod env;
mod error;
mod session;

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

// Re-export
pub use env::EnvBuilder;

#[derive(Debug)]
#[repr(u32)]
pub enum LoggingLevel {
    Verbose = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE,
    Info = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO,
    Warning = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
    Error = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_ERROR,
    Fatal = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_FATAL,
}

#[derive(Debug)]
#[repr(u32)]
pub enum GraphOptimizationLevel {
    DisableAll = sys::GraphOptimizationLevel_ORT_DISABLE_ALL,
    Basic = sys::GraphOptimizationLevel_ORT_ENABLE_BASIC,
    Extended = sys::GraphOptimizationLevel_ORT_ENABLE_EXTENDED,
    All = sys::GraphOptimizationLevel_ORT_ENABLE_ALL,
}

// FIXME: Use https://docs.rs/bindgen/0.54.1/bindgen/struct.Builder.html#method.rustified_enum
#[derive(Debug)]
#[repr(u32)]
pub enum TensorElementDataType {
    Undefined = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
    Float = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    Uint8 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    Int8 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
    Uint16 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
    Int16 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
    Int32 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    Int64 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    String = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
    Bool = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
    Float16 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
    Double = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
    Uint32 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
    Uint64 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
    Complex64 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
    Complex128 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
    Bfloat16 = sys::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
}

#[derive(Debug)]
#[repr(i32)]
pub enum AllocatorType {
    Invalid = sys::OrtAllocatorType_Invalid,
    Device = sys::OrtAllocatorType_OrtDeviceAllocator,
    Arena = sys::OrtAllocatorType_OrtArenaAllocator,
}

#[derive(Debug)]
#[repr(i32)]
pub enum MemType {
    // FIXME: C API's `OrtMemType_OrtMemTypeCPU` defines it equal to `OrtMemType_OrtMemTypeCPUOutput`. How to handle this??
    // CPUInput = sys::OrtMemType_OrtMemTypeCPUInput,
    // CPUOutput = sys::OrtMemType_OrtMemTypeCPUOutput,
    // CPU = sys::OrtMemType_OrtMemTypeCPU,
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
