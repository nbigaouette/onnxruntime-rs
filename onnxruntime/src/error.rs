use std::path::PathBuf;

use thiserror::Error;

use onnxruntime_sys as sys;

use crate::{char_p_to_string, g_ort};

pub type Result<T> = std::result::Result<T, OrtError>;

#[derive(Error, Debug)]
pub enum OrtError {
    #[error("Failed to construct String")]
    StringConversion(OrtApiError),
    // FIXME: Move these to another enum (they are C API calls errors)
    #[error("Failed to create environment: {0}")]
    Environment(OrtApiError),
    #[error("Failed to create session options: {0}")]
    SessionOptions(OrtApiError),
    #[error("Failed to create session: {0}")]
    Session(OrtApiError),
    #[error("Failed to get allocator: {0}")]
    Allocator(OrtApiError),
    #[error("Failed to get input name: {0}")]
    InputName(OrtApiError),
    #[error("Failed to get input type info: {0}")]
    GetInputTypeInfo(OrtApiError),
    #[error("Failed to cast type info to tensor info: {0}")]
    CastTypeInfoToTensorInfo(OrtApiError),
    #[error("Failed to get tensor element type: {0}")]
    TensorElementType(OrtApiError),
    #[error("Failed to get dimensions count: {0}")]
    GetDimensionsCount(OrtApiError),
    #[error("Failed to get dimensions: {0}")]
    GetDimensions(OrtApiError),
    #[error("Failed to get dimensions: {0}")]
    CreateCpuMemoryInfo(OrtApiError),
    #[error("Failed to create tensor with data: {0}")]
    CreateTensorWithData(OrtApiError),
    #[error("Failed to check if tensor: {0}")]
    IsTensor(OrtApiError),

    #[error("Dimensions do not match")]
    NonMatchingDimensions,
    #[error("File {filename:?} does not exists")]
    FileDoesNotExists { filename: PathBuf },
    #[error("Path {path:?} cannot be converted to UTF-8")]
    NonUtf8Path { path: PathBuf },
    #[error("Failed to build CString when original contains null: {0}")]
    CStringNulError(#[from] std::ffi::NulError),
}

#[derive(Error, Debug)]
pub enum OrtApiError {
    #[error("Error calling ONNX Runtime C function and failed to convert error message to UTF-8")]
    IntoStringError(std::ffi::IntoStringError),
    #[error("Error calling ONNX Runtime C function")]
    Msg(String),
}

pub struct OrtStatusWrapper(*const sys::OrtStatus);

impl From<*const sys::OrtStatus> for OrtStatusWrapper {
    fn from(status: *const sys::OrtStatus) -> Self {
        OrtStatusWrapper(status)
    }
}

impl From<OrtStatusWrapper> for std::result::Result<(), OrtApiError> {
    fn from(status: OrtStatusWrapper) -> Self {
        if status.0 == std::ptr::null() {
            Ok(())
        } else {
            let raw: *const i8 = unsafe { (*g_ort()).GetErrorMessage.unwrap()(status.0) };
            match char_p_to_string(raw) {
                Ok(msg) => Err(OrtApiError::Msg(msg)),
                Err(err) => match err {
                    OrtError::StringConversion(e) => match e {
                        OrtApiError::IntoStringError(e) => Err(OrtApiError::IntoStringError(e)),
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                },
            }
        }
    }
}

pub(crate) fn status_to_result(
    status: *const sys::OrtStatus,
) -> std::result::Result<(), OrtApiError> {
    let status_wrapper: OrtStatusWrapper = status.into();
    status_wrapper.into()
}
