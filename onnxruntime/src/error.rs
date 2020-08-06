use std::path::PathBuf;

use thiserror::Error;

use onnxruntime_sys as sys;

use crate::{char_p_to_string, g_ort};

pub type Result<T> = std::result::Result<T, OrtError>;

#[derive(Error, Debug)]
pub enum OrtError {
    #[error("Failed to construct String")]
    StringConversion(OrtApiError),
    #[error("Failed to create environment")]
    Environment(OrtApiError),
    #[error("Failed to create session options")]
    SessionOptions(OrtApiError),
    #[error("Failed to create session")]
    Session(OrtApiError),
    #[error("Failed to get allocator")]
    Allocator(OrtApiError),
    #[error("Failed to get input name")]
    InputName(OrtApiError),
    #[error("File {filename:?} does not exists")]
    FileDoesNotExists { filename: PathBuf },
    #[error("Path {path:?} cannot be converted to UTF-8")]
    NonUtf8Path { path: PathBuf },
    #[error("Failed to build CString when original contains null: {0}")]
    CStringNulError(#[from] std::ffi::NulError),
    // #[error("Failed to acquire lock")]
    // Lock(std::sync::LockResult),
    // #[error("data store disconnected")]
    // Disconnect(#[from] io::Error),
    // #[error("the data for key `{0}` is not available")]
    // Redaction(String),
    // #[error("invalid header (expected {expected:?}, found {found:?})")]
    // InvalidHeader { expected: String, found: String },
    // #[error("unknown data store error")]
    // Unknown,
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
