use std::sync::{atomic::AtomicPtr, Arc, Mutex};

use lazy_static::lazy_static;

use onnxruntime_sys as sys;

lazy_static! {
    static ref G_ORT: Arc<Mutex<AtomicPtr<sys::OrtApi>>> =
        Arc::new(Mutex::new(AtomicPtr::new(unsafe {
            sys::OrtGetApiBase().as_ref().unwrap().GetApi.unwrap()(sys::ORT_API_VERSION)
        } as *mut sys::OrtApi)));
}

fn g_ort() -> *mut sys::OrtApi {
    *G_ORT.lock().unwrap().get_mut()
}

fn char_p_to_string(raw: *const i8) -> Result<String, std::ffi::IntoStringError> {
    let c_string = unsafe { std::ffi::CString::from_raw(raw as *mut i8) };
    c_string.into_string()
}

mod env;
mod error;
mod session;

// Re-export
pub use env::EnvBuilder;

#[repr(u32)]
pub enum LoggingLevel {
    Verbose = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE,
    Info = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO,
    Warning = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
    Error = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_ERROR,
    Fatal = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_FATAL,
}

#[repr(u32)]
pub enum GraphOptimizationLevel {
    DisableAll = sys::GraphOptimizationLevel_ORT_DISABLE_ALL,
    Basic = sys::GraphOptimizationLevel_ORT_ENABLE_BASIC,
    Extended = sys::GraphOptimizationLevel_ORT_ENABLE_EXTENDED,
    All = sys::GraphOptimizationLevel_ORT_ENABLE_ALL,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_base_non_null() {
        assert_ne!(*G_ORT.lock().unwrap().get_mut(), std::ptr::null_mut());
    }
}
