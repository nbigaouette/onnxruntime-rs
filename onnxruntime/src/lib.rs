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
pub use env::Env;
pub use session::GraphOptimizationLevel;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_base_non_null() {
        assert_ne!(*G_ORT.lock().unwrap().get_mut(), std::ptr::null_mut());
    }
}
