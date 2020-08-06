use std::{
    ffi::CString,
    path::PathBuf,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc, Mutex,
    },
};

use onnxruntime_sys as sys;

use crate::{
    env::{Env, NamedEnv},
    error::{status_to_result, OrtError, Result},
    g_ort,
};

// FIXME: Create a high-level wrapper
pub struct SessionOptions {
    ptr: *mut sys::OrtSessionOptions,
}

pub struct Session {
    ptr: *mut sys::OrtSession,
}

#[repr(u32)]
pub enum GraphOptimizationLevel {
    DisableAll = sys::GraphOptimizationLevel_ORT_DISABLE_ALL,
    Basic = sys::GraphOptimizationLevel_ORT_ENABLE_BASIC,
    Extended = sys::GraphOptimizationLevel_ORT_ENABLE_EXTENDED,
    All = sys::GraphOptimizationLevel_ORT_ENABLE_ALL,
}

pub struct SessionBuilder {
    pub(crate) inner: Arc<Mutex<NamedEnv>>,

    pub(crate) name: String,
    pub(crate) options: Option<SessionOptions>,
    pub(crate) opt_level: GraphOptimizationLevel,
    pub(crate) num_threads: i16,
    pub(crate) model_filename: PathBuf,
    pub(crate) use_cuda: bool,
}

impl SessionBuilder {
    pub fn with_options(mut self, options: SessionOptions) -> SessionBuilder {
        self.options = Some(options);
        self
    }

    pub fn with_cuda(mut self, use_cuda: bool) -> SessionBuilder {
        unimplemented!()
        // self.use_cuda = use_cuda;
        // self
    }

    pub fn with_optimization_level(mut self, opt_level: GraphOptimizationLevel) -> SessionBuilder {
        self.opt_level = opt_level;
        self
    }

    pub fn with_number_threads(mut self, num_threads: i16) -> SessionBuilder {
        self.num_threads = num_threads;
        self
    }

    pub fn build(self) -> Result<Session> {
        let mut session_options_ptr: *mut sys::OrtSessionOptions = std::ptr::null_mut();
        let status = unsafe { (*g_ort()).CreateSessionOptions.unwrap()(&mut session_options_ptr) };
        status_to_result(status).map_err(OrtError::SessionOptions)?;
        assert_eq!(status, std::ptr::null_mut());
        assert_ne!(session_options_ptr, std::ptr::null_mut());

        match self.options {
            Some(_options) => unimplemented!(),
            None => {}
        }

        // We use a u16 in the builder to cover the 16-bits positive values of a i32.
        let num_threads = self.num_threads as i32;
        unsafe { (*g_ort()).SetIntraOpNumThreads.unwrap()(session_options_ptr, num_threads) };

        // Sets graph optimization level
        let opt_level = self.opt_level as u32;
        unsafe {
            (*g_ort()).SetSessionGraphOptimizationLevel.unwrap()(session_options_ptr, opt_level)
        };

        let env_ptr: *const sys::OrtEnv = *self.inner.lock().unwrap().env_ptr.0.get_mut();
        let mut session_ptr: *mut sys::OrtSession = std::ptr::null_mut();

        if !self.model_filename.exists() {
            return Err(OrtError::FileDoesNotExists {
                filename: self.model_filename.clone(),
            });
        }
        let model_filename = self.model_filename.clone();
        let model_path: CString =
            CString::new(
                self.model_filename
                    .to_str()
                    .ok_or_else(|| OrtError::NonUtf8Path {
                        path: model_filename,
                    })?,
            )?;

        let status = unsafe {
            (*g_ort()).CreateSession.unwrap()(
                env_ptr,
                model_path.as_ptr(),
                session_options_ptr,
                &mut session_ptr,
            )
        };
        status_to_result(status).map_err(OrtError::Session)?;
        assert_eq!(status, std::ptr::null_mut());
        assert_ne!(session_ptr, std::ptr::null_mut());

        Ok(Session { ptr: session_ptr })
    }
}
