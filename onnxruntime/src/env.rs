use std::{
    ffi::CString,
    path::PathBuf,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc, Mutex,
    },
};

use lazy_static::lazy_static;

use onnxruntime_sys as sys;

use crate::{
    error::{status_to_result, OrtError, Result},
    g_ort,
    session::{GraphOptimizationLevel, SessionBuilder},
};

lazy_static! {
    static ref G_NAMED_ENV: Arc<Mutex<NamedEnv>> = Arc::new(Mutex::new(NamedEnv {
        env_ptr: EnvPointer(AtomicPtr::new(std::ptr::null_mut())),
        name: CString::new("uninitialized").unwrap(),
    }));
}

// FIXME: Implement Deref
#[derive(Debug)]
pub(crate) struct EnvPointer(pub(crate) AtomicPtr<sys::OrtEnv>);

impl Drop for EnvPointer {
    fn drop(&mut self) {
        println!("Dropping InnerEnv!");
        if *self.0.get_mut() == std::ptr::null_mut() {
            eprintln!("ERROR: InnerEnv pointer already null, cannot double-free!");
        } else {
            unsafe { (*g_ort()).ReleaseEnv.unwrap()(*self.0.get_mut()) };
            *self.0.get_mut() = std::ptr::null_mut();
        }
    }
}

#[derive(Debug)]
pub struct NamedEnv {
    pub(crate) env_ptr: EnvPointer,
    pub(crate) name: CString,
}

#[derive(Debug)]
pub struct Env {
    inner: Arc<Mutex<NamedEnv>>,
}

impl Env {
    pub fn new<S>(name: S) -> Result<Env>
    where
        S: AsRef<str>,
    {
        Env::from_str(name.as_ref())
    }

    pub fn from_str(name: &str) -> Result<Env> {
        let mut g_named_env = G_NAMED_ENV.lock().unwrap();

        let name = CString::new(name)?;

        if *g_named_env.env_ptr.0.get_mut() == std::ptr::null_mut() {
            println!(
                "Uninitialized environment found, initializing it with name {:?}.",
                name
            );
            let mut env_ptr: *mut sys::OrtEnv = std::ptr::null_mut();

            // FIXME: Pass log level to function
            let status = unsafe {
                (*g_ort()).CreateEnv.unwrap()(
                    sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE,
                    name.as_ptr(),
                    &mut env_ptr,
                )
            };

            status_to_result(status).map_err(OrtError::Environment)?;
            assert_eq!(status, std::ptr::null_mut());
            assert_ne!(env_ptr, std::ptr::null_mut());

            // Replace the pointer stored in the lazy_static with the new one
            let old_ptr: *mut sys::OrtEnv = g_named_env.env_ptr.0.swap(env_ptr, Ordering::AcqRel);
            assert_eq!(old_ptr, std::ptr::null_mut());
            // Replace the name stored in the lazy_static with the new one
            g_named_env.name = name.clone();

            let env_ptr = EnvPointer(AtomicPtr::new(env_ptr));
            let named_env = NamedEnv { env_ptr, name };
            let env = Env {
                inner: Arc::new(Mutex::new(named_env)),
            };

            Ok(env)
        } else {
            println!(
                "Initialized environment found ({:?}), not initializing it.",
                g_named_env.name
            );

            Ok(Env {
                inner: G_NAMED_ENV.clone(),
            })
        }
    }

    pub fn load_model<P>(&self, filename: P) -> SessionBuilder
    where
        P: Into<PathBuf>,
    {
        SessionBuilder {
            inner: self.inner.clone(),

            name: "default".into(),
            num_threads: 1,
            options: None,
            opt_level: GraphOptimizationLevel::DisableAll,
            model_filename: filename.into(),
            use_cuda: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_env() {
        let env1 = Env::new("test1").unwrap();

        assert_eq!(
            G_NAMED_ENV.lock().unwrap().name,
            CString::new("test1").unwrap()
        );
        assert_eq!(
            env1.inner.lock().unwrap().name,
            CString::new("test1").unwrap()
        );
        assert_eq!(
            *G_NAMED_ENV.lock().unwrap().env_ptr.0.get_mut() as usize,
            *env1.inner.lock().unwrap().env_ptr.0.get_mut() as usize
        );

        let env2 = Env::new("test2").unwrap();

        assert_eq!(
            G_NAMED_ENV.lock().unwrap().name,
            CString::new("test1").unwrap(),
            "lazy_static must keep its name"
        );
        assert_eq!(
            env2.inner.lock().unwrap().name,
            CString::new("test1").unwrap(),
            "Environment should contain information from first creation"
        );

        let env3 = Env::new("test3").unwrap();

        assert_eq!(
            G_NAMED_ENV.lock().unwrap().name,
            CString::new("test1").unwrap(),
            "lazy_static must keep its name"
        );
        assert_eq!(
            env3.inner.lock().unwrap().name,
            CString::new("test1").unwrap(),
            "Environment should contain information from first creation"
        );
    }
}
