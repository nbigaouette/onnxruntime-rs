//! Module containing environment types
//!
//! An [`Env`](session/struct.Env.html) is the main entry point of the ONNX Runtime.
//!
//! Only one ONNX environment can be created per process. The `onnxruntime` crate
//! uses a singleton (through `lazy_static!()`) to enforce this.
//!
//! Once an environment is created, a [`Session`](../session/struct.Session.html)
//! can be obtained from it.
//!
//! **NOTE**: While the [`Env`](env/struct.Env.html) constructor takes a `name` parameter
//! to name the environment, only the first name will be considered if many environments
//! are created.
//!
//! # Example
//!
//! ```no_run
//! # use std::error::Error;
//! # use onnxruntime::{env::Env, LoggingLevel};
//! # fn main() -> Result<(), Box<dyn Error>> {
//! let env = Env::builder()
//!     .with_name("test")
//!     .with_log_level(LoggingLevel::Verbose)
//!     .build()?;
//! # Ok(())
//! # }
//! ```

use std::{
    ffi::CString,
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
    session::SessionBuilder,
    LoggingLevel,
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
        println!("Dropping the environment.");
        if self.0.get_mut().is_null() {
            eprintln!("ERROR: InnerEnv pointer already null, cannot double-free!");
        } else {
            unsafe { (*g_ort()).ReleaseEnv.unwrap()(*self.0.get_mut()) };
            *self.0.get_mut() = std::ptr::null_mut();
        }
    }
}

#[derive(Debug)]
pub(crate) struct NamedEnv {
    pub(crate) env_ptr: EnvPointer,
    pub(crate) name: CString,
}

/// Struct used to build an environment [`Env`](env/struct.Env.html)
///
/// This is the crate's main entry point. An environment _must_ be created
/// as the first step. An [`Env`](env/struct.Env.html) can only be built
/// using `EnvBuilder` to configure it.
///
/// **NOTE**: If the same configuration method (for example [`with_name()`](struct.EnvBuilder.html#method.with_name))
/// is called multiple times, the last value will have precedence.
pub struct EnvBuilder {
    name: String,
    log_level: LoggingLevel,
}

impl EnvBuilder {
    /// Configure the environment with a given name
    ///
    /// **NOTE**: Since ONNX can only define one environment per process,
    /// creating multiple environments using multiple `EnvBuilder` will
    /// end up re-using the same environment internally; a new one will _not_
    /// be created. New parameters will be ignored.
    pub fn with_name<S>(mut self, name: S) -> EnvBuilder
    where
        S: Into<String>,
    {
        self.name = name.into();
        self
    }

    /// Configure the environment with a given log level
    ///
    /// **NOTE**: Since ONNX can only define one environment per process,
    /// creating multiple environments using multiple `EnvBuilder` will
    /// end up re-using the same environment internally; a new one will _not_
    /// be created. New parameters will be ignored.
    pub fn with_log_level(mut self, log_level: LoggingLevel) -> EnvBuilder {
        self.log_level = log_level;
        self
    }

    /// Commit the configuration to a new [`Env`](env/struct.Env.html)
    pub fn build(self) -> Result<Env> {
        let mut g_named_env = G_NAMED_ENV.lock().unwrap();

        let name = CString::new(self.name)?;

        if g_named_env.env_ptr.0.get_mut().is_null() {
            println!(
                "Uninitialized environment found, initializing it with name {:?}.",
                name
            );
            let mut env_ptr: *mut sys::OrtEnv = std::ptr::null_mut();

            // FIXME: Pass log level to function
            let status = unsafe {
                (*g_ort()).CreateEnv.unwrap()(self.log_level as u32, name.as_ptr(), &mut env_ptr)
            };

            status_to_result(status).map_err(OrtError::Environment)?;
            assert_eq!(status, std::ptr::null_mut());
            assert_ne!(env_ptr, std::ptr::null_mut());

            // Disable telemetry by default
            let status = unsafe { (*g_ort()).DisableTelemetryEvents.unwrap()(env_ptr) };
            status_to_result(status).map_err(OrtError::Environment)?;

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
}

/// Wrapper around the ONNX environment singleton
///
/// **NOTE**: Since ONNX can only define one environment per process,
/// creating multiple environments will
/// end up re-using the same environment internally; a new one will _not_
/// be created.
#[derive(Debug)]
pub struct Env {
    inner: Arc<Mutex<NamedEnv>>,
}

impl Env {
    /// Create a new environment builder using default values
    /// (name: `default`, log level: [LoggingLevel::Warning](../enum.LoggingLevel.html#variant.Warning))
    pub fn builder() -> EnvBuilder {
        EnvBuilder {
            name: "default".into(),
            log_level: LoggingLevel::Warning,
        }
    }

    /// Create a new [`SessionBuilder`](../session/struct.SessionBuilder.html)
    /// used to create a new ONNX session.
    pub fn new_session_builder(&self) -> Result<SessionBuilder> {
        SessionBuilder::new(self.inner.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_env() {
        let env1 = Env::builder().with_name("test1").build().unwrap();

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

        let env2 = Env::builder().with_name("test2").build().unwrap();

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

        let env3 = Env::builder().with_name("test3").build().unwrap();

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
