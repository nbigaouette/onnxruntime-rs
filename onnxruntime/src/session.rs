//! Module containing session types

use std::{
    ffi::CString,
    fmt::Debug,
    path::Path,
    sync::{Arc, Mutex},
};

#[cfg(feature = "model-fetching")]
use std::env;

use ndarray::Array;

use onnxruntime_sys as sys;

use crate::{
    char_p_to_string,
    environment::NamedEnvironment,
    error::{status_to_result, OrtError, Result},
    g_ort,
    memory::MemoryInfo,
    tensor::{
        ort_owned_tensor::{OrtOwnedTensor, OrtOwnedTensorExtractor},
        OrtTensor,
    },
    AllocatorType, GraphOptimizationLevel, MemType, TensorElementDataType,
    TypeToTensorElementDataType,
};

#[cfg(feature = "model-fetching")]
use crate::{download::AvailableOnnxModel, error::OrtDownloadError};

/// Type used to create a session using the _builder pattern_
///
/// A `SessionBuilder` is created by calling the
/// [`Environment::new_session_builder()`](../env/struct.Environment.html#method.new_session_builder)
/// method on the environment.
///
/// Once created, use the different methods to configure the session.
///
/// Once configured, use the [`SessionBuilder::with_model_from_file()`](../session/struct.SessionBuilder.html#method.with_model_from_file)
/// method to "commit" the builder configuration into a [`Session`](../session/struct.Session.html).
///
/// # Example
///
/// ```no_run
/// # use std::error::Error;
/// # use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let environment = Environment::builder()
///     .with_name("test")
///     .with_log_level(LoggingLevel::Verbose)
///     .build()?;
/// let mut session = environment
///     .new_session_builder()?
///     .with_optimization_level(GraphOptimizationLevel::Basic)?
///     .with_number_threads(1)?
///     .with_model_from_file("squeezenet.onnx")?;
/// # Ok(())
/// # }
/// ```
pub struct SessionBuilder {
    env: Arc<Mutex<NamedEnvironment>>,

    session_options_ptr: *mut sys::OrtSessionOptions,

    allocator: AllocatorType,
    memory_type: MemType,
}

impl Drop for SessionBuilder {
    fn drop(&mut self) {
        println!("Dropping the session options.");
        unsafe { (*g_ort()).ReleaseSessionOptions.unwrap()(self.session_options_ptr) };
    }
}

impl SessionBuilder {
    pub(crate) fn new(env: Arc<Mutex<NamedEnvironment>>) -> Result<SessionBuilder> {
        let mut session_options_ptr: *mut sys::OrtSessionOptions = std::ptr::null_mut();
        let status = unsafe { (*g_ort()).CreateSessionOptions.unwrap()(&mut session_options_ptr) };
        status_to_result(status).map_err(OrtError::SessionOptions)?;
        assert_eq!(status, std::ptr::null_mut());
        assert_ne!(session_options_ptr, std::ptr::null_mut());

        Ok(SessionBuilder {
            env,
            session_options_ptr,
            allocator: AllocatorType::Arena,
            memory_type: MemType::Default,
        })
    }

    /// Configure the session to use a number of threads
    pub fn with_number_threads(self, num_threads: i16) -> Result<SessionBuilder> {
        // We use a u16 in the builder to cover the 16-bits positive values of a i32.
        let num_threads = num_threads as i32;
        let status = unsafe {
            (*g_ort()).SetIntraOpNumThreads.unwrap()(self.session_options_ptr, num_threads)
        };
        status_to_result(status).map_err(OrtError::SessionOptions)?;
        assert_eq!(status, std::ptr::null_mut());
        Ok(self)
    }

    /// Set the session's optimization level
    pub fn with_optimization_level(
        self,
        opt_level: GraphOptimizationLevel,
    ) -> Result<SessionBuilder> {
        // Sets graph optimization level
        let opt_level = opt_level as u32;
        unsafe {
            (*g_ort()).SetSessionGraphOptimizationLevel.unwrap()(
                self.session_options_ptr,
                opt_level,
            )
        };
        Ok(self)
    }

    /// Set the session's allocator
    ///
    /// Defaults to [`AllocatorType::Arena`](../enum.AllocatorType.html#variant.Arena)
    pub fn with_allocator(mut self, allocator: AllocatorType) -> Result<SessionBuilder> {
        self.allocator = allocator;
        Ok(self)
    }

    /// Set the session's memory type
    ///
    /// Defaults to [`MemType::Default`](../enum.MemType.html#variant.Default)
    pub fn with_memory_type(mut self, memory_type: MemType) -> Result<SessionBuilder> {
        self.memory_type = memory_type;
        Ok(self)
    }

    /// Download an ONNX pre-trained model from the [ONNX Model Zoo](https://github.com/onnx/models) and commit the session
    #[cfg(feature = "model-fetching")]
    pub fn with_model_downloaded<M>(self, model: M) -> Result<Session>
    where
        M: Into<AvailableOnnxModel>,
    {
        self.with_model_downloaded_monomorphized(model.into())
    }

    #[cfg(feature = "model-fetching")]
    fn with_model_downloaded_monomorphized(self, model: AvailableOnnxModel) -> Result<Session> {
        let download_dir = env::current_dir().map_err(OrtDownloadError::IoError)?;
        let downloaded_path = model.download_to(download_dir)?;
        self.with_model_from_file_monomorphized(downloaded_path.as_ref())
    }

    // TODO: Add all functions changing the options.
    //       See all OrtApi methods taking a `options: *mut OrtSessionOptions`.

    /// Load an ONNX graph from a file and commit the session
    pub fn with_model_from_file<P>(self, model_filepath: P) -> Result<Session>
    where
        P: AsRef<Path>,
    {
        self.with_model_from_file_monomorphized(model_filepath.as_ref())
    }

    fn with_model_from_file_monomorphized(self, model_filepath: &Path) -> Result<Session> {
        let env_ptr: *const sys::OrtEnv = *self.env.lock().unwrap().env_ptr.0.get_mut();
        let mut session_ptr: *mut sys::OrtSession = std::ptr::null_mut();

        if !model_filepath.exists() {
            return Err(OrtError::FileDoesNotExists {
                filename: model_filepath.to_path_buf(),
            });
        }
        let model_path: CString =
            CString::new(
                model_filepath
                    .to_str()
                    .ok_or_else(|| OrtError::NonUtf8Path {
                        path: model_filepath.to_path_buf(),
                    })?,
            )?;

        let status = unsafe {
            (*g_ort()).CreateSession.unwrap()(
                env_ptr,
                model_path.as_ptr(),
                self.session_options_ptr,
                &mut session_ptr,
            )
        };
        status_to_result(status).map_err(OrtError::Session)?;
        assert_eq!(status, std::ptr::null_mut());
        assert_ne!(session_ptr, std::ptr::null_mut());

        let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
        let status =
            unsafe { (*g_ort()).GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr) };
        status_to_result(status).map_err(OrtError::Allocator)?;
        assert_eq!(status, std::ptr::null_mut());
        assert_ne!(allocator_ptr, std::ptr::null_mut());

        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default)?;

        // Extract input and output properties
        let num_input_nodes = dangerous::extract_inputs_count(session_ptr)?;
        let num_output_nodes = dangerous::extract_outputs_count(session_ptr)?;
        let inputs = (0..num_input_nodes)
            .map(|i| dangerous::extract_input(session_ptr, allocator_ptr, i))
            .collect::<Result<Vec<Input>>>()?;
        let outputs = (0..num_output_nodes)
            .map(|i| dangerous::extract_output(session_ptr, allocator_ptr, i))
            .collect::<Result<Vec<Output>>>()?;

        Ok(Session {
            session_ptr,
            allocator_ptr,
            memory_info,
            inputs,
            outputs,
        })
    }
}

/// Type storing the session information, built from an [`Environment`](environment/struct.Environment.html)
#[derive(Debug)]
pub struct Session {
    session_ptr: *mut sys::OrtSession,
    allocator_ptr: *mut sys::OrtAllocator,
    memory_info: MemoryInfo,
    /// Information about the ONNX's inputs as stored in loaded file
    pub inputs: Vec<Input>,
    /// Information about the ONNX's outputs as stored in loaded file
    pub outputs: Vec<Output>,
}

/// Information about an ONNX's input as stored in loaded file
#[derive(Debug)]
pub struct Input {
    /// Name of the input layer
    pub name: String,
    /// Type of the input layer's elements
    pub input_type: TensorElementDataType,
    /// Shape of the input layer
    ///
    /// C API uses a i64 for the dimensions. We use an unsigned of the same range of the positive values.
    pub dimensions: Vec<u32>,
}

/// Information about an ONNX's output as stored in loaded file
#[derive(Debug)]
pub struct Output {
    /// Name of the output layer
    pub name: String,
    /// Type of the output layer's elements
    pub output_type: TensorElementDataType,
    /// Shape of the output layer
    ///
    /// C API uses a i64 for the dimensions. We use an unsigned of the same range of the positive values.
    pub dimensions: Vec<u32>,
}

impl Input {
    /// Return an iterator over the shape elements of the input layer
    ///
    /// Note: The member [`Input::dimensions`](struct.Input.html#structfield.dimensions)
    /// stores `u32` (since ONNX uses `i64` but which cannot be negative) so the
    /// iterator converts to `usize`.
    pub fn dimensions(&self) -> impl Iterator<Item = usize> + '_ {
        self.dimensions.iter().map(|d| *d as usize)
    }
}

impl Output {
    /// Return an iterator over the shape elements of the output layer
    ///
    /// Note: The member [`Output::dimensions`](struct.Output.html#structfield.dimensions)
    /// stores `u32` (since ONNX uses `i64` but which cannot be negative) so the
    /// iterator converts to `usize`.
    pub fn dimensions(&self) -> impl Iterator<Item = usize> + '_ {
        self.dimensions.iter().map(|d| *d as usize)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        println!("Dropping the session.");
        unsafe { (*g_ort()).ReleaseSession.unwrap()(self.session_ptr) };
        // FIXME: There is no C function to release the allocator?

        self.session_ptr = std::ptr::null_mut();
        self.allocator_ptr = std::ptr::null_mut();
    }
}

impl Session {
    /// Run the input data through the ONNX graph, performing inference.
    ///
    /// Note that ONNX models can have multiple inputs; a `Vec<_>` is thus
    /// used for the input data here.
    pub fn run<'s, 't, 'm, T, D>(
        &'s mut self,
        input_arrays: Vec<Array<T, D>>,
    ) -> Result<Vec<OrtOwnedTensor<'t, 'm, T, ndarray::IxDyn>>>
    where
        T: TypeToTensorElementDataType + Debug + Clone,
        D: ndarray::Dimension,
        'm: 't, // 'm outlives 't (memory info outlives tensor)
        's: 'm, // 's outlives 'm (session outlives memory info)
    {
        // Make sure all dimensions match
        for (input_idx, input_array) in input_arrays.iter().enumerate() {
            let inputs_shape_as_usize: Vec<usize> = self.inputs[input_idx]
                .dimensions
                .iter()
                .map(|d| *d as usize)
                .collect();

            if input_array.shape() != inputs_shape_as_usize.as_slice() {
                return Err(OrtError::NonMatchingDimensions {
                    input: input_array.shape().to_vec(),
                    model: inputs_shape_as_usize,
                });
            }
        }

        let input_names: Vec<String> = self.inputs.iter().map(|input| input.name.clone()).collect();
        let input_names_cstring: Vec<CString> = input_names
            .iter()
            .cloned()
            .map(|n| CString::new(n).unwrap())
            .collect();
        let input_names_ptr: Vec<*const i8> = input_names_cstring
            .into_iter()
            .map(|n| n.into_raw() as *const i8)
            .collect();
        let input_names_ptr_ptr: *const *const i8 = input_names_ptr.as_ptr();

        let output_names: Vec<String> = self
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();
        let output_names_cstring: Vec<CString> = output_names
            .into_iter()
            .map(|n| CString::new(n).unwrap())
            .collect();
        let output_names_ptr: Vec<*const i8> = output_names_cstring
            .iter()
            .map(|n| n.as_ptr() as *const i8)
            .collect();
        let output_names_ptr_ptr: *const *const i8 = output_names_ptr.as_ptr();

        let mut outputs: Vec<OrtOwnedTensor<T, ndarray::Dim<ndarray::IxDynImpl>>> =
            Vec::with_capacity(input_arrays.len());

        for (input_idx, input_array) in input_arrays.into_iter().enumerate() {
            let input_tensor = OrtTensor::from_array(&self.memory_info, input_array)?;

            let input_tensor_ptr2: *const sys::OrtValue =
                input_tensor.c_ptr as *const sys::OrtValue;
            let input_tensor_ptr3: *const *const sys::OrtValue = &input_tensor_ptr2;

            // score model & input tensor, get back output tensor

            // FIXME: Still required to prevent leaking?
            // let input_node_names_cstring =
            //     unsafe { std::ffi::CString::from_raw(input_node_names_ptr[0] as *mut i8) };

            let output_shape = self.outputs[input_idx]
                .dimensions
                .iter()
                .map(|d| *d as usize)
                .collect::<Vec<usize>>();
            let mut output_tensor_extractor =
                OrtOwnedTensorExtractor::new(&self.memory_info, ndarray::IxDyn(&output_shape));

            let run_options_ptr: *const sys::OrtRunOptions = std::ptr::null();

            // FIXME: Why would we Run() once per input? Should there be only one Run() for all inputs?
            //        Or is this what the '1's mean (number of inputs and outputs)?
            // FIXME: This leaks. output_tensor_ptr_ptr Gets allocated inside C code
            //        BUT, it checks if its null (onnxruntime_c_api.cc, line 493) so
            //        maybe we can pass our own pointer, allocated from Rust?
            //        Note that 'output[i]' (line 493) dereference 'output_tensor_ptr_ptr'
            //        which thus points to 'output_tensor_ptr', which itself is NULL.
            //        Line 515 allocates using 'new OrtValue' if initially null.
            let status = unsafe {
                (*g_ort()).Run.unwrap()(
                    self.session_ptr,
                    run_options_ptr,
                    input_names_ptr_ptr,
                    input_tensor_ptr3,
                    1, // FIXME: This should be the lengths of the input vector, not just 1
                    output_names_ptr_ptr,
                    1, // FIXME: This should be the lengths of the output vector, not just 1
                    &mut output_tensor_extractor.tensor_ptr,
                )
            };
            status_to_result(status).map_err(OrtError::Run)?;

            let output_tensor: OrtOwnedTensor<T, ndarray::Dim<ndarray::IxDynImpl>> =
                output_tensor_extractor.extract::<T>()?;

            outputs.push(output_tensor);
        }

        let _: Vec<CString> = input_names_ptr
            .into_iter()
            .map(|p| {
                assert_ne!(p, std::ptr::null());
                unsafe { CString::from_raw(p as *mut i8) }
            })
            .collect();

        Ok(outputs)
    }

    // pub fn tensor_from_array<'a, 'b, T, D>(&'a self, array: Array<T, D>) -> Tensor<'b, T, D>
    // where
    //     'a: 'b, // 'a outlives 'b
    // {
    //     Tensor::from_array(self, array)
    // }
}

/// This module contains dangerous functions working on raw pointers.
/// Those functions are only to be used from inside the
/// `SessionBuilder::with_model_from_file_monomorphized()` method.
mod dangerous {
    use super::*;

    pub(super) fn extract_inputs_count(session_ptr: *mut sys::OrtSession) -> Result<u64> {
        let f = unsafe { *g_ort() }.SessionGetInputCount.unwrap();
        extract_io_count(f, session_ptr)
    }

    pub(super) fn extract_outputs_count(session_ptr: *mut sys::OrtSession) -> Result<u64> {
        let f = unsafe { *g_ort() }.SessionGetOutputCount.unwrap();
        extract_io_count(f, session_ptr)
    }

    fn extract_io_count(
        f: unsafe extern "C" fn(*const sys::OrtSession, *mut u64) -> *mut sys::OrtStatus,
        session_ptr: *mut sys::OrtSession,
    ) -> Result<u64> {
        let mut num_nodes: u64 = 0;
        let status = unsafe { f(session_ptr, &mut num_nodes) };
        status_to_result(status).map_err(OrtError::InOutCount)?;
        assert_eq!(status, std::ptr::null_mut());
        assert_ne!(num_nodes, 0);
        Ok(num_nodes)
    }

    fn extract_input_name(
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: u64,
    ) -> Result<String> {
        let f = unsafe { *g_ort() }.SessionGetInputName.unwrap();
        extract_io_name(f, session_ptr, allocator_ptr, i)
    }

    fn extract_output_name(
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: u64,
    ) -> Result<String> {
        let f = unsafe { *g_ort() }.SessionGetOutputName.unwrap();
        extract_io_name(f, session_ptr, allocator_ptr, i)
    }

    fn extract_io_name(
        f: unsafe extern "C" fn(
            *const sys::OrtSession,
            u64,
            *mut sys::OrtAllocator,
            *mut *mut i8,
        ) -> *mut sys::OrtStatus,
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: u64,
    ) -> Result<String> {
        let mut name_bytes: *mut i8 = std::ptr::null_mut();

        let status = unsafe { f(session_ptr, i, allocator_ptr, &mut name_bytes) };
        status_to_result(status).map_err(OrtError::InputName)?;
        assert_ne!(name_bytes, std::ptr::null_mut());

        // FIXME: Is it safe to keep ownership of the memory?
        let name = char_p_to_string(name_bytes)?;

        Ok(name)
    }

    pub(super) fn extract_input(
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: u64,
    ) -> Result<Input> {
        let input_name = extract_input_name(session_ptr, allocator_ptr, i)?;
        let f = unsafe { *g_ort() }.SessionGetInputTypeInfo.unwrap();
        let (input_type, dimensions) = extract_io(f, session_ptr, i)?;
        Ok(Input {
            name: input_name,
            input_type,
            dimensions,
        })
    }

    pub(super) fn extract_output(
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: u64,
    ) -> Result<Output> {
        let output_name = extract_output_name(session_ptr, allocator_ptr, i)?;
        let f = unsafe { *g_ort() }.SessionGetOutputTypeInfo.unwrap();
        let (output_type, dimensions) = extract_io(f, session_ptr, i)?;
        Ok(Output {
            name: output_name,
            output_type,
            dimensions,
        })
    }

    fn extract_io(
        f: unsafe extern "C" fn(
            *const sys::OrtSession,
            u64,
            *mut *mut sys::OrtTypeInfo,
        ) -> *mut sys::OrtStatus,
        session_ptr: *mut sys::OrtSession,
        i: u64,
    ) -> Result<(TensorElementDataType, Vec<u32>)> {
        let mut typeinfo_ptr: *mut sys::OrtTypeInfo = std::ptr::null_mut();

        let status = unsafe { f(session_ptr, i as u64, &mut typeinfo_ptr) };
        status_to_result(status).map_err(OrtError::GetTypeInfo)?;
        assert_ne!(typeinfo_ptr, std::ptr::null_mut());

        let mut tensor_info_ptr: *const sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let status = unsafe {
            (*g_ort()).CastTypeInfoToTensorInfo.unwrap()(typeinfo_ptr, &mut tensor_info_ptr)
        };
        status_to_result(status).map_err(OrtError::CastTypeInfoToTensorInfo)?;
        assert_ne!(tensor_info_ptr, std::ptr::null_mut());

        let mut type_sys: sys::ONNXTensorElementDataType = 0;
        let status =
            unsafe { (*g_ort()).GetTensorElementType.unwrap()(tensor_info_ptr, &mut type_sys) };
        status_to_result(status).map_err(OrtError::TensorElementType)?;
        assert_ne!(type_sys, 0);
        // This transmute should be safe since its value is read from GetTensorElementType which we must trust.
        let io_type: TensorElementDataType = unsafe { std::mem::transmute(type_sys) };

        // println!("{} : type={}", i, type_);

        // print input shapes/dims
        let mut num_dims = 0;
        let status =
            unsafe { (*g_ort()).GetDimensionsCount.unwrap()(tensor_info_ptr, &mut num_dims) };
        status_to_result(status).map_err(OrtError::GetDimensionsCount)?;
        assert_ne!(num_dims, 0);

        // println!("{} : num_dims={}", i, num_dims);
        let mut node_dims: Vec<i64> = vec![0; num_dims as usize];
        let status = unsafe {
            (*g_ort()).GetDimensions.unwrap()(
                tensor_info_ptr,
                node_dims.as_mut_ptr(), // FIXME: UB?
                num_dims,
            )
        };
        status_to_result(status).map_err(OrtError::GetDimensions)?;

        // for j in 0..num_dims {
        //     println!("{} : dim {}={}", i, j, node_dims[j as usize]);
        // }

        unsafe { (*g_ort()).ReleaseTypeInfo.unwrap()(typeinfo_ptr) };

        Ok((io_type, node_dims.into_iter().map(|d| d as u32).collect()))
    }
}
