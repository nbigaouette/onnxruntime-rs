use std::{
    ffi::CString,
    path::Path,
    sync::{Arc, Mutex},
};

use onnxruntime_sys as sys;

use crate::{
    char_p_to_string,
    env::NamedEnv,
    error::{status_to_result, OrtError, Result},
    g_ort, AllocatorType, GraphOptimizationLevel, MemType, TensorElementDataType,
    TypeToTensorElementDataType,
};

pub struct SessionBuilder {
    env: Arc<Mutex<NamedEnv>>,

    session_options_ptr: *mut sys::OrtSessionOptions,

    allocator: AllocatorType,
    memory_type: MemType,
}

impl SessionBuilder {
    pub(crate) fn new(env: Arc<Mutex<NamedEnv>>) -> Result<SessionBuilder> {
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

    pub fn load_model_from_file<P>(self, model_filepath: P) -> Result<Session>
    where
        P: AsRef<Path>,
    {
        self.load_model_from_file_monorphomized(model_filepath.as_ref())
    }

    fn load_model_from_file_monorphomized(self, model_filepath: &Path) -> Result<Session> {
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

        let mut memory_info_ptr: *mut sys::OrtMemoryInfo = std::ptr::null_mut();
        let status = unsafe {
            (*g_ort()).CreateCpuMemoryInfo.unwrap()(
                self.allocator.clone() as i32,
                self.memory_type.clone() as i32,
                &mut memory_info_ptr,
            )
        };
        status_to_result(status).map_err(OrtError::CreateCpuMemoryInfo)?;
        assert_ne!(memory_info_ptr, std::ptr::null_mut());

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
            memory_info_ptr,
            inputs,
            outputs,
        })
    }

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

    pub fn with_allocator(mut self, allocator: AllocatorType) -> Result<SessionBuilder> {
        self.allocator = allocator;
        Ok(self)
    }

    pub fn with_memory_type(mut self, memory_type: MemType) -> Result<SessionBuilder> {
        self.memory_type = memory_type;
        Ok(self)
    }

    // TODO: Add all functions changing the options.
    //       See all OrtApi methods taking a `options: *mut OrtSessionOptions`.
}

impl Drop for SessionBuilder {
    fn drop(&mut self) {
        println!("Dropping the session options.");
        unsafe { (*g_ort()).ReleaseSessionOptions.unwrap()(self.session_options_ptr) };
    }
}

pub struct Session {
    session_ptr: *mut sys::OrtSession,
    allocator_ptr: *mut sys::OrtAllocator,
    memory_info_ptr: *mut sys::OrtMemoryInfo,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
}

#[derive(Debug)]
pub struct Input {
    pub name: String,
    pub input_type: TensorElementDataType,
    /// C API uses a i64 for the dimensions. We use an unsigned of the same range of the positive values.
    pub dimensions: Vec<u32>,
}

#[derive(Debug)]
pub struct Output {
    pub name: String,
    pub output_type: TensorElementDataType,
    /// C API uses a i64 for the dimensions. We use an unsigned of the same range of the positive values.
    pub dimensions: Vec<u32>,
}

impl Drop for Session {
    fn drop(&mut self) {
        println!("Dropping the session.");
        unsafe { (*g_ort()).ReleaseSession.unwrap()(self.session_ptr) };
        // FIXME: There is no C function to release the allocator?
        println!("Dropping the memory information.");
        unsafe { (*g_ort()).ReleaseMemoryInfo.unwrap()(self.memory_info_ptr) };

        self.session_ptr = std::ptr::null_mut();
        self.allocator_ptr = std::ptr::null_mut();
        self.memory_info_ptr = std::ptr::null_mut();
    }
}

impl Session {
    // FIXME: Use ndarray instead of flatten 1D vector
    pub fn run<T>(&mut self, mut flatten_array: Vec<Vec<T>>) -> Result<Vec<Vec<T>>>
    where
        T: TypeToTensorElementDataType,
    {
        // Make sure all dimensions match
        for (input_idx, input_flatten_array) in flatten_array.iter().enumerate() {
            if input_flatten_array.len()
                != self.inputs[input_idx]
                    .dimensions
                    .iter()
                    .map(|d| *d as usize)
                    .product()
            {
                return Err(OrtError::NonMatchingDimensions);
            }
        }

        let mut outputs = Vec::with_capacity(flatten_array.len());

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
            .map(|n| CString::new(n.clone()).unwrap())
            .collect();
        let output_names_ptr: Vec<*const i8> = output_names_cstring
            .iter()
            .map(|n| n.as_ptr() as *const i8)
            .collect();
        let output_names_ptr_ptr: *const *const i8 = output_names_ptr.as_ptr();

        for (input_idx, input_flatten_array) in flatten_array.iter_mut().enumerate() {
            let mut input_tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();
            let input_tensor_ptr_ptr: *mut *mut sys::OrtValue = &mut input_tensor_ptr;
            let input_tensor_values_ptr: *mut std::ffi::c_void =
                input_flatten_array.as_mut_ptr() as *mut std::ffi::c_void;
            assert_ne!(input_tensor_values_ptr, std::ptr::null_mut());

            // For API calls, we need i64, not u32. We use u32 in the safe API to prevent negative values.
            let input_node_dims_i64: Vec<i64> = self.inputs[input_idx]
                .dimensions
                .iter()
                .map(|d| *d as i64)
                .collect();
            let shape: *const i64 = input_node_dims_i64.as_ptr();
            assert_ne!(shape, std::ptr::null_mut());

            // FIXME: This leaks
            let status = unsafe {
                (*g_ort()).CreateTensorWithDataAsOrtValue.unwrap()(
                    self.memory_info_ptr,
                    input_tensor_values_ptr,
                    (input_flatten_array.len() * std::mem::size_of::<T>()) as u64,
                    shape,
                    self.inputs[input_idx].dimensions.len() as u64,
                    T::tensor_element_data_type() as u32,
                    input_tensor_ptr_ptr,
                )
            };
            status_to_result(status).map_err(OrtError::CreateTensorWithData)?;
            assert_ne!(input_tensor_ptr, std::ptr::null_mut());

            let mut is_tensor = 0;
            let status = unsafe { (*g_ort()).IsTensor.unwrap()(input_tensor_ptr, &mut is_tensor) };
            status_to_result(status).map_err(OrtError::IsTensor)?;
            assert_eq!(is_tensor, 1);

            // unsafe { (*g_ort()).ReleaseMemoryInfo.unwrap()(memory_info_ptr) };

            let input_tensor_ptr2: *const sys::OrtValue = input_tensor_ptr as *const sys::OrtValue;
            let input_tensor_ptr3: *const *const sys::OrtValue = &input_tensor_ptr2;

            // score model & input tensor, get back output tensor

            // FIXME: Still required to prevent leaking?
            // let input_node_names_cstring =
            //     unsafe { std::ffi::CString::from_raw(input_node_names_ptr[0] as *mut i8) };

            let run_options_ptr: *const sys::OrtRunOptions = std::ptr::null();
            let mut output_tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();
            let mut output_tensor_ptr_ptr: *mut *mut sys::OrtValue = &mut output_tensor_ptr;

            // FIXME: Why would we Run() once per input? Should there be only one Run() for all inputs?
            //        Or is this what the '1's mean (number of inputs and outputs)?
            // FIXME: This leaks
            let status = unsafe {
                (*g_ort()).Run.unwrap()(
                    self.session_ptr,
                    run_options_ptr,
                    input_names_ptr_ptr,
                    input_tensor_ptr3,
                    1,
                    output_names_ptr_ptr,
                    1,
                    output_tensor_ptr_ptr,
                )
            };
            status_to_result(status).map_err(OrtError::Run)?;
            assert_ne!(output_tensor_ptr, std::ptr::null_mut());

            let mut is_tensor = 0;
            let status = unsafe { (*g_ort()).IsTensor.unwrap()(output_tensor_ptr, &mut is_tensor) };
            status_to_result(status).map_err(OrtError::IsTensor)?;
            assert_eq!(is_tensor, 1);

            // Get pointer to output tensor float values
            let mut output_array_ptr: *mut T = std::ptr::null_mut();
            let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
            let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void =
                output_array_ptr_ptr as *mut *mut std::ffi::c_void;
            let status = unsafe {
                (*g_ort()).GetTensorMutableData.unwrap()(
                    output_tensor_ptr,
                    output_array_ptr_ptr_void,
                )
            };
            status_to_result(status).map_err(OrtError::IsTensor)?;
            assert_ne!(output_array_ptr, std::ptr::null_mut());

            // FIXME: That looks like UB... Replace with newtype with custom drop impl.
            let n = self.outputs[input_idx]
                .dimensions
                .iter()
                .map(|d| *d as usize)
                .product();
            let output: Vec<T> = unsafe { Vec::from_raw_parts(output_array_ptr, n, n) };
            outputs.push(output);
        }

        let _: Vec<CString> = input_names_ptr
            .into_iter()
            .map(|p| {
                assert_ne!(p, std::ptr::null());
                unsafe { CString::from_raw(p as *mut i8) }
            })
            .collect();

        let _: Vec<CString> = output_names_ptr
            .into_iter()
            .map(|p| {
                assert_ne!(p, std::ptr::null());
                unsafe { CString::from_raw(p as *mut i8) }
            })
            .collect();

        Ok(outputs)
    }
}

/// This module contains dangerous functions working on raw pointers.
/// Those functions are only to be used from inside the
/// `SessionBuilder::load_model_from_file_monorphomized()` method.
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
