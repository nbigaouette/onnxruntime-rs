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
        println!("Droping the session options.");
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

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { (*g_ort()).ReleaseSession.unwrap()(self.session_ptr) };
        // FIXME: There is no C function to release the allocator?
        unsafe { (*g_ort()).ReleaseMemoryInfo.unwrap()(self.memory_info_ptr) };

        self.session_ptr = std::ptr::null_mut();
        self.allocator_ptr = std::ptr::null_mut();
        self.memory_info_ptr = std::ptr::null_mut();
    }
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

// pub struct SessionBuilder {
//     pub(crate) inner: Arc<Mutex<NamedEnv>>,

//     pub(crate) name: String,
//     pub(crate) options: Option<SessionOptions>,
//     pub(crate) opt_level: GraphOptimizationLevel,
//     pub(crate) num_threads: i16,
//     pub(crate) model_filename: PathBuf,
//     pub(crate) use_cuda: bool,
// }

// impl SessionBuilder {
//     pub fn with_options(mut self, options: SessionOptions) -> SessionBuilder {
//         self.options = Some(options);
//         self
//     }

//     pub fn with_cuda(mut self, use_cuda: bool) -> SessionBuilder {
//         unimplemented!()
//         // self.use_cuda = use_cuda;
//         // self
//     }

//     pub fn with_optimization_level(mut self, opt_level: GraphOptimizationLevel) -> SessionBuilder {
//         self.opt_level = opt_level;
//         self
//     }

//     pub fn with_number_threads(mut self, num_threads: i16) -> SessionBuilder {
//         self.num_threads = num_threads;
//         self
//     }

//     pub fn build(self) -> Result<Session> {
//         // let mut session_options_ptr: *mut sys::OrtSessionOptions = std::ptr::null_mut();
//         // let status = unsafe { (*g_ort()).CreateSessionOptions.unwrap()(&mut session_options_ptr) };
//         // status_to_result(status).map_err(OrtError::SessionOptions)?;
//         // assert_eq!(status, std::ptr::null_mut());
//         // assert_ne!(session_options_ptr, std::ptr::null_mut());

//         // match self.options {
//         //     Some(_options) => unimplemented!(),
//         //     None => {}
//         // }

//         // // We use a u16 in the builder to cover the 16-bits positive values of a i32.
//         // let num_threads = self.num_threads as i32;
//         // unsafe { (*g_ort()).SetIntraOpNumThreads.unwrap()(session_options_ptr, num_threads) };

//         // // Sets graph optimization level
//         // let opt_level = self.opt_level as u32;
//         // unsafe {
//         //     (*g_ort()).SetSessionGraphOptimizationLevel.unwrap()(session_options_ptr, opt_level)
//         // };

//         let env_ptr: *const sys::OrtEnv = *self.inner.lock().unwrap().env_ptr.0.get_mut();
//         let mut session_ptr: *mut sys::OrtSession = std::ptr::null_mut();

//         if !self.model_filename.exists() {
//             return Err(OrtError::FileDoesNotExists {
//                 filename: self.model_filename.clone(),
//             });
//         }
//         let model_filename = self.model_filename.clone();
//         let model_path: CString =
//             CString::new(
//                 self.model_filename
//                     .to_str()
//                     .ok_or_else(|| OrtError::NonUtf8Path {
//                         path: model_filename,
//                     })?,
//             )?;

//         let status = unsafe {
//             (*g_ort()).CreateSession.unwrap()(
//                 env_ptr,
//                 model_path.as_ptr(),
//                 session_options_ptr,
//                 &mut session_ptr,
//             )
//         };
//         status_to_result(status).map_err(OrtError::Session)?;
//         assert_eq!(status, std::ptr::null_mut());
//         assert_ne!(session_ptr, std::ptr::null_mut());

//         let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
//         let status =
//             unsafe { (*g_ort()).GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr) };
//         status_to_result(status).map_err(OrtError::Allocator)?;
//         assert_eq!(status, std::ptr::null_mut());
//         assert_ne!(allocator_ptr, std::ptr::null_mut());

//         Ok(Session {
//             session_options_ptr,
//             session_ptr,
//             allocator_ptr,
//         })
//     }
// }

// impl Session {
//     fn read_inputs_count(&self) -> Result<u64> {
//         let mut num_input_nodes: u64 = 0;
//         let status = unsafe {
//             (*g_ort()).SessionGetInputCount.unwrap()(self.session_ptr, &mut num_input_nodes)
//         };
//         status_to_result(status).map_err(OrtError::Allocator)?;
//         assert_eq!(status, std::ptr::null_mut());
//         assert_ne!(num_input_nodes, 0);
//         Ok(num_input_nodes)
//     }

//     fn read_input_name(&self, i: u64) -> Result<String> {
//         let mut input_name_bytes: *mut i8 = std::ptr::null_mut();

//         let status = unsafe {
//             (*g_ort()).SessionGetInputName.unwrap()(
//                 self.session_ptr,
//                 i,
//                 self.allocator_ptr,
//                 &mut input_name_bytes,
//             )
//         };
//         status_to_result(status).map_err(OrtError::InputName)?;
//         assert_ne!(input_name_bytes, std::ptr::null_mut());

//         // FIXME: Is it safe to keep ownership of the memory
//         let input_name = char_p_to_string(input_name_bytes)?;

//         Ok(input_name)
//     }

//     fn read_input(&self, i: u64) -> Result<Input> {
//         let input_name = self.read_input_name(i)?;

//         let mut typeinfo_ptr: *mut sys::OrtTypeInfo = std::ptr::null_mut();

//         let status = unsafe {
//             (*g_ort()).SessionGetInputTypeInfo.unwrap()(
//                 self.session_ptr,
//                 i as u64,
//                 &mut typeinfo_ptr,
//             )
//         };
//         status_to_result(status).map_err(OrtError::GetInputTypeInfo)?;
//         assert_ne!(typeinfo_ptr, std::ptr::null_mut());

//         let mut tensor_info_ptr: *const sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
//         let status = unsafe {
//             (*g_ort()).CastTypeInfoToTensorInfo.unwrap()(typeinfo_ptr, &mut tensor_info_ptr)
//         };
//         status_to_result(status).map_err(OrtError::CastTypeInfoToTensorInfo)?;
//         assert_ne!(tensor_info_ptr, std::ptr::null_mut());

//         let mut input_type_sys: sys::ONNXTensorElementDataType = 0;
//         let status = unsafe {
//             (*g_ort()).GetTensorElementType.unwrap()(tensor_info_ptr, &mut input_type_sys)
//         };
//         status_to_result(status).map_err(OrtError::TensorElementType)?;
//         assert_ne!(input_type_sys, 0);
//         // This transmute should be safe since its value is read from GetTensorElementType which we must trust.
//         let input_type: TensorElementDataType = unsafe { std::mem::transmute(input_type_sys) };

//         // println!("Input {} : type={}", i, type_);

//         // print input shapes/dims
//         let mut num_dims = 0;
//         let status =
//             unsafe { (*g_ort()).GetDimensionsCount.unwrap()(tensor_info_ptr, &mut num_dims) };
//         status_to_result(status).map_err(OrtError::GetDimensionsCount)?;
//         assert_ne!(num_dims, 0);

//         // println!("Input {} : num_dims={}", i, num_dims);
//         let mut input_node_dims: Vec<i64> = vec![0; num_dims as usize];
//         let status = unsafe {
//             (*g_ort()).GetDimensions.unwrap()(
//                 tensor_info_ptr,
//                 input_node_dims.as_mut_ptr(), // FIXME: UB?
//                 num_dims,
//             )
//         };
//         status_to_result(status).map_err(OrtError::GetDimensions)?;

//         // for j in 0..num_dims {
//         //     println!("Input {} : dim {}={}", i, j, input_node_dims[j as usize]);
//         // }

//         unsafe { (*g_ort()).ReleaseTypeInfo.unwrap()(typeinfo_ptr) };

//         Ok(Input {
//             name: input_name,
//             input_type: input_type,
//             dimensions: input_node_dims.into_iter().map(|d| d as u32).collect(),
//         })
//     }

//     pub fn read_inputs(&self) -> Result<Vec<Input>> {
//         let num_input_nodes = self.read_inputs_count()?;

//         (0..num_input_nodes)
//             .map(|i| self.read_input(i))
//             .collect::<Result<Vec<Input>>>()
//     }
//     /*
//     // FIXME: Use ndarray instead of flatten 1D vector
//     pub fn run<T>(&mut self, mut flatten_array: Vec<T>, input_node_dims: &[u32]) -> Result<()>
//     where
//         T: TypeToTensorElementDataType,
//     {
//         if flatten_array.len() != input_node_dims.iter().map(|d| *d as usize).product() {
//             return Err(OrtError::NonMatchingDimensions);
//         }

//         // create input tensor object from data values
//         let mut memory_info_ptr: *mut sys::OrtMemoryInfo = std::ptr::null_mut();
//         let status = unsafe {
//             (*g_ort()).CreateCpuMemoryInfo.unwrap()(
//                 AllocatorType::Arena as i32, // FIXME: Pass as argument
//                 MemType::Default as i32,     // FIXME: Pass as argument
//                 &mut memory_info_ptr,
//             )
//         };
//         status_to_result(status).map_err(OrtError::CreateCpuMemoryInfo)?;
//         assert_ne!(memory_info_ptr, std::ptr::null_mut());

//         let mut input_tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();
//         let input_tensor_ptr_ptr: *mut *mut sys::OrtValue = &mut input_tensor_ptr;
//         let input_tensor_values_ptr: *mut std::ffi::c_void =
//             flatten_array.as_mut_ptr() as *mut std::ffi::c_void;
//         assert_ne!(input_tensor_values_ptr, std::ptr::null_mut());

//         // For API calls, we need i64, not u32. We use u32 in the safe API to prevent negative values.
//         let input_node_dims_i64: Vec<i64> = input_node_dims.iter().map(|d| *d as i64).collect();
//         let shape: *const i64 = input_node_dims_i64.as_ptr();
//         assert_ne!(shape, std::ptr::null_mut());

//         let status = unsafe {
//             (*g_ort()).CreateTensorWithDataAsOrtValue.unwrap()(
//                 memory_info_ptr,
//                 input_tensor_values_ptr,
//                 (flatten_array.len() * std::mem::size_of::<f32>()) as u64,
//                 shape,
//                 4,
//                 T::tensor_element_data_type() as u32,
//                 input_tensor_ptr_ptr,
//             )
//         };
//         status_to_result(status).map_err(OrtError::CreateTensorWithData)?;
//         assert_ne!(input_tensor_ptr, std::ptr::null_mut());

//         let mut is_tensor = 0;
//         let status = unsafe { (*g_ort()).IsTensor.unwrap()(input_tensor_ptr, &mut is_tensor) };
//         status_to_result(status).map_err(OrtError::IsTensor)?;
//         assert_eq!(is_tensor, 1);

//         unsafe { (*g_ort()).ReleaseMemoryInfo.unwrap()(memory_info_ptr) };

//         let input_tensor_ptr2: *const sys::OrtValue = input_tensor_ptr as *const sys::OrtValue;
//         let input_tensor_ptr3: *const *const sys::OrtValue = &input_tensor_ptr2;

//         // score model & input tensor, get back output tensor

//         let input_node_names_cstring: Vec<std::ffi::CString> = input_node_names
//             .into_iter()
//             .map(|n| std::ffi::CString::new(n).unwrap())
//             .collect();
//         let input_node_names_ptr: Vec<*const i8> = input_node_names_cstring
//             .into_iter()
//             .map(|n| n.into_raw() as *const i8)
//             .collect();
//         let input_node_names_ptr_ptr: *const *const i8 = input_node_names_ptr.as_ptr();

//         // let output_node_names_cstring: Vec<std::ffi::CString> = output_node_names
//         //     .into_iter()
//         //     .map(|n| std::ffi::CString::new(n.clone()).unwrap())
//         //     .collect();
//         // let output_node_names_ptr: Vec<*const i8> = output_node_names_cstring
//         //     .iter()
//         //     .map(|n| n.as_ptr() as *const i8)
//         //     .collect();
//         // let output_node_names_ptr_ptr: *const *const i8 = output_node_names_ptr.as_ptr();

//         // let input_node_names_cstring =
//         //     unsafe { std::ffi::CString::from_raw(input_node_names_ptr[0] as *mut i8) };
//         // let run_options_ptr: *const OrtRunOptions = std::ptr::null();
//         // let mut output_tensor_ptr: *mut OrtValue = std::ptr::null_mut();
//         // let mut output_tensor_ptr_ptr: *mut *mut OrtValue = &mut output_tensor_ptr;

//         // let status = unsafe {
//         //     g_ort.as_ref().unwrap().Run.unwrap()(
//         //         session_ptr,
//         //         run_options_ptr,
//         //         input_node_names_ptr_ptr,
//         //         input_tensor_ptr3,
//         //         1,
//         //         output_node_names_ptr_ptr,
//         //         1,
//         //         output_tensor_ptr_ptr,
//         //     )
//         // };
//         // CheckStatus(g_ort, status).unwrap();
//         // assert_ne!(output_tensor_ptr, std::ptr::null_mut());

//         // let mut is_tensor = 0;
//         // let status =
//         //     unsafe { g_ort.as_ref().unwrap().IsTensor.unwrap()(output_tensor_ptr, &mut is_tensor) };
//         // CheckStatus(g_ort, status).unwrap();
//         // assert_eq!(is_tensor, 1);

//         // // Get pointer to output tensor float values
//         // let mut floatarr: *mut f32 = std::ptr::null_mut();
//         // let floatarr_ptr: *mut *mut f32 = &mut floatarr;
//         // let floatarr_ptr_void: *mut *mut std::ffi::c_void = floatarr_ptr as *mut *mut std::ffi::c_void;
//         // let status = unsafe {
//         //     g_ort.as_ref().unwrap().GetTensorMutableData.unwrap()(output_tensor_ptr, floatarr_ptr_void)
//         // };
//         // CheckStatus(g_ort, status).unwrap();
//         // assert_ne!(floatarr, std::ptr::null_mut());

//         unimplemented!()
//         // Ok(())
//     }
//     */
// }
