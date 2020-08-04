#![allow(non_snake_case)]

use onnxruntime_sys::*;

// https://github.com/microsoft/onnxruntime/blob/v1.3.1/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp

fn main() {
    let g_ort = unsafe { OrtGetApiBase().as_ref().unwrap().GetApi.unwrap()(ORT_API_VERSION) };
    println!("g_ort: {:?}", g_ort);
    assert_ne!(g_ort, std::ptr::null_mut());

    //*************************************************************************
    // initialize  enviroment...one enviroment per process
    // enviroment maintains thread pools and other state info
    let mut env_ptr: *mut OrtEnv = std::ptr::null_mut();
    let status = unsafe {
        g_ort.as_ref().unwrap().CreateEnv.unwrap()(
            OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE,
            std::ffi::CString::new("test").unwrap().into_raw(),
            &mut env_ptr,
        )
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(env_ptr, std::ptr::null_mut());

    // initialize session options if needed
    let mut session_options_ptr: *mut OrtSessionOptions = std::ptr::null_mut();
    let status =
        unsafe { g_ort.as_ref().unwrap().CreateSessionOptions.unwrap()(&mut session_options_ptr) };
    CheckStatus(g_ort, status).unwrap();
    unsafe { g_ort.as_ref().unwrap().SetIntraOpNumThreads.unwrap()(session_options_ptr, 1) };
    assert_ne!(session_options_ptr, std::ptr::null_mut());

    // Sets graph optimization level
    unsafe {
        g_ort
            .as_ref()
            .unwrap()
            .SetSessionGraphOptimizationLevel
            .unwrap()(session_options_ptr, GraphOptimizationLevel_ORT_ENABLE_BASIC)
    };

    // Optionally add more execution providers via session_options
    // E.g. for CUDA include cuda_provider_factory.h and uncomment the following line:
    // OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

    //*************************************************************************
    // create session and load model into memory
    // using squeezenet version 1.3
    // https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.1-7.onnx
    const MODEL_PATH: &str = "squeezenet.onnx";

    let mut session_ptr: *mut OrtSession = std::ptr::null_mut();

    println!("Using Onnxruntime C API");
    let status = unsafe {
        g_ort.as_ref().unwrap().CreateSession.unwrap()(
            env_ptr,
            std::ffi::CString::new(MODEL_PATH).unwrap().into_raw(),
            session_options_ptr,
            &mut session_ptr,
        )
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(session_ptr, std::ptr::null_mut());

    //*************************************************************************
    // print model input layer (node names, types, shape etc.)
    // size_t num_input_nodes;
    let mut allocator_ptr: *mut OrtAllocator = std::ptr::null_mut();
    let status = unsafe {
        g_ort
            .as_ref()
            .unwrap()
            .GetAllocatorWithDefaultOptions
            .unwrap()(&mut allocator_ptr)
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(allocator_ptr, std::ptr::null_mut());

    // print number of model input nodes
    let mut num_input_nodes: u64 = 0;
    let status = unsafe {
        g_ort.as_ref().unwrap().SessionGetInputCount.unwrap()(session_ptr, &mut num_input_nodes)
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(num_input_nodes, 0);
    println!("Number of inputs = {:?}", num_input_nodes);
    let mut input_node_names: Vec<String> = Vec::new();
    let mut input_node_dims: Vec<i64> = Vec::new(); // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                                    // Otherwise need vector<vector<>>

    // iterate over all input nodes
    for i in 0..num_input_nodes {
        // print input node names
        let mut input_name: *mut i8 = std::ptr::null_mut();
        let status = unsafe {
            g_ort.as_ref().unwrap().SessionGetInputName.unwrap()(
                session_ptr,
                i,
                allocator_ptr,
                &mut input_name,
            )
        };
        CheckStatus(g_ort, status).unwrap();
        assert_ne!(input_name, std::ptr::null_mut());
        let input_name = char_p_to_string(input_name).unwrap();
        println!("Input {} : name={}", i, input_name);
        input_node_names.push(input_name);

        // print input node types
        let mut typeinfo_ptr: *mut OrtTypeInfo = std::ptr::null_mut();
        let status = unsafe {
            g_ort.as_ref().unwrap().SessionGetInputTypeInfo.unwrap()(
                session_ptr,
                i,
                &mut typeinfo_ptr,
            )
        };
        CheckStatus(g_ort, status).unwrap();
        assert_ne!(typeinfo_ptr, std::ptr::null_mut());

        let mut tensor_info_ptr: *const OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let status = unsafe {
            g_ort.as_ref().unwrap().CastTypeInfoToTensorInfo.unwrap()(
                typeinfo_ptr,
                &mut tensor_info_ptr,
            )
        };
        CheckStatus(g_ort, status).unwrap();
        assert_ne!(tensor_info_ptr, std::ptr::null_mut());

        let mut type_: ONNXTensorElementDataType = 0;
        let status = unsafe {
            g_ort.as_ref().unwrap().GetTensorElementType.unwrap()(tensor_info_ptr, &mut type_)
        };
        CheckStatus(g_ort, status).unwrap();
        assert_ne!(type_, 0);

        println!("Input {} : type={}", i, type_);

        // print input shapes/dims
        let mut num_dims = 0;
        let status = unsafe {
            g_ort.as_ref().unwrap().GetDimensionsCount.unwrap()(tensor_info_ptr, &mut num_dims)
        };
        CheckStatus(g_ort, status).unwrap();
        assert_ne!(num_dims, 0);

        println!("Input {} : num_dims={}", i, num_dims);
        input_node_dims.resize_with(num_dims as usize, Default::default);
        let status = unsafe {
            g_ort.as_ref().unwrap().GetDimensions.unwrap()(
                tensor_info_ptr,
                input_node_dims.as_mut_ptr(),
                num_dims,
            )
        };
        CheckStatus(g_ort, status).unwrap();

        for j in 0..num_dims {
            println!("Input {} : dim {}={}", i, j, input_node_dims[j as usize]);
        }

        unsafe { g_ort.as_ref().unwrap().ReleaseTypeInfo.unwrap()(typeinfo_ptr) };
    }

    // Results should be...
    // Number of inputs = 1
    // Input 0 : name = data_0
    // Input 0 : type = 1
    // Input 0 : num_dims = 4
    // Input 0 : dim 0 = 1
    // Input 0 : dim 1 = 3
    // Input 0 : dim 2 = 224
    // Input 0 : dim 3 = 224

    //*************************************************************************
    // Similar operations to get output node information.
    // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
    // OrtSessionGetOutputTypeInfo() as shown above.

    //*************************************************************************
    // Score the model using sample data, and inspect values

    let input_tensor_size = 224 * 224 * 3; // simplify ... using known dim values to calculate size
                                           // use OrtGetTensorShapeElementCount() to get official size!

    let output_node_names = &["softmaxout_1"];

    // initialize input data with values in [0.0, 1.0]
    let mut input_tensor_values: Vec<f64> = (0..input_tensor_size)
        .map(|i| (i as f64) / ((input_tensor_size + 1) as f64))
        .collect();

    // create input tensor object from data values
    let mut memory_info_ptr: *mut OrtMemoryInfo = std::ptr::null_mut();
    let status = unsafe {
        g_ort.as_ref().unwrap().CreateCpuMemoryInfo.unwrap()(
            OrtAllocatorType_OrtArenaAllocator,
            OrtMemType_OrtMemTypeDefault,
            &mut memory_info_ptr,
        )
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(memory_info_ptr, std::ptr::null_mut());

    /*
    // bindings.rs
    fn CreateTensorWithDataAsOrtValue(
        info: *const OrtMemoryInfo,
        p_data: *mut c_void,
        p_data_len: size_t,
        shape: *const i64,
        shape_len: size_t,
        type_: ONNXTensorElementDataType,
        out: *mut *mut OrtValue
    ) -> OrtStatusPtr

    // onnxruntime_c_api.h:448
    ORT_API2_STATUS(
        CreateTensorWithDataAsOrtValue,
        _In_ const OrtMemoryInfo* info,
        _Inout_ void* p_data,
        size_t p_data_len,
        _In_ const int64_t* shape,
        size_t shape_len,
        ONNXTensorElementDataType type,
        _Outptr_ OrtValue** out
    );

    // onnxruntime_c_api.cc:195
    ORT_API_STATUS_IMPL(
        OrtApis::CreateTensorWithDataAsOrtValue,
        _In_ const OrtMemoryInfo* info,
        _Inout_ void* p_data,
        size_t p_data_len,
        _In_ const int64_t* shape,
        size_t shape_len,
        ONNXTensorElementDataType type,
        _Outptr_ OrtValue** out
    ) { ... }

    // c_api_sample.cpp
    std::vector<float> input_tensor_values(input_tensor_size);
    OrtValue *input_tensor = NULL;
    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        input_tensor_values.data(),
        input_tensor_size * sizeof(float),
        input_node_dims.data(),
        4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor
    );
    CheckStatus(status);
    */

    // FIXME: Check me!
    let mut input_tensor_ptr: *mut OrtValue = std::ptr::null_mut();
    let input_tensor_ptr_ptr: *mut *mut OrtValue = &mut input_tensor_ptr;
    let input_tensor_values_ptr: *mut std::ffi::c_void =
        input_tensor_values.as_mut_ptr() as *mut std::ffi::c_void;
    assert_ne!(input_tensor_values_ptr, std::ptr::null_mut());

    println!("input_node_dims: {:?}", input_node_dims);
    let shape: *const i64 = input_node_dims.as_ptr();
    assert_ne!(shape, std::ptr::null_mut());

    let status = unsafe {
        g_ort
            .as_ref()
            .unwrap()
            .CreateTensorWithDataAsOrtValue
            .unwrap()(
            memory_info_ptr,                                               // info
            input_tensor_values_ptr,                                       // p_data
            (input_tensor_size * std::mem::size_of::<f64>()) as u64,       // p_data_len
            shape,                                                         // shape
            4,                                                             // shape_len
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, // type_
            input_tensor_ptr_ptr,                                          // out
        )
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(input_tensor_ptr, std::ptr::null_mut());

    let mut is_tensor = 0;
    let status =
        unsafe { g_ort.as_ref().unwrap().IsTensor.unwrap()(input_tensor_ptr, &mut is_tensor) };
    CheckStatus(g_ort, status).unwrap();
    assert_eq!(is_tensor, 1);

    let input_tensor_ptr2: *const OrtValue = input_tensor_ptr as *const OrtValue;
    let input_tensor_ptr3: *const *const OrtValue = &input_tensor_ptr2;

    unsafe { g_ort.as_ref().unwrap().ReleaseMemoryInfo.unwrap()(memory_info_ptr) };

    // score model & input tensor, get back output tensor

    // FIXME: This leaks!
    let input_node_names_cstring: Vec<std::ffi::CString> = input_node_names
        .into_iter()
        .map(|n| std::ffi::CString::new(n).unwrap())
        .collect();
    let input_node_names_ptr: Vec<*const i8> = input_node_names_cstring
        .into_iter()
        .map(|n| n.into_raw() as *const i8)
        .collect();
    let input_node_names_ptr_ptr: *const *const i8 = input_node_names_ptr.as_ptr();

    // FIXME: This leaks!
    let output_node_names_cstring: Vec<std::ffi::CString> = output_node_names
        .into_iter()
        .map(|n| std::ffi::CString::new(n.clone()).unwrap())
        .collect();
    let output_node_names_ptr: Vec<*const i8> = output_node_names_cstring
        .into_iter()
        .map(|n| n.into_raw() as *const i8)
        .collect();
    let output_node_names_ptr_ptr: *const *const i8 = output_node_names_ptr.as_ptr();

    let input_node_names_cstring =
        unsafe { std::ffi::CString::from_raw(input_node_names_ptr[0] as *mut i8) };
    println!("input_node_names[0]:  {:?}", input_node_names_cstring);
    std::mem::forget(input_node_names_cstring);
    let output_node_names_cstring =
        unsafe { std::ffi::CString::from_raw(output_node_names_ptr[0] as *mut i8) };
    println!("output_node_names[0]: {:?}", output_node_names_cstring);
    std::mem::forget(output_node_names_cstring);
    let run_options_ptr: *const OrtRunOptions = std::ptr::null();
    let mut output_tensor_ptr: *mut OrtValue = std::ptr::null_mut();
    let mut output_tensor_ptr_ptr: *mut *mut OrtValue = &mut output_tensor_ptr;

    /*
    // bindings.rs
    fn Run(
        sess: *mut OrtSession,
        run_options: *const OrtRunOptions,
        input_names: *const *const ::std::os::raw::c_char,
        input: *const *const OrtValue,
        input_len: size_t,
        output_names1: *const *const ::std::os::raw::c_char,
        output_names_len: size_t,
        output: *mut *mut OrtValue,
    ) -> OrtStatusPtr

    // onnxruntime_c_api.h
    ORT_API2_STATUS(
        Run,
        _Inout_ OrtSession* sess,
        _In_opt_ const OrtRunOptions* run_options,
        _In_reads_(input_len) const char* const* input_names,
        _In_reads_(input_len) const OrtValue* const* input,
        size_t input_len,
        _In_reads_(output_names_len) const char* const* output_names1,
        size_t output_names_len,
        _Inout_updates_all_(output_names_len) OrtValue** output
    );

    // onnxruntime_c_api.cc
    ORT_API_STATUS_IMPL(
        OrtApis::Run,
        _Inout_ OrtSession* sess,
        _In_opt_ const OrtRunOptions* run_options,
        _In_reads_(input_len) const char* const* input_names,
        _In_reads_(input_len) const OrtValue* const* input,
        size_t input_len,
        _In_reads_(output_names_len) const char* const* output_names1,
        size_t output_names_len,
        _Inout_updates_all_(output_names_len) OrtValue** output
    ) { ... }
    */

    let status = unsafe {
        g_ort.as_ref().unwrap().Run.unwrap()(
            session_ptr,               // Ok!
            run_options_ptr,           // Ok!
            input_node_names_ptr_ptr,  // Looks fine
            input_tensor_ptr3,         // ?
            1,                         // Ok!
            output_node_names_ptr_ptr, // Looks fine
            1,                         // Ok!
            output_tensor_ptr_ptr,     // Ok!
        )
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(output_tensor_ptr, std::ptr::null_mut());
    // println!("input_tensor: {:?}", input_tensor);

    let mut is_tensor = 0;
    let status =
        unsafe { g_ort.as_ref().unwrap().IsTensor.unwrap()(output_tensor_ptr, &mut is_tensor) };
    CheckStatus(g_ort, status).unwrap();
    assert_eq!(is_tensor, 1);

    // Get pointer to output tensor float values
    // float *floatarr;
    // CheckStatus(g_ort->GetTensorMutableData(output_tensor, (void**)&floatarr));
    // fn GetTensorMutableData(
    //     value: *mut OrtValue,
    //     out: *mut *mut ::std::os::raw::c_void,
    // ) -> OrtStatusPtr
    let mut floatarr: *mut f64 = std::ptr::null_mut();
    let floatarr_ptr: *mut *mut f64 = &mut floatarr;
    let floatarr_ptr_void: *mut *mut std::ffi::c_void = floatarr_ptr as *mut *mut std::ffi::c_void;
    let status = unsafe {
        g_ort.as_ref().unwrap().GetTensorMutableData.unwrap()(output_tensor_ptr, floatarr_ptr_void)
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(floatarr, std::ptr::null_mut());

    assert!((unsafe { *floatarr } - 0.000045).abs() < 1e-6);

    // score the model, and print scores for first 5 classes
    let floatarr_vec: Vec<f64> = unsafe { Vec::from_raw_parts(floatarr, 5, 5) };
    for i in 0..5 {
        println!("Score for class [{}] =  {}", i, floatarr_vec[i]);
    }

    // Results should be as below...
    // Score for class[0] = 0.000045
    // Score for class[1] = 0.003846
    // Score for class[2] = 0.000125
    // Score for class[3] = 0.001180
    // Score for class[4] = 0.001317

    unsafe { g_ort.as_ref().unwrap().ReleaseValue.unwrap()(output_tensor_ptr) };
    unsafe { g_ort.as_ref().unwrap().ReleaseValue.unwrap()(input_tensor_ptr) };
    unsafe { g_ort.as_ref().unwrap().ReleaseSession.unwrap()(session_ptr) };
    unsafe { g_ort.as_ref().unwrap().ReleaseSessionOptions.unwrap()(session_options_ptr) };
    unsafe { g_ort.as_ref().unwrap().ReleaseEnv.unwrap()(env_ptr) };

    println!("Done!");
}

fn CheckStatus(g_ort: *const OrtApi, status: *const OrtStatus) -> Result<(), String> {
    if status != std::ptr::null() {
        let raw = unsafe { g_ort.as_ref().unwrap().GetErrorMessage.unwrap()(status) };
        Err(char_p_to_string(raw).unwrap())
    } else {
        Ok(())
    }
}

fn char_p_to_string(raw: *const i8) -> Result<String, std::ffi::IntoStringError> {
    let c_string = unsafe { std::ffi::CString::from_raw(raw as *mut i8) };
    c_string.into_string()
}
