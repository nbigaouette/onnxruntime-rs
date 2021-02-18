use std::error::Error;

use ndarray;
use onnxruntime::tensor::{DynOrtTensor, OrtOwnedTensor};
use onnxruntime::{environment::Environment, LoggingLevel};

#[test]
fn run_model_with_ort_customops() -> Result<(), Box<dyn Error>> {
    let lib_path = match std::env::var("ONNXRUNTIME_RS_TEST_ORT_CUSTOMOPS_LIB") {
        Ok(s) => s,
        Err(_e) => {
            println!("Skipping ort_customops test -- no lib specified");
            return Ok(());
        }
    };

    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;

    let mut session = environment
        .new_session_builder()?
        .with_custom_op_lib(&lib_path)?
        .with_model_from_file("../test-models/tensorflow/regex_model.onnx")?;

    //Inputs:
    //   0:
    //     name = input_1:0
    //     type = String
    //     dimensions = [None]
    // Outputs:
    //   0:
    //     name = Identity:0
    //     type = String
    //     dimensions = [None]

    let array = ndarray::Array::from(vec![String::from("Hello world!")]);
    let input_tensor_values = vec![array];

    let outputs: Vec<DynOrtTensor<_>> = session.run(input_tensor_values)?;
    let strings: OrtOwnedTensor<String, _> = outputs[0].try_extract()?;

    // ' ' replaced with '_'
    assert_eq!(
        &[String::from("Hello_world!")],
        strings.view().as_slice().unwrap()
    );

    Ok(())
}
