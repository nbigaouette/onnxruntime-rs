use std::error::Error;

use ndarray;
use onnxruntime::tensor::{OrtOwnedTensor, TensorElementDataType};
use onnxruntime::{environment::Environment, tensor::DynOrtTensor, LoggingLevel};

#[test]
fn run_model_with_string_input_output() -> Result<(), Box<dyn Error>> {
    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;

    let mut session = environment
        .new_session_builder()?
        .with_model_from_file("../test-models/tensorflow/unique_model.onnx")?;

    // Inputs:
    //   0:
    //     name = input_1:0
    //     type = String
    //     dimensions = [None]
    // Outputs:
    //   0:
    //     name = Identity:0
    //     type = Int32
    //     dimensions = [None]
    //   1:
    //     name = Identity_1:0
    //     type = String
    //     dimensions = [None]

    let array = ndarray::Array::from(vec!["foo", "bar", "foo", "foo"]);
    let input_tensor_values = vec![array];

    let outputs: Vec<DynOrtTensor<_>> = session.run(input_tensor_values)?;

    assert_eq!(TensorElementDataType::Int32, outputs[0].data_type());
    assert_eq!(TensorElementDataType::String, outputs[1].data_type());

    let int_output: OrtOwnedTensor<i32, _> = outputs[0].try_extract()?;

    assert_eq!(&[0, 1, 0, 0], int_output.as_slice().unwrap());

    // TODO get the string output once string extraction is implemented

    Ok(())
}
