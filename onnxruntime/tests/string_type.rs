use std::error::Error;

use ndarray;
use onnxruntime::tensor::{OrtOwnedTensor, TensorElementDataType};
use onnxruntime::{environment::Environment, tensor::DynOrtTensor, LoggingLevel};

#[test]
fn run_model_with_string_1d_input_output() -> Result<(), Box<dyn Error>> {
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

    let array = ndarray::Array::from(vec!["foo", "bar", "foo", "foo", "baz"]);
    let input_tensor_values = vec![array];

    let outputs: Vec<DynOrtTensor<_>> = session.run(input_tensor_values)?;

    assert_eq!(TensorElementDataType::Int32, outputs[0].data_type());
    assert_eq!(TensorElementDataType::String, outputs[1].data_type());

    let int_output: OrtOwnedTensor<i32, _> = outputs[0].try_extract()?;
    let string_output: OrtOwnedTensor<String, _> = outputs[1].try_extract()?;

    assert_eq!(&[5], int_output.view().shape());
    assert_eq!(&[3], string_output.view().shape());

    assert_eq!(&[0, 1, 0, 0, 2], int_output.view().as_slice().unwrap());
    assert_eq!(
        vec!["foo", "bar", "baz"]
            .into_iter()
            .map(|s| s.to_owned())
            .collect::<Vec<_>>(),
        string_output.view().as_slice().unwrap()
    );

    Ok(())
}
