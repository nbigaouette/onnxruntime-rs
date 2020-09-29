//! Example reproducing issue #22.
//!
//! `model.onnx` available to download here:
//! https://drive.google.com/file/d/1FmL-Wpm06V-8wgRqvV3Skey_X98Ue4D_/view?usp=sharing

use ndarray::Array2;
use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor, GraphOptimizationLevel};

fn main() {
    let env = Environment::builder().with_name("env").build().unwrap();
    let mut session = env
        .new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_model_from_file("model.onnx")
        .unwrap();

    println!("{:#?}", session.inputs);
    println!("{:#?}", session.outputs);

    let input_ids = Array2::<f32>::from_shape_vec((1, 3), vec![1f32, 2f32, 3f32]).unwrap();
    let attention_mask = Array2::<f32>::from_shape_vec((1, 3), vec![1f32, 1f32, 1f32]).unwrap();

    let outputs: Vec<OrtOwnedTensor<f32, _>> =
        session.run(vec![input_ids, attention_mask]).unwrap();
    print!("outputs: {:#?}", outputs);
}
