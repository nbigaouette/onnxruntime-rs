#![forbid(unsafe_code)]

use ndarray::Array;

use onnxruntime::{
    download::vision::ImageClassificationModel, environment::Environment, GraphOptimizationLevel,
    LoggingLevel,
};

type Error = Box<dyn std::error::Error>;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Error> {
    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;

    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(1)?
        // .load_model_from_file("squeezenet.onnx")?;
        // .with_downloaded_model(ImageClassificationModel::MobileNet)?;
        .with_downloaded_model(ImageClassificationModel::SqueezeNet)?;

    let input0_shape: Vec<usize> = session.inputs[0].dimensions().collect();
    let output0_shape: Vec<usize> = session.outputs[0].dimensions().collect();

    assert_eq!(input0_shape, [1, 3, 224, 224]);
    assert_eq!(output0_shape, [1, 1000, 1, 1]);

    // initialize input data with values in [0.0, 1.0]
    let n: u32 = session.inputs[0].dimensions.iter().product();
    let array = Array::linspace(0.0_f32, 1.0, n as usize)
        .into_shape(input0_shape)
        .unwrap();
    let input_tensor_values = vec![array];

    let outputs = session.run(input_tensor_values)?;

    assert_eq!(outputs[0].shape(), output0_shape.as_slice());
    for i in 0..5 {
        println!("Score for class [{}] =  {}", i, outputs[0][[0, i, 0, 0]]);
    }

    Ok(())
}
