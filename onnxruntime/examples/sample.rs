use ndarray::Array;

use onnxruntime::{EnvBuilder, GraphOptimizationLevel, LoggingLevel};

type Error = Box<dyn std::error::Error>;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Error> {
    let env = EnvBuilder::new()
        .with_name("test")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;

    let mut session = env
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(1)?
        .load_model_from_file("squeezenet.onnx")?;

    // initialize input data with values in [0.0, 1.0]
    let n: u32 = session.inputs[0].dimensions.iter().product();
    let array = Array::linspace(0.0_f32, 1.0, n as usize)
        .into_shape((1, 3, 224, 224))
        .unwrap();
    let input_tensor_values = vec![array];

    let outputs = session.run(input_tensor_values)?;

    assert_eq!(outputs[0].shape(), [1, 1000, 1, 1]);
    for i in 0..5 {
        println!("Score for class [{}] =  {}", i, outputs[0][[0, i, 0, 0]]);
    }

    // FIXME: Use a newtype with custom Drop impl to forget
    outputs
        .into_iter()
        .for_each(|output| std::mem::forget(output));

    Ok(())
}
