use onnxruntime::{EnvBuilder, GraphOptimizationLevel, LoggingLevel};

type Error = Box<dyn std::error::Error>;

fn main() -> Result<(), Error> {
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
    let n = session.inputs[0].dimensions.iter().product();
    let input_tensor_values: Vec<Vec<f32>> =
        vec![(0..n).map(|i| (i as f32) / ((n + 1) as f32)).collect()];

    let outputs = session.run(input_tensor_values)?;

    for i in 0..5 {
        println!("Score for class [{}] =  {}", i, outputs[0][i]);
    }

    // FIXME: Use a newtype with custom Drop impl to forget
    outputs
        .into_iter()
        .for_each(|output| std::mem::forget(output));

    Ok(())
}
