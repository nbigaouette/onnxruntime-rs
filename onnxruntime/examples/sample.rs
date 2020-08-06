use onnxruntime::{EnvBuilder, GraphOptimizationLevel, LoggingLevel};

type Error = Box<dyn std::error::Error>;

fn main() -> Result<(), Error> {
    let env = EnvBuilder::new()
        .with_name("test")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;

    let mut session = env
        .load_model("squeezenet.onnx")
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .with_number_threads(1)
        .build()?;

    let inputs = session.read_inputs()?;
    println!("inputs: {:#?}", inputs);

    let input_tensor_size: u32 = inputs[0].dimensions.iter().product();

    let output_node_names = &["softmaxout_1"];

    // initialize input data with values in [0.0, 1.0]
    let mut data_1d: Vec<f32> = (0..input_tensor_size)
        .map(|i| (i as f32) / ((input_tensor_size + 1) as f32))
        .collect();

    session.set_inputs(data_1d, &inputs[0].dimensions)?;

    Ok(())
}
