use onnxruntime::{EnvBuilder, GraphOptimizationLevel, LoggingLevel};

type Error = Box<dyn std::error::Error>;

fn main() -> Result<(), Error> {
    let env = EnvBuilder::new()
        .with_name("test")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;

    let session = env
        .load_model("squeezenet.onnx")
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .with_number_threads(1)
        .build()?;

    let inputs = session.read_inputs()?;
    println!("inputs: {:#?}", inputs);

    Ok(())
}
