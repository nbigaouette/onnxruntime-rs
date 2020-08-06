use onnxruntime::*;

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

    let env2 = EnvBuilder::new()
        .with_name("test2")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;

    let session2 = env
        .load_model("squeezenet.onnx")
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .with_number_threads(1)
        .build()?;

    Ok(())
}
