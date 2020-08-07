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
        .load_model_from_file("squeezenet.onnx");

    Ok(())
}
