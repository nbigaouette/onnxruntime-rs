//! Module defining machine comprehension models available to download.
//!
//! See [https://github.com/onnx/models#machine_comprehension](https://github.com/onnx/models#machine_comprehension)

use crate::download::{language::Language, AvailableOnnxModel, ModelUrl};

/// Machine Comprehension
///
/// > This subset of natural language processing models that answer questions about a given context paragraph.
///
/// Source: [https://github.com/onnx/models#machine_comprehension](https://github.com/onnx/models#machine_comprehension)
#[derive(Debug, Clone)]
pub enum MachineComprehension {
    /// Answers a query about a given context paragraph.
    ///
    /// > This model is a neural network for answering a query about a given context paragraph.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/text/machine_comprehension/bidirectional_attention_flow](https://github.com/onnx/models/tree/master/text/machine_comprehension/bidirectional_attention_flow)
    ///
    /// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
    BiDAF,
    /// Answers questions based on the context of the given input paragraph.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/text/machine_comprehension/bert-squad](https://github.com/onnx/models/tree/master/text/machine_comprehension/bert-squad)
    ///
    /// Variant downloaded: ONNX Version 1.5 with Opset Version 10.
    BERTSquad,
}

impl ModelUrl for MachineComprehension {
    fn fetch_url(&self) -> &'static str {
        match self {
            MachineComprehension::BiDAF => "https://github.com/onnx/models/raw/master/text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.onnx",
            MachineComprehension::BERTSquad => "https://github.com/onnx/models/blob/master/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx",
        }
    }
}

impl From<MachineComprehension> for AvailableOnnxModel {
    fn from(model: MachineComprehension) -> Self {
        AvailableOnnxModel::Language(Language::MachineComprehension(model))
    }
}
