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
    /// Large transformer-based model that predicts sentiment based on given input text.
    ///
    /// > Transformer-based language model for text generation.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/text/machine_comprehension/roberta](https://github.com/onnx/models/tree/master/text/machine_comprehension/roberta)
    RoBERTa(RoBERTa),
}

/// Large transformer-based model that predicts sentiment based on given input text.
///
/// > Transformer-based language model for text generation.
///
/// Source: [https://github.com/onnx/models/tree/master/text/machine_comprehension/roberta](https://github.com/onnx/models/tree/master/text/machine_comprehension/roberta)
#[derive(Debug, Clone)]
pub enum RoBERTa {
    /// Variant with input is a sequence of words as a string. Example: "Text to encode: Hello, World"
    ///
    /// Variant downloaded: ONNX Version 1.6 with Opset Version 11.
    RoBERTaBase,
    /// Variant with input is a sequence of words as a string including sentiment. Example: "This film is so good"
    ///
    /// Variant downloaded: ONNX Version 1.6 with Opset Version 9.
    RoBERTaSequenceClassification,
}

impl ModelUrl for MachineComprehension {
    fn fetch_url(&self) -> &'static str {
        match self {
            MachineComprehension::BiDAF => "https://github.com/onnx/models/raw/master/text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.onnx",
            MachineComprehension::BERTSquad => "https://github.com/onnx/models/raw/master/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx",
            MachineComprehension::RoBERTa(variant) => variant.fetch_url(),
        }
    }
}

impl ModelUrl for RoBERTa {
    fn fetch_url(&self) -> &'static str {
        match self {
            RoBERTa::RoBERTaBase => "https://github.com/onnx/models/raw/master/text/machine_comprehension/roberta/model/roberta-base-11.onnx",
            RoBERTa::RoBERTaSequenceClassification => "https://github.com/onnx/models/raw/master/text/machine_comprehension/roberta/model/roberta-sequence-classification-9.onnx",
        }
    }
}

impl From<MachineComprehension> for AvailableOnnxModel {
    fn from(model: MachineComprehension) -> Self {
        AvailableOnnxModel::Language(Language::MachineComprehension(model))
    }
}
