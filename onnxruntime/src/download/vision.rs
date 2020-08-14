//! Module defining computer vision models available to download.

use super::ModelUrl;

/// Computer vision model
#[derive(Debug, Clone)]
pub enum Vision {
    /// Image classification model
    ImageClassification(ImageClassificationModel),
}
/// Image classification model
///
/// > This collection of models take images as input, then classifies the major objects in the images
/// > into 1000 object categories such as keyboard, mouse, pencil, and many animals.
///
/// Source: [https://github.com/onnx/models#image-classification-](https://github.com/onnx/models#image-classification-)
#[derive(Debug, Clone)]
pub enum ImageClassificationModel {
    /// Handwritten digits prediction using CNN
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/mnist](https://github.com/onnx/models/tree/master/vision/classification/mnist)
    ///
    /// Variant downloaded: ONNX Version 1.3 with Opset Version 8.
    Mnist,
    /// Image classification aimed for mobile targets.
    ///
    /// > MobileNet models perform image classification - they take images as input and classify the major
    /// > object in the image into a set of pre-defined classes. They are trained on ImageNet dataset which
    /// > contains images from 1000 classes. MobileNet models are also very efficient in terms of speed and
    /// > size and hence are ideal for embedded and mobile applications.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/mobilenet](https://github.com/onnx/models/tree/master/vision/classification/mobilenet)
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    MobileNet,
    /// A small CNN with AlexNet level accuracy on ImageNet with 50x fewer parameters.
    ///
    /// > SqueezeNet is a small CNN which achieves AlexNet level accuracy on ImageNet with 50x fewer parameters.
    /// > SqueezeNet requires less communication across servers during distributed training, less bandwidth to
    /// > export a new model from the cloud to an autonomous car and more feasible to deploy on FPGAs and other
    /// > hardware with limited memory.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/squeezenet](https://github.com/onnx/models/tree/master/vision/classification/squeezenet)
    ///
    /// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
    SqueezeNet,
    /// Google's Inception
    Inception(InceptionVersion),
}

/// Google's Inception
#[derive(Debug, Clone)]
pub enum InceptionVersion {
    /// Google's Inception v1
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v1](https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v1)
    ///
    /// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
    V1,
    /// Google's Inception v2
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v2](https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v2)
    ///
    /// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
    V2,
}

impl ModelUrl for Vision {
    fn fetch_url(&self) -> &'static str {
        match self {
            Vision::ImageClassification(ic) => ic.fetch_url(),
        }
    }
}

impl ModelUrl for ImageClassificationModel {
    fn fetch_url(&self) -> &'static str {
        match self {
            ImageClassificationModel::Mnist => "https://github.com/onnx/models/raw/master/vision/classification/mnist/model/mnist-8.onnx",
            ImageClassificationModel::MobileNet => "https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
            ImageClassificationModel::SqueezeNet => "https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.0-9.onnx",
            ImageClassificationModel::Inception(version) => version.fetch_url(),
        }
    }
}

impl ModelUrl for InceptionVersion {
    fn fetch_url(&self) -> &'static str {
        match self {
            InceptionVersion::V1 => "https://github.com/onnx/models/raw/master/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-9.onnx",
            InceptionVersion::V2 => "https://github.com/onnx/models/raw/master/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.onnx",
        }
    }
}
