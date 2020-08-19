//! Module defining computer vision models available to download.

use super::ModelUrl;

pub mod body_face_gesture_analysis;
pub mod domain_based_image_classification;
pub mod image_classification;
pub mod object_detection_image_segmentation;

use body_face_gesture_analysis::BodyFaceGestureAnalysis;
use domain_based_image_classification::DomainBasedImageClassification;
use image_classification::ImageClassificationModel;
use object_detection_image_segmentation::ObjectDetectionImageSegmentation;

/// Computer vision model
#[derive(Debug, Clone)]
pub enum Vision {
    /// Domain-based Image Classification
    DomainBasedImageClassification(DomainBasedImageClassification),
    /// Image classification model
    ImageClassification(ImageClassificationModel),
    /// Object Detection & Image Segmentation
    ObjectDetectionImageSegmentation(ObjectDetectionImageSegmentation),
    /// Body, Face & Gesture Analysis
    BodyFaceGestureAnalysis(BodyFaceGestureAnalysis),
    /// Image Manipulation
    ImageManipulation(ImageManipulation),
}

/// Image Manipulation
///
/// > Image manipulation models use neural networks to transform input images to modified output images. Some
/// > popular models in this category involve style transfer or enhancing images by increasing resolution.
///
/// Source: [https://github.com/onnx/models#image_manipulation](https://github.com/onnx/models#image_manipulation)
#[derive(Debug, Clone)]
pub enum ImageManipulation {
    /// Super Resolution
    ///
    /// > The Super Resolution machine learning model sharpens and upscales the input image to refine the
    /// > details and improve quality.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/super_resolution/sub_pixel_cnn_2016](https://github.com/onnx/models/tree/master/vision/super_resolution/sub_pixel_cnn_2016)
    ///
    /// Variant downloaded: ONNX Version 1.5 with Opset Version 10.
    SuperResolution,
    /// Fast Neural Style Transfer
    ///
    /// > This artistic style transfer model mixes the content of an image with the style of another image.
    /// > Examples of the styles can be seen
    /// > [in this PyTorch example](https://github.com/pytorch/examples/tree/master/fast_neural_style#models).
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/style_transfer/fast_neural_style](https://github.com/onnx/models/tree/master/vision/style_transfer/fast_neural_style)
    FastNeuralStyleTransfer(FastNeuralStyleTransferStyle),
}

/// Fast Neural Style Transfer Style
///
/// Source: [https://github.com/onnx/models/tree/master/vision/style_transfer/fast_neural_style](https://github.com/onnx/models/tree/master/vision/style_transfer/fast_neural_style)
///
/// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
#[derive(Debug, Clone)]
pub enum FastNeuralStyleTransferStyle {
    /// Mosaic style
    Mosaic,
    /// Candy style
    Candy,
    /// RainPrincess style
    RainPrincess,
    /// Udnie style
    Udnie,
    /// Pointilism style
    Pointilism,
}

impl ModelUrl for Vision {
    fn fetch_url(&self) -> &'static str {
        match self {
            Vision::DomainBasedImageClassification(variant) => variant.fetch_url(),
            Vision::ImageClassification(variant) => variant.fetch_url(),
            Vision::ObjectDetectionImageSegmentation(variant) => variant.fetch_url(),
            Vision::BodyFaceGestureAnalysis(variant) => variant.fetch_url(),
            Vision::ImageManipulation(variant) => variant.fetch_url(),
        }
    }
}

impl ModelUrl for ImageManipulation {
    fn fetch_url(&self) -> &'static str {
        match self {
            ImageManipulation::SuperResolution => "https://github.com/onnx/models/raw/master/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx",
            ImageManipulation::FastNeuralStyleTransfer(style) => style.fetch_url(),
        }
    }
}

impl ModelUrl for FastNeuralStyleTransferStyle {
    fn fetch_url(&self) -> &'static str {
        match self {
            FastNeuralStyleTransferStyle::Mosaic => "https://github.com/onnx/models/raw/master/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx",
            FastNeuralStyleTransferStyle::Candy => "https://github.com/onnx/models/raw/master/vision/style_transfer/fast_neural_style/model/candy-9.onnx",
            FastNeuralStyleTransferStyle::RainPrincess => "https://github.com/onnx/models/raw/master/vision/style_transfer/fast_neural_style/model/rain-princess-9.onnx",
            FastNeuralStyleTransferStyle::Udnie => "https://github.com/onnx/models/raw/master/vision/style_transfer/fast_neural_style/model/udnie-9.onnx",
            FastNeuralStyleTransferStyle::Pointilism => "https://github.com/onnx/models/raw/master/vision/style_transfer/fast_neural_style/model/pointilism-9.onnx",
        }
    }
}
