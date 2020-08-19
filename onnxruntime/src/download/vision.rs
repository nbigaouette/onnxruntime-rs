//! Module defining computer vision models available to download.

use super::{AvailableOnnxModel, ModelUrl};

pub mod domain_based_image_classification;
pub mod image_classification;

use domain_based_image_classification::DomainBasedImageClassification;
use image_classification::ImageClassificationModel;

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

/// Object Detection & Image Segmentation
///
/// > Object detection models detect the presence of multiple objects in an image and segment out areas of the
/// > image where the objects are detected. Semantic segmentation models partition an input image by labeling each pixel
/// > into a set of pre-defined categories.
///
/// Source: [https://github.com/onnx/models#object_detection](https://github.com/onnx/models#object_detection)
#[derive(Debug, Clone)]
pub enum ObjectDetectionImageSegmentation {
    /// A real-time CNN for object detection that detects 20 different classes. A smaller version of the
    /// more complex full YOLOv2 network.
    ///
    /// Variant downloaded: ONNX Version 1.3 with Opset Version 8.
    TinyYoloV2,
    /// Single Stage Detector: real-time CNN for object detection that detects 80 different classes.
    ///
    /// Variant downloaded: ONNX Version 1.5 with Opset Version 10.
    Ssd,
    /// A variant of MobileNet that uses the Single Shot Detector (SSD) model framework. The model detects 80
    /// different object classes and locates up to 10 objects in an image.
    ///
    /// Variant downloaded: ONNX Version 1.7.0 with Opset Version 10.
    SSDMobileNetV1,
    /// Increases efficiency from R-CNN by connecting a RPN with a CNN to create a single, unified network for
    /// object detection that detects 80 different classes.
    ///
    /// Variant downloaded: ONNX Version 1.5 with Opset Version 10.
    FasterRcnn,
    /// A real-time neural network for object instance segmentation that detects 80 different classes. Extends
    /// Faster R-CNN as each of the 300 elected ROIs go through 3 parallel branches of the network: label
    /// prediction, bounding box prediction and mask prediction.
    ///
    /// Variant downloaded: ONNX Version 1.5 with Opset Version 10.
    MaskRcnn,
    /// A real-time dense detector network for object detection that addresses class imbalance through Focal Loss.
    /// RetinaNet is able to match the speed of previous one-stage detectors and defines the state-of-the-art in
    /// two-stage detectors (surpassing R-CNN).
    ///
    /// Variant downloaded: ONNX Version 1.6.0 with Opset Version 9.
    RetinaNet,
    /// A CNN model for real-time object detection system that can detect over 9000 object categories. It uses a
    /// single network evaluation, enabling it to be more than 1000x faster than R-CNN and 100x faster than
    /// Faster R-CNN.
    ///
    /// Variant downloaded: ONNX Version 1.3 with Opset Version 8.
    YoloV2,
    /// A CNN model for real-time object detection system that can detect over 9000 object categories. It uses
    /// a single network evaluation, enabling it to be more than 1000x faster than R-CNN and 100x faster than
    /// Faster R-CNN. This model is trained with COCO dataset and contains 80 classes.
    ///
    /// Variant downloaded: ONNX Version 1.5 with Opset Version 9.
    YoloV2Coco,
    /// A deep CNN model for real-time object detection that detects 80 different classes. A little bigger than
    /// YOLOv2 but still very fast. As accurate as SSD but 3 times faster.
    ///
    /// Variant downloaded: ONNX Version 1.5 with Opset Version 10.
    YoloV3,
    /// A smaller version of YOLOv3 model.
    ///
    /// Variant downloaded: ONNX Version 1.6 with Opset Version 11.
    TinyYoloV3,
    /// Optimizes the speed and accuracy of object detection. Two times faster than EfficientDet. It improves
    /// YOLOv3's AP and FPS by 10% and 12%, respectively, with mAP50 of 52.32 on the COCO 2017 dataset and
    /// FPS of 41.7 on Tesla 100.
    ///
    /// Variant downloaded: ONNX Version 1.6 with Opset Version 11.
    YoloV4,
    /// Deep CNN based pixel-wise semantic segmentation model with >80% mIOU (mean Intersection Over Union).
    /// Trained on cityscapes dataset, which can be effectively implemented in self driving vehicle systems.
    ///
    /// Variant downloaded: ONNX Version 1.2.2 with Opset Version 7.
    Duc,
}

/// Body, Face & Gesture Analysis
///
/// > Face detection models identify and/or recognize human faces and emotions in given images. Body and Gesture
/// > Analysis models identify gender and age in given image.
///
/// Source: [https://github.com/onnx/models#body_analysis](https://github.com/onnx/models#body_analysis)
#[derive(Debug, Clone)]
pub enum BodyFaceGestureAnalysis {
    /// A CNN based model for face recognition which learns discriminative features of faces and produces
    /// embeddings for input face images.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/body_analysis/arcface](https://github.com/onnx/models/tree/master/vision/body_analysis/arcface)
    ///
    /// Variant downloaded: ONNX Version 1.3 with Opset Version 8.
    ArcFace,
    /// Deep CNN for emotion recognition trained on images of faces.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus](https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus)
    ///
    /// Variant downloaded: ONNX Version 1.3 with Opset Version 8.
    EmotionFerPlus,
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
            Vision::DomainBasedImageClassification(dbic) => dbic.fetch_url(),
            Vision::ImageClassification(ic) => ic.fetch_url(),
            Vision::ObjectDetectionImageSegmentation(odis) => odis.fetch_url(),
            Vision::BodyFaceGestureAnalysis(bfga) => bfga.fetch_url(),
            Vision::ImageManipulation(im) => im.fetch_url(),
        }
    }
}

impl ModelUrl for ObjectDetectionImageSegmentation {
    fn fetch_url(&self) -> &'static str {
        match self {
            ObjectDetectionImageSegmentation::TinyYoloV2 => "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx",
            ObjectDetectionImageSegmentation::Ssd => "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/ssd/model/ssd-10.onnx",
            ObjectDetectionImageSegmentation::SSDMobileNetV1 => "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx",
            ObjectDetectionImageSegmentation::FasterRcnn => "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx",
            ObjectDetectionImageSegmentation::MaskRcnn => "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.onnx",
            ObjectDetectionImageSegmentation::RetinaNet => "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/retinanet/model/retinanet-9.onnx",
            ObjectDetectionImageSegmentation::YoloV2 => "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/yolov2/model/yolov2-voc-8.onnx",
            ObjectDetectionImageSegmentation::YoloV2Coco => "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/yolov2-coco/model/yolov2-coco-9.onnx",
            ObjectDetectionImageSegmentation::YoloV3 => "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx",
            ObjectDetectionImageSegmentation::TinyYoloV3 => "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx",
            ObjectDetectionImageSegmentation::YoloV4 => "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/yolov4/model/yolov4.onnx",
            ObjectDetectionImageSegmentation::Duc => "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/duc/model/ResNet101-DUC-7.onnx",
        }
    }
}

impl ModelUrl for BodyFaceGestureAnalysis {
    fn fetch_url(&self) -> &'static str {
        match self {
            BodyFaceGestureAnalysis::ArcFace => "https://github.com/onnx/models/raw/master/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
            BodyFaceGestureAnalysis::EmotionFerPlus => "https://github.com/onnx/models/raw/master/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
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
