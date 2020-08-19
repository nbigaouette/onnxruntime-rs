//! Module defining computer vision models available to download.

use super::{AvailableOnnxModel, ModelUrl};

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

/// Image classification model
#[derive(Debug, Clone)]
pub enum DomainBasedImageClassification {
    /// Handwritten digits prediction using CNN
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/mnist](https://github.com/onnx/models/tree/master/vision/classification/mnist)
    ///
    /// Variant downloaded: ONNX Version 1.3 with Opset Version 8.
    Mnist,
}

/// Image classification model
///
/// > This collection of models take images as input, then classifies the major objects in the images
/// > into 1000 object categories such as keyboard, mouse, pencil, and many animals.
///
/// Source: [https://github.com/onnx/models#image-classification-](https://github.com/onnx/models#image-classification-)
#[derive(Debug, Clone)]
pub enum ImageClassificationModel {
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
    /// Image classification, trained on ImageNet with 1000 classes.
    ///
    /// > ResNet models provide very high accuracies with affordable model sizes. They are ideal for cases when
    /// > high accuracy of classification is required.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/resnet](https://github.com/onnx/models/tree/master/vision/classification/resnet)
    ResNet(ResNet),
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
    /// Image classification, trained on ImageNet with 1000 classes.
    ///
    /// > VGG models provide very high accuracies but at the cost of increased model sizes. They are ideal for
    /// > cases when high accuracy of classification is essential and there are limited constraints on model sizes.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/vgg](https://github.com/onnx/models/tree/master/vision/classification/vgg)
    Vgg(Vgg),
    /// Convolutional neural network for classification, which competed in the ImageNet Large Scale Visual Recognition Challenge in 2012.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/alexnet](https://github.com/onnx/models/tree/master/vision/classification/alexnet)
    ///
    /// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
    AlexNet,
    /// Convolutional neural network for classification, which competed in the ImageNet Large Scale Visual Recognition Challenge in 2014.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/googlenet](https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/googlenet)
    ///
    /// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
    GoogleNet,
    /// Variant of AlexNet, it's the name of a convolutional neural network for classification, which competed in the ImageNet Large Scale Visual Recognition Challenge in 2012.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/caffenet](https://github.com/onnx/models/tree/master/vision/classification/caffenet)
    ///
    /// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
    CaffeNet,
    /// Convolutional neural network for detection.
    ///
    /// > This model was made by transplanting the R-CNN SVM classifiers into a fc-rcnn classification layer.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/rcnn_ilsvrc13](https://github.com/onnx/models/tree/master/vision/classification/rcnn_ilsvrc13)
    ///
    /// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
    RcnnIlsvrc13,
    /// Convolutional neural network for classification.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/rcnn_ilsvrc13](https://github.com/onnx/models/tree/master/vision/classification/rcnn_ilsvrc13)
    ///
    /// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
    DenseNet121,
    /// Google's Inception
    Inception(InceptionVersion),
    /// Computationally efficient CNN architecture designed specifically for mobile devices with very limited computing power.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/shufflenet](https://github.com/onnx/models/tree/master/vision/classification/shufflenet)
    ShuffleNet(ShuffleNetVersion),
    /// Deep convolutional networks for classification.
    ///
    /// > This model's 4th layer has 512 maps instead of 1024 maps mentioned in the paper.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/zfnet-512](https://github.com/onnx/models/tree/master/vision/classification/zfnet-512)
    ZFNet512,
    /// Image classification model that achieves state-of-the-art accuracy.
    ///
    /// >  It is designed to run on mobile CPU, GPU, and EdgeTPU devices, allowing for applications on mobile and loT, where computational resources are limited.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/efficientnet-lite4](https://github.com/onnx/models/tree/master/vision/classification/efficientnet-lite4)
    ///
    /// Variant downloaded: ONNX Version 1.7.0 with Opset Version 11.
    EfficientNetLite4,
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

/// ResNet
///
/// Source: [https://github.com/onnx/models/tree/master/vision/classification/resnet](https://github.com/onnx/models/tree/master/vision/classification/resnet)
#[derive(Debug, Clone)]
pub enum ResNet {
    /// ResNet v1
    V1(ResNetV1),
    /// ResNet v2
    V2(ResNetV2),
}
/// ResNet v1
///
/// Source: [https://github.com/onnx/models/tree/master/vision/classification/resnet](https://github.com/onnx/models/tree/master/vision/classification/resnet)
#[derive(Debug, Clone)]
pub enum ResNetV1 {
    /// ResNet18
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet18,
    /// ResNet34
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet34,
    /// ResNet50
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet50,
    /// ResNet101
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet101,
    /// ResNet152
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet152,
}
/// ResNet v2
///
/// Source: [https://github.com/onnx/models/tree/master/vision/classification/resnet](https://github.com/onnx/models/tree/master/vision/classification/resnet)
#[derive(Debug, Clone)]
pub enum ResNetV2 {
    /// ResNet18
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet18,
    /// ResNet34
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet34,
    /// ResNet50
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet50,
    /// ResNet101
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet101,
    /// ResNet152
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet152,
}

/// ResNet
///
/// Source: [https://github.com/onnx/models/tree/master/vision/classification/resnet](https://github.com/onnx/models/tree/master/vision/classification/resnet)
#[derive(Debug, Clone)]
pub enum Vgg {
    /// VGG with 16 convolutional layers
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    Vgg16,
    /// VGG with 16 convolutional layers, with batch normalization applied after each convolutional layer.
    ///
    /// The batch normalization leads to better convergence and slightly better accuracies.
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    Vgg16Bn,
    /// VGG with 19 convolutional layers
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    Vgg19,
    /// VGG with 19 convolutional layers, with batch normalization applied after each convolutional layer.
    ///
    /// The batch normalization leads to better convergence and slightly better accuracies.
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    Vgg19Bn,
}

/// Computationally efficient CNN architecture designed specifically for mobile devices with very limited computing power.
///
/// Source: [https://github.com/onnx/models/tree/master/vision/classification/shufflenet](https://github.com/onnx/models/tree/master/vision/classification/shufflenet)
#[derive(Debug, Clone)]
pub enum ShuffleNetVersion {
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/shufflenet](https://github.com/onnx/models/tree/master/vision/classification/shufflenet)
    ///
    /// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
    V1,
    /// ShuffleNetV2 is an improved architecture that is the state-of-the-art in terms of speed and accuracy tradeoff used for image classification.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/shufflenet](https://github.com/onnx/models/tree/master/vision/classification/shufflenet)
    ///
    /// Variant downloaded: ONNX Version 1.6 with Opset Version 10.
    V2,
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

impl ModelUrl for DomainBasedImageClassification {
    fn fetch_url(&self) -> &'static str {
        match self {
            DomainBasedImageClassification::Mnist => "https://github.com/onnx/models/raw/master/vision/classification/mnist/model/mnist-8.onnx",
        }
    }
}

impl ModelUrl for ImageClassificationModel {
    fn fetch_url(&self) -> &'static str {
        match self {
            ImageClassificationModel::MobileNet => "https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
            ImageClassificationModel::SqueezeNet => "https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.0-9.onnx",
            ImageClassificationModel::Inception(version) => version.fetch_url(),
            ImageClassificationModel::ResNet(version) => version.fetch_url(),
            ImageClassificationModel::Vgg(variant) => variant.fetch_url(),
            ImageClassificationModel::AlexNet => "https://github.com/onnx/models/raw/master/vision/classification/alexnet/model/bvlcalexnet-9.onnx",
            ImageClassificationModel::GoogleNet => "https://github.com/onnx/models/raw/master/vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.onnx",
            ImageClassificationModel::CaffeNet => "https://github.com/onnx/models/raw/master/vision/classification/caffenet/model/caffenet-9.onnx",
            ImageClassificationModel::RcnnIlsvrc13 => "https://github.com/onnx/models/raw/master/vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-9.onnx",
            ImageClassificationModel::DenseNet121 => "https://github.com/onnx/models/raw/master/vision/classification/densenet-121/model/densenet-9.onnx",
            ImageClassificationModel::ShuffleNet(version) => version.fetch_url(),
            ImageClassificationModel::ZFNet512 => "https://github.com/onnx/models/raw/master/vision/classification/zfnet-512/model/zfnet512-9.onnx",
            ImageClassificationModel::EfficientNetLite4 => "https://github.com/onnx/models/raw/master/vision/classification/efficientnet-lite4/model/efficientnet-lite4.onnx"
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

impl ModelUrl for ResNet {
    fn fetch_url(&self) -> &'static str {
        match self {
            ResNet::V1(variant) => variant.fetch_url(),
            ResNet::V2(variant) => variant.fetch_url(),
        }
    }
}

impl ModelUrl for ResNetV1 {
    fn fetch_url(&self) -> &'static str {
        match self {
            ResNetV1::ResNet18 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet18-v1-7.onnx",
            ResNetV1::ResNet34 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet34-v1-7.onnx",
            ResNetV1::ResNet50 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v1-7.onnx",
            ResNetV1::ResNet101 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet101-v1-7.onnx",
            ResNetV1::ResNet152 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet152-v1-7.onnx",
        }
    }
}

impl ModelUrl for ResNetV2 {
    fn fetch_url(&self) -> &'static str {
        match self {
            ResNetV2::ResNet18 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet18-v2-7.onnx",
            ResNetV2::ResNet34 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet34-v2-7.onnx",
            ResNetV2::ResNet50 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx",
            ResNetV2::ResNet101 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet101-v2-7.onnx",
            ResNetV2::ResNet152 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet152-v2-7.onnx",
        }
    }
}

impl ModelUrl for Vgg {
    fn fetch_url(&self) -> &'static str {
        match self {
            Vgg::Vgg16 => "https://github.com/onnx/models/raw/master/vision/classification/vgg/model/vgg16-7.onnx",
            Vgg::Vgg16Bn => "https://github.com/onnx/models/raw/master/vision/classification/vgg/model/vgg16-bn-7.onnx",
            Vgg::Vgg19 => "https://github.com/onnx/models/raw/master/vision/classification/vgg/model/vgg19-7.onnx",
            Vgg::Vgg19Bn => "https://github.com/onnx/models/raw/master/vision/classification/vgg/model/vgg19-bn-7.onnx",
        }
    }
}

impl ModelUrl for ShuffleNetVersion {
    fn fetch_url(&self) -> &'static str {
        match self {
            ShuffleNetVersion::V1 => "https://github.com/onnx/models/raw/master/vision/classification/shufflenet/model/shufflenet-9.onnx",
            ShuffleNetVersion::V2 => "https://github.com/onnx/models/raw/master/vision/classification/shufflenet/model/shufflenet-v2-10.onnx",
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

impl From<ImageClassificationModel> for AvailableOnnxModel {
    fn from(model: ImageClassificationModel) -> Self {
        AvailableOnnxModel::Vision(Vision::ImageClassification(model))
    }
}

impl From<InceptionVersion> for AvailableOnnxModel {
    fn from(model: InceptionVersion) -> Self {
        AvailableOnnxModel::Vision(Vision::ImageClassification(
            ImageClassificationModel::Inception(model),
        ))
    }
}
