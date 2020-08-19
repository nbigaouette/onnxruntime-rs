//! Module defining computer vision models available to download.

use super::ModelUrl;

pub mod body_face_gesture_analysis;
pub mod domain_based_image_classification;
pub mod image_classification;
pub mod image_manipulation;
pub mod object_detection_image_segmentation;

use body_face_gesture_analysis::BodyFaceGestureAnalysis;
use domain_based_image_classification::DomainBasedImageClassification;
use image_classification::ImageClassificationModel;
use image_manipulation::ImageManipulation;
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
