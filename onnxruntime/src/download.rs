//! Module controlling models downloadable from ONNX Model Zoom
//!
//! Pre-trained models are available from the
//! [ONNX Model Zoo](https://github.com/onnx/models).
//!
//! A pre-trained model can be downloaded automatically using the
//! [`SessionBuilder`](../session/struct.SessionBuilder.html)'s
//! [`with_downloaded_model()`](../session/struct.SessionBuilder.html#method.with_downloaded_model) method.
//!
//! See [`AvailableOnnxModel`](enum.AvailableOnnxModel.html) for the different models available
//! to download.

use std::{
    fs, io,
    path::{Path, PathBuf},
    time::Duration,
};

use crate::error::{OrtDownloadError, Result};

/// Available pre-trained models to download from [ONNX Model Zoo](https://github.com/onnx/models).
///
/// According to [ONNX Model Zoo](https://github.com/onnx/models)'s GitHub page:
///
/// > The ONNX Model Zoo is a collection of pre-trained, state-of-the-art models in the ONNX format
/// > contributed by community members like you.
#[derive(Debug, Clone)]
pub enum AvailableOnnxModel {
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

trait ModelUrl {
    fn fetch_url(&self) -> &'static str;
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

impl ModelUrl for AvailableOnnxModel {
    fn fetch_url(&self) -> &'static str {
        match self {
            AvailableOnnxModel::MobileNet => "https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
            AvailableOnnxModel::SqueezeNet => "https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.0-9.onnx",
            AvailableOnnxModel::Inception(version) => version.fetch_url(),
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

impl AvailableOnnxModel {
    pub(crate) fn download_to<P>(&self, download_dir: P) -> Result<PathBuf>
    where
        P: AsRef<Path>,
    {
        let url = self.fetch_url();

        let model_filename = PathBuf::from(url.split('/').last().unwrap());
        let model_filepath = download_dir.as_ref().join(model_filename);

        if model_filepath.exists() {
            println!(
                "File {:?} already exists, not re-downloading.",
                model_filepath
            );
            Ok(model_filepath)
        } else {
            println!("Downloading {:?} to {:?}...", url, model_filepath);

            let resp = ureq::get(url)
                .timeout_connect(1_000) // 1 second
                .timeout(Duration::from_secs(180)) // 3 minutes
                .call();

            assert!(resp.has("Content-Length"));
            let len = resp
                .header("Content-Length")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap();
            println!("Downloading {} bytes...", len);

            let mut reader = resp.into_reader();

            let f = fs::File::create(&model_filepath).unwrap();
            let mut writer = io::BufWriter::new(f);

            let bytes_io_count =
                io::copy(&mut reader, &mut writer).map_err(OrtDownloadError::IoError)?;

            if bytes_io_count == len as u64 {
                Ok(model_filepath)
            } else {
                Err(OrtDownloadError::CopyError {
                    expected: len as u64,
                    io: bytes_io_count,
                }
                .into())
            }
        }
    }
}
