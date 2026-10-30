//! PP-OCRv6 text detection and recognition. Load models once, then reuse [`PpOcr`] on a
//! worker thread. Images use RGB channel order; the models' BGR convention is handled internally.
mod det;
mod onnx;
mod dict;
mod image_ops;
pub mod japanese;
mod rec;
mod runtime;
use image::RgbImage;
pub use runtime::{Backend, BackendInfo};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Errors from configuration, model contracts, image decoding, or inference.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    InvalidInput(String),
    #[error("unexpected model output: {0}")]
    ModelContract(String),
    #[error(transparent)]
    Runtime(#[from] ort::Error),
    #[error(transparent)]
    Image(#[from] image::ImageError),
}
pub type Result<T> = std::result::Result<T, Error>;

/// The published PP-OCRv6 model sizes. Each tier pairs a detector with a recognizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Tier {
    #[default]
    Medium,
    Small,
    Tiny,
}
impl Tier {
    pub fn name(self) -> &'static str {
        match self {
            Tier::Medium => "medium",
            Tier::Small => "small",
            Tier::Tiny => "tiny",
        }
    }
}

/// Model files are supplied by the host application; inference never uses the network.
///
/// PaddlePaddle publishes every model as `inference.onnx` beside an `inference.yml`, so hosts
/// store them under distinct names. `dictionary` is the recognizer's `inference.yml`, whose
/// `PostProcess.character_dict` lists the output classes.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub detection: PathBuf,
    pub recognition: PathBuf,
    pub dictionary: PathBuf,
}
impl ModelPaths {
    /// `PP-OCRv6_<tier>_det.onnx`, `PP-OCRv6_<tier>_rec.onnx` and `PP-OCRv6_<tier>_rec.yml`.
    pub fn from_dir(dir: impl AsRef<Path>, tier: Tier) -> Self {
        let dir = dir.as_ref();
        let stem = format!("PP-OCRv6_{}", tier.name());
        Self {
            detection: dir.join(format!("{stem}_det.onnx")),
            recognition: dir.join(format!("{stem}_rec.onnx")),
            dictionary: dir.join(format!("{stem}_rec.yml")),
        }
    }
}
#[derive(Debug, Clone)]
pub struct Config {
    pub backend: Backend,
    /// Maximum recognition lines per inference call. Must be positive.
    pub max_batch_size: usize,
    /// CPU worker threads; None uses ONNX Runtime's default.
    pub intra_threads: Option<usize>,
    /// Longest side of the detector's input, in pixels; larger images are scaled down.
    /// Smaller is faster; larger finds smaller text. Rounded to a multiple of 32.
    pub detection_max_side: u32,
    /// DirectML/CUDA adapter index; None uses the provider's default (DXGI adapter 0).
    pub device_id: Option<i32>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            backend: Backend::Auto,
            max_batch_size: 8,
            intra_threads: None,
            detection_max_side: 960,
            device_id: None,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct OcrOptions {
    /// Probability above which a detector pixel counts as text (DB `thresh`).
    pub detection_threshold: f32,
    /// Minimum mean probability inside a candidate box (DB `box_thresh`).
    pub box_threshold: f32,
    /// How far a text kernel is grown into its line box (DB `unclip_ratio`).
    pub unclip_ratio: f32,
    /// Lines whose mean character confidence is below this are dropped.
    pub recognition_threshold: f32,
}
impl Default for OcrOptions {
    fn default() -> Self {
        // PP-OCRv6 det's inference.yml, plus PaddleOCR's default recognition score filter.
        Self {
            detection_threshold: 0.2,
            box_threshold: 0.45,
            unclip_ratio: 1.4,
            recognition_threshold: 0.5,
        }
    }
}
impl OcrOptions {
    fn validate(self) -> Result<()> {
        for v in [
            self.detection_threshold,
            self.box_threshold,
            self.recognition_threshold,
        ] {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                return Err(Error::InvalidInput(
                    "thresholds must be finite and in [0, 1]".into(),
                ));
            }
        }
        if !self.unclip_ratio.is_finite() || self.unclip_ratio < 0.0 {
            return Err(Error::InvalidInput(
                "unclip ratio must be finite and nonnegative".into(),
            ));
        }
        Ok(())
    }
}
/// Pixel coordinates [left, top, right, bottom], with exclusive right/bottom edges.
pub type BoundingBox = [u32; 4];
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub confidence: f32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Character {
    pub character: char,
    pub bbox: BoundingBox,
    pub confidence: f32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextLine {
    pub text: String,
    pub bbox: BoundingBox,
    pub detection_confidence: Option<f32>,
    pub characters: Vec<Character>,
    pub is_vertical: bool,
}
/// Reusable sessions. Methods take &mut self; share through a Mutex or dedicated worker.
pub struct PpOcr {
    detection: runtime::Model,
    recognition: runtime::Model,
    dictionary: Vec<String>,
    config: Config,
    backend: BackendInfo,
}
impl PpOcr {
    pub fn new(models: ModelPaths) -> Result<Self> {
        Self::with_config(models, Config::default())
    }
    pub fn with_config(models: ModelPaths, config: Config) -> Result<Self> {
        if config.max_batch_size == 0
            || config.intra_threads == Some(0)
            || config.detection_max_side < 32
        {
            return Err(Error::InvalidInput(
                "batch size and thread count must be positive, detection side at least 32".into(),
            ));
        }
        for path in [&models.detection, &models.recognition, &models.dictionary] {
            if !path.is_file() {
                return Err(Error::InvalidInput(format!(
                    "model file missing: {}",
                    path.display()
                )));
            }
        }
        let yaml = std::fs::read_to_string(&models.dictionary)
            .map_err(|e| Error::InvalidInput(format!("{}: {e}", models.dictionary.display())))?;
        let dictionary = dict::character_dict(&yaml)?;
        let (detection, mut recognition, backend) = runtime::load(&models, &config)?;
        rec::check_classes(&mut recognition, dictionary.len())?;
        Ok(Self {
            detection,
            recognition,
            dictionary,
            config,
            backend,
        })
    }
    /// Provider successfully registered for all sessions; individual unsupported ops can use CPU.
    pub fn backend_info(&self) -> &BackendInfo {
        &self.backend
    }
    pub fn ocr_file(
        &mut self,
        path: impl AsRef<Path>,
        options: OcrOptions,
    ) -> Result<Vec<TextLine>> {
        self.ocr(&image::open(path)?.to_rgb8(), options)
    }
    pub fn ocr(&mut self, image: &RgbImage, options: OcrOptions) -> Result<Vec<TextLine>> {
        options.validate()?;
        let boxes = det::detect(
            &mut self.detection,
            image,
            self.config.detection_max_side,
            options,
        )?;
        self.recognize_boxes(image, &boxes, options)
    }
    /// Detect text lines in reading order: top to bottom, then left to right.
    pub fn detect(&mut self, image: &RgbImage, options: OcrOptions) -> Result<Vec<Detection>> {
        options.validate()?;
        det::detect(
            &mut self.detection,
            image,
            self.config.detection_max_side,
            options,
        )
    }
    /// Recognize already cropped lines. Coordinates are relative to each crop.
    pub fn recognize(&mut self, images: &[RgbImage], options: OcrOptions) -> Result<Vec<TextLine>> {
        options.validate()?;
        let options = OcrOptions {
            recognition_threshold: 0.0,
            ..options
        };
        images
            .iter()
            .map(|img| {
                image_ops::validate(img)?;
                let boxes = [Detection {
                    bbox: [0, 0, img.width(), img.height()],
                    confidence: 1.0,
                }];
                let mut lines = self.recognize_boxes(img, &boxes, options)?;
                let mut line = lines.pop().ok_or_else(|| {
                    Error::ModelContract("recognition returned no line".into())
                })?;
                line.detection_confidence = None;
                Ok(line)
            })
            .collect()
    }
    fn recognize_boxes(
        &mut self,
        image: &RgbImage,
        boxes: &[Detection],
        options: OcrOptions,
    ) -> Result<Vec<TextLine>> {
        rec::recognize(
            &mut self.recognition,
            &self.dictionary,
            image,
            boxes,
            options.recognition_threshold,
            self.config.max_batch_size,
        )
    }
}
