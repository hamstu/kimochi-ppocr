use crate::{Config, Error, ModelPaths, Result};
use ndarray::Array4;
use ort::{
    execution_providers::ExecutionProviderDispatch,
    session::{Session, SessionOutputs, builder::GraphOptimizationLevel},
    value::{Tensor, ValueType},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Backend {
    /// Try platform accelerators, then CPU. Failures are recorded in BackendInfo.
    #[default]
    Auto,
    Cpu,
    CoreMl,
    Cuda,
    DirectMl,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    pub requested: Backend,
    pub selected: Backend,
    pub fallback_reasons: Vec<String>,
}
fn provider(backend: Backend, device: Option<i32>) -> Result<Option<ExecutionProviderDispatch>> {
    let _ = device; // only CUDA and DirectML take a device
    match backend {
        Backend::Cpu => Ok(None),
        #[cfg(feature = "coreml")]
        Backend::CoreMl => Ok(Some(
            ort::execution_providers::CoreMLExecutionProvider::default()
                .with_static_input_shapes(true)
                .with_model_format(ort::execution_providers::coreml::CoreMLModelFormat::MLProgram)
                .with_compute_units(ort::execution_providers::coreml::CoreMLComputeUnits::All)
                .build()
                .error_on_failure(),
        )),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(Some(
            ort::execution_providers::CUDAExecutionProvider::default()
                .with_device_id(device.unwrap_or(0))
                .build()
                .error_on_failure(),
        )),
        #[cfg(feature = "directml")]
        Backend::DirectMl => Ok(Some(
            ort::execution_providers::DirectMLExecutionProvider::default()
                .with_device_id(device.unwrap_or(0))
                .build()
                .error_on_failure(),
        )),
        _ => Err(Error::InvalidInput(format!(
            "backend {backend:?} is not compiled in"
        ))),
    }
}

/// Whether a backend gets one session per input shape.
///
/// PP-OCR's exports have dynamic batch, height and width. ONNX Runtime 1.22's DirectML provider
/// runs a dynamic session's first input shape at full speed, but once the session has seen a
/// second shape every run takes the slow path (3–4× slower on an RTX 3080 Ti, for any shape).
/// Sessions with every dimension pinned stay fast. Core ML likewise compiles only static shapes.
fn per_shape(backend: Backend) -> bool {
    matches!(backend, Backend::DirectMl | Backend::CoreMl)
}

/// One model file and the sessions that run it.
pub(crate) struct Model {
    bytes: Vec<u8>,
    backend: Backend,
    config: Config,
    /// The input's symbolic dimension names, by axis ("" for fixed axes).
    symbols: Vec<String>,
    /// Most recently used first. One entry (any shape) unless `per_shape`.
    sessions: Vec<(Vec<usize>, Session)>,
    capacity: usize,
    /// Whether the graph was rewritten to return best classes and scores (`onnx.rs`), and the
    /// class count its original output declared.
    pub compact: bool,
    pub declared_classes: Option<i64>,
}
impl Model {
    fn build(&self, shape: Option<&[usize]>) -> Result<Session> {
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_op_spinning(false)?
            .with_config_entry("session.inter_op.allow_spinning", "0")?;
        if let Some(n) = self.config.intra_threads {
            builder = builder.with_intra_threads(n)?;
        }
        if self.backend == Backend::DirectMl {
            builder = builder
                .with_parallel_execution(false)?
                .with_memory_pattern(false)?;
        }
        if let Some(shape) = shape {
            for (symbol, &size) in self.symbols.iter().zip(shape) {
                if !symbol.is_empty() {
                    builder = builder.with_dimension_override(symbol, size as i64)?;
                }
            }
        }
        if let Some(ep) = provider(self.backend, self.config.device_id)? {
            builder = builder.with_execution_providers([ep])?;
        }
        Ok(builder.commit_from_memory(&self.bytes)?)
    }
    fn open(bytes: Vec<u8>, backend: Backend, config: &Config, capacity: usize) -> Result<Self> {
        let mut model = Self {
            bytes,
            backend,
            config: config.clone(),
            symbols: Vec::new(),
            sessions: Vec::new(),
            capacity: if per_shape(backend) { capacity } else { 1 },
            compact: false,
            declared_classes: None,
        };
        // The dynamic session supplies the dimension names; per-shape backends then drop it.
        let session = model.build(None)?;
        let ValueType::Tensor { dimension_symbols, shape, .. } = &session.inputs[0].input_type else {
            return Err(Error::ModelContract("model input is not a tensor".into()));
        };
        if shape.len() != 4 {
            return Err(Error::ModelContract(format!("model input {shape:?} is not NCHW")));
        }
        model.symbols = dimension_symbols.iter().map(|s| s.to_string()).collect();
        if !per_shape(backend) {
            model.sessions.push((Vec::new(), session));
        }
        Ok(model)
    }
    /// The session for `shape`, built (and the least recently used evicted) on first use.
    fn session(&mut self, shape: &[usize]) -> Result<&mut Session> {
        if per_shape(self.backend) {
            match self.sessions.iter().position(|(s, _)| s == shape) {
                Some(0) => {}
                Some(i) => {
                    let hit = self.sessions.remove(i);
                    self.sessions.insert(0, hit);
                }
                None => {
                    let session = self.build(Some(shape))?;
                    self.sessions.truncate(self.capacity - 1);
                    self.sessions.insert(0, (shape.to_vec(), session));
                }
            }
        }
        Ok(&mut self.sessions[0].1)
    }
    /// Whether every distinct input shape costs a session (so callers should keep shapes few).
    pub fn per_shape(&self) -> bool {
        per_shape(self.backend)
    }
    /// Run the model's single input and hand the first output's shape and values to `read`,
    /// which borrows them in place (a recognizer output is 75 KB per time step).
    pub fn run<R>(
        &mut self,
        input: Array4<f32>,
        read: impl FnOnce(&[usize], &[f32]) -> Result<R>,
    ) -> Result<R> {
        self.run_outputs(input, |outputs| {
            let (shape, values) = tensor::<f32>(outputs, 0)?;
            read(&shape, values)
        })
    }
    /// Run the model's single input and hand every output to `read`.
    pub fn run_outputs<R>(
        &mut self,
        input: Array4<f32>,
        read: impl FnOnce(&SessionOutputs<'_>) -> Result<R>,
    ) -> Result<R> {
        let shape = input.shape().to_vec();
        let session = self.session(&shape)?;
        let name = session.inputs[0].name.clone();
        let outputs = session.run(ort::inputs![name.as_str() => Tensor::from_array(input)?])?;
        read(&outputs)
    }
}

/// Output `index` as its shape and borrowed values.
pub(crate) fn tensor<'a, T: ort::tensor::PrimitiveTensorElementType + std::fmt::Debug + 'a>(
    outputs: &'a SessionOutputs<'_>,
    index: usize,
) -> Result<(Vec<usize>, &'a [T])> {
    let (shape, values) = outputs[index].try_extract_tensor::<T>()?;
    if shape.iter().any(|&d| d <= 0) {
        return Err(Error::ModelContract(format!("output shape {shape:?}")));
    }
    Ok((shape.iter().map(|&d| d as usize).collect(), values))
}

fn read(path: &std::path::Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| Error::InvalidInput(format!("{}: {e}", path.display())))
}

/// The recognizer with compact outputs where the rewrite and the backend accept it, else as shipped.
fn recognizer(path: &std::path::Path, backend: Backend, config: &Config) -> Result<Model> {
    let bytes = read(path)?;
    let capacity = crate::rec::WIDTHS.len();
    if let Ok((compact, declared)) = crate::onnx::compact_recognizer(&bytes)
        && let Ok(mut model) = Model::open(compact, backend, config, capacity) {
            model.compact = true;
            model.declared_classes = declared;
            return Ok(model);
        }
    Model::open(bytes, backend, config, capacity)
}

/// Input shapes exercised at load: they prove the backend works (some Windows adapters accept
/// DirectML sessions but fail on the first inference) and pre-build the common sessions.
const DETECTION_WARM: [usize; 4] = [1, 3, 544, 960];

pub(crate) fn load(models: &ModelPaths, config: &Config) -> Result<(Model, Model, BackendInfo)> {
    let mut candidates = Vec::new();
    if config.backend == Backend::Auto {
        if cfg!(all(
            target_os = "macos",
            target_arch = "aarch64",
            feature = "coreml"
        )) {
            candidates.push(Backend::CoreMl);
        }
        if cfg!(all(
            any(target_os = "windows", target_os = "linux"),
            feature = "cuda"
        )) {
            candidates.push(Backend::Cuda);
        }
        if cfg!(all(target_os = "windows", feature = "directml")) {
            candidates.push(Backend::DirectMl);
        }
        candidates.push(Backend::Cpu);
    } else {
        candidates.push(config.backend);
    }
    let mut reasons = Vec::new();
    for backend in candidates {
        let attempt = (|| {
            let mut det = Model::open(read(&models.detection)?, backend, config, 2)?;
            let mut rec = recognizer(&models.recognition, backend, config)?;
            if backend != Backend::Cpu {
                det.run(Array4::zeros(DETECTION_WARM), |_, _| Ok(())).map_err(|err| {
                    Error::ModelContract(format!("{backend:?} detection inference probe failed: {err}"))
                })?;
                for &width in crate::rec::WARM_WIDTHS {
                    rec.run_outputs(Array4::zeros((1, 3, crate::rec::HEIGHT, width)), |_| Ok(())).map_err(|err| {
                        Error::ModelContract(format!("{backend:?} recognition inference probe failed: {err}"))
                    })?;
                }
            }
            Ok::<_, Error>((det, rec))
        })();
        match attempt {
            Ok((det, rec)) => {
                return Ok((
                    det,
                    rec,
                    BackendInfo {
                        requested: config.backend,
                        selected: backend,
                        fallback_reasons: reasons,
                    },
                ));
            }
            Err(err) => {
                if config.backend != Backend::Auto || backend == Backend::Cpu {
                    return Err(err);
                }
                reasons.push(format!("{backend:?}: {err}"));
            }
        }
    }
    unreachable!("CPU is always a candidate")
}
