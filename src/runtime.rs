use crate::{Config, Error, ModelPaths, Result};
use std::path::Path;
use ndarray::Array4;
use ort::{
    execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
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
fn provider(backend: Backend, config: &Config) -> Result<Option<ExecutionProviderDispatch>> {
    let device = config.device_id; // only CUDA and DirectML take a device
    match backend {
        Backend::Cpu => Ok(None),
        #[cfg(feature = "coreml")]
        Backend::CoreMl => Ok(Some(
            {
                let mut ep = ort::execution_providers::CoreMLExecutionProvider::default()
                    .with_static_input_shapes(true)
                    .with_model_format(ort::execution_providers::coreml::CoreMLModelFormat::MLProgram)
                    // GPU rather than `All`: the same speed (Apple M5 Max: the warm pipeline runs on
                    // the GPU either way; the Neural Engine alone is 5× slower), without the Neural
                    // Engine compile, which is a quarter of a cold load.
                    .with_compute_units(ort::execution_providers::coreml::CoreMLComputeUnits::CPUAndGPU);
                if let Some(dir) = &config.cache_dir {
                    ep = ep.with_model_cache_dir(dir.display());
                }
                ep.build().error_on_failure()
            },
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

/// Core ML's compiled-model cache is keyed by the model's `COREML_CACHE_KEY` metadata, else by a
/// hash of its input and node names (ONNX Runtime's `coreml_execution_provider.cc`), which is the
/// same for every input shape: without a key, a session pinned to one shape would load a model
/// compiled for another. The key is alphanumeric and at most 64 characters, as ONNX Runtime requires; the
/// leading version letter retires every entry when this scheme or the rewrite changes.
fn cache_key(content_hash: u64, shape: &[usize]) -> String {
    let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
    format!("{}{}", cache_prefix(content_hash), dims.join("x"))
}
fn cache_prefix(content_hash: u64) -> String {
    format!("a{content_hash:016x}s")
}

/// Compiled models kept in the cache, most recently built first. Each is one model at one input
/// shape (detector 19 MB, recognizer 68 MB): the recognizer's few widths and the detector's
/// recent capture sizes.
const CACHE_ENTRIES: usize = 24;

/// Mark a cache entry as used, for `prune_cache`'s ordering. Errors only cost that ordering.
fn touch(dir: &Path, key: &str) {
    let _ = set_modified(&dir.join(key), std::time::SystemTime::now());
}

/// Set a cache entry's (a directory's) modification time. Windows opens a directory only with
/// FILE_FLAG_BACKUP_SEMANTICS, and needs write access to change its times; Unix needs neither.
fn set_modified(path: &Path, time: std::time::SystemTime) -> std::io::Result<()> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.write(true).custom_flags(0x0200_0000);
    }
    options.open(path)?.set_modified(time)
}

/// Delete this crate's entries that belong to other model files or an older key scheme, then the
/// least recently built beyond `CACHE_ENTRIES`. Other names in the directory are left alone.
fn prune_cache(dir: &Path, current: &[u64]) {
    let Ok(listing) = std::fs::read_dir(dir) else { return };
    let ours = |name: &str| {
        let b = name.as_bytes();
        b.len() > 18
            && b[0].is_ascii_lowercase()
            && b[1..17].iter().all(|c| c.is_ascii_hexdigit())
            && b[17] == b's'
            && b[18..].iter().all(|c| c.is_ascii_digit() || *c == b'x')
    };
    let prefixes: Vec<String> = current.iter().map(|&h| cache_prefix(h)).collect();
    let mut kept = Vec::new();
    for entry in listing.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !ours(&name) {
            continue;
        }
        if prefixes.iter().any(|p| name.starts_with(p.as_str())) {
            let used = entry.metadata().and_then(|m| m.modified()).ok();
            kept.push((used, entry.path()));
        } else {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
    kept.sort_by_key(|k| std::cmp::Reverse(k.0));
    for (_, path) in kept.into_iter().skip(CACHE_ENTRIES) {
        let _ = std::fs::remove_dir_all(path);
    }
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
    /// The model's content hash, when Core ML caches its compiled models (`cache_key`).
    content_hash: Option<u64>,
}
/// Whether sessions built ahead of use may be built concurrently, one thread each.
///
/// DirectML session creation is mostly shader compilation on the CPU: on a GTX 1080 the three
/// warm-up recognizer sessions took 1.0–2.4 s one after another and 0.7 s together. Core ML stays
/// sequential: on an M5 Max, building together saved 0.3 s of a cached load (1.6 → 1.3 s) and
/// 0.7 s of a cold one, for 390 MiB more peak memory.
fn parallel_build(backend: Backend) -> bool {
    backend == Backend::DirectMl
}

fn build_session(
    bytes: &[u8],
    backend: Backend,
    config: &Config,
    symbols: &[String],
    shape: Option<&[usize]>,
    content_hash: Option<u64>,
) -> Result<Session> {
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_op_spinning(false)?
        .with_config_entry("session.inter_op.allow_spinning", "0")?;
    if let Some(n) = config.intra_threads {
        builder = builder.with_intra_threads(n)?;
    }
    if backend == Backend::DirectMl {
        builder = builder
            .with_parallel_execution(false)?
            .with_memory_pattern(false)?;
    }
    if let Some(shape) = shape {
        for (symbol, &size) in symbols.iter().zip(shape) {
            if !symbol.is_empty() {
                builder = builder.with_dimension_override(symbol, size as i64)?;
            }
        }
    }
    if let Some(ep) = provider(backend, config)? {
        builder = builder.with_execution_providers([ep])?;
    }
    match (content_hash, shape) {
        (Some(hash), Some(shape)) => {
            let key = cache_key(hash, shape);
            let keyed = crate::onnx::with_metadata(bytes, "COREML_CACHE_KEY", &key);
            let session = builder.commit_from_memory(&keyed)?;
            if let Some(dir) = &config.cache_dir {
                touch(dir, &key);
            }
            Ok(session)
        }
        _ => Ok(builder.commit_from_memory(bytes)?),
    }
}

impl Model {
    fn build(&self, shape: Option<&[usize]>) -> Result<Session> {
        build_session(&self.bytes, self.backend, &self.config, &self.symbols, shape, self.content_hash)
    }
    /// Build the sessions for `shapes` that are not cached yet (per-shape backends only; the
    /// others have their one session already), concurrently where `parallel_build` allows.
    /// Called from a thread while another model does the same, so it takes the plain parts.
    fn build_shapes(&self, shapes: &[[usize; 4]]) -> Result<Vec<([usize; 4], Session)>> {
        let missing: Vec<[usize; 4]> = shapes
            .iter()
            .copied()
            .filter(|shape| !self.sessions.iter().any(|(s, _)| s == shape))
            .collect();
        if !per_shape(self.backend) || missing.is_empty() {
            return Ok(Vec::new());
        }
        let build = |shape: [usize; 4]| -> Result<([usize; 4], Session)> {
            Ok((shape, self.build(Some(&shape))?))
        };
        if !parallel_build(self.backend) || missing.len() == 1 {
            return missing.into_iter().map(build).collect();
        }
        std::thread::scope(|scope| {
            let handles: Vec<_> = missing.iter().map(|&shape| scope.spawn(move || build(shape))).collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err(Error::ModelContract("session build panicked".into()))))
                .collect()
        })
    }
    /// Cache sessions from `build_shapes`, most recently built first.
    fn adopt(&mut self, built: Vec<([usize; 4], Session)>) {
        for (shape, session) in built {
            self.sessions.retain(|(s, _)| s != &shape);
            self.sessions.truncate(self.capacity.saturating_sub(1));
            self.sessions.insert(0, (shape.to_vec(), session));
        }
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
            content_hash: None,
        };
        if backend == Backend::CoreMl
            && let Some(dir) = &config.cache_dir
        {
            // ONNX Runtime creates only the last directory level.
            let _ = std::fs::create_dir_all(dir);
            model.content_hash = Some(crate::onnx::fnv1a(&model.bytes));
        }
        if per_shape(backend) {
            // Every session is pinned, so the dimension names come from the file: a dynamic
            // session built only to read them would cost ~300 ms per model on DirectML.
            model.symbols = crate::onnx::input_symbols(&model.bytes)?;
            if model.symbols.len() != 4 {
                return Err(Error::ModelContract(format!("model input with {} axes is not NCHW", model.symbols.len())));
            }
            return Ok(model);
        }
        let session = model.build(None)?;
        let ValueType::Tensor { dimension_symbols, shape, .. } = &session.inputs[0].input_type else {
            return Err(Error::ModelContract("model input is not a tensor".into()));
        };
        if shape.len() != 4 {
            return Err(Error::ModelContract(format!("model input {shape:?} is not NCHW")));
        }
        model.symbols = dimension_symbols.iter().map(|s| s.to_string()).collect();
        model.sessions.push((Vec::new(), session));
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
/// Detector sessions kept on per-shape backends: one per capture size seen lately. Each costs
/// about 650 ms to build on a GTX 1080, so alternating between a few capture regions or window
/// sizes must not rebuild them every scan.
const DETECTION_SESSIONS: usize = 4;

pub(crate) fn load(models: &ModelPaths, config: &Config) -> Result<(Model, Model, BackendInfo)> {
    let mut candidates = Vec::new();
    if config.backend == Backend::Auto {
        // Only providers the loaded ONNX Runtime was built with: Kimochi's Windows runtime is
        // the DirectML build, so CUDA would otherwise be tried (and fail on a missing provider
        // library) before DirectML on every load.
        let available = |ep: &dyn ExecutionProvider| ep.is_available().unwrap_or(false);
        if cfg!(all(
            target_os = "macos",
            target_arch = "aarch64",
            feature = "coreml"
        )) {
            candidates.push(Backend::CoreMl);
        }
        #[cfg(feature = "cuda")]
        if cfg!(any(target_os = "windows", target_os = "linux"))
            && available(&ort::execution_providers::CUDAExecutionProvider::default())
        {
            candidates.push(Backend::Cuda);
        }
        #[cfg(feature = "directml")]
        if cfg!(target_os = "windows")
            && available(&ort::execution_providers::DirectMLExecutionProvider::default())
        {
            candidates.push(Backend::DirectMl);
        }
        candidates.push(Backend::Cpu);
    } else {
        candidates.push(config.backend);
    }
    let mut reasons = Vec::new();
    for backend in candidates {
        let attempt = (|| {
            let mut det = Model::open(read(&models.detection)?, backend, config, DETECTION_SESSIONS)?;
            let mut rec = recognizer(&models.recognition, backend, config)?;
            if per_shape(backend) {
                // The warm-up sessions, the detector's and the recognizer's side by side where
                // the backend allows it (and each recognizer width on its own thread in turn).
                let rec_shapes: Vec<[usize; 4]> =
                    crate::rec::WARM_WIDTHS.iter().map(|&w| [1, 3, crate::rec::HEIGHT, w]).collect();
                let (det_built, rec_built) = if parallel_build(backend) {
                    std::thread::scope(|scope| {
                        let d = scope.spawn(|| det.build_shapes(&[DETECTION_WARM]));
                        let r = scope.spawn(|| rec.build_shapes(&rec_shapes));
                        let join = |h: std::thread::ScopedJoinHandle<'_, Result<_>>| {
                            h.join().unwrap_or_else(|_| Err(Error::ModelContract("session build panicked".into())))
                        };
                        (join(d), join(r))
                    })
                } else {
                    (det.build_shapes(&[DETECTION_WARM]), rec.build_shapes(&rec_shapes))
                };
                det.adopt(det_built?);
                rec.adopt(rec_built?);
            }
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
                if let (Some(dir), Some(d), Some(r)) = (&config.cache_dir, det.content_hash, rec.content_hash) {
                    prune_cache(dir, &[d, r]);
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_keys_differ_by_shape_and_satisfy_onnx_runtime() {
        let a = cache_key(0xdead_beef, &[1, 3, 48, 320]);
        let b = cache_key(0xdead_beef, &[1, 3, 48, 640]);
        assert_ne!(a, b);
        assert!(a.len() <= 64 && a.chars().all(|c| c.is_ascii_alphanumeric()));
    }
    #[test]
    fn pruning_drops_stale_models_and_the_oldest_entries_only() {
        let dir = std::env::temp_dir().join(format!("ppocr-prune-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        let mut current = Vec::new();
        for i in 0..CACHE_ENTRIES + 2 {
            let key = cache_key(1, &[1, 3, 32, 32 * (i + 1)]);
            std::fs::create_dir(dir.join(&key)).unwrap();
            if i < 2 {
                set_modified(&dir.join(&key), old).unwrap();
            }
            current.push(key);
        }
        let stale = cache_key(2, &[1, 3, 48, 320]);
        std::fs::create_dir(dir.join(&stale)).unwrap();
        std::fs::create_dir(dir.join("not-ours")).unwrap();
        prune_cache(&dir, &[1]);
        let exists = |k: &str| dir.join(k).exists();
        assert!(!exists(&stale) && exists("not-ours"));
        assert!(!exists(&current[0]) && !exists(&current[1]));
        assert!(current[2..].iter().all(|k| exists(k)));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
