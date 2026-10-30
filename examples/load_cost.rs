//! Where a backend's load time goes: the cost of building each session kind, by graph
//! optimisation level, and the first and steady run that follows.
//!
//! load_cost DET.onnx REC.onnx [cpu|directml|cuda] [DEVICE]
//!
//! Every session is built from the file bytes as the crate does (`commit_from_memory`), with the
//! crate's DirectML options (sequential execution, no memory pattern). Static sessions pin the
//! detector to 1x3x544x960 and the recognizer to 1x3x48x320.
use ndarray::Array4;
use ort::{
    execution_providers::ExecutionProviderDispatch,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use std::time::Instant;

fn provider(backend: &str, device: i32) -> Option<ExecutionProviderDispatch> {
    match backend {
        #[cfg(feature = "directml")]
        "directml" => Some(ort::execution_providers::DirectMLExecutionProvider::default().with_device_id(device).build().error_on_failure()),
        #[cfg(feature = "cuda")]
        "cuda" => Some(ort::execution_providers::CUDAExecutionProvider::default().with_device_id(device).build().error_on_failure()),
        _ => None,
    }
}

fn build(bytes: &[u8], backend: &str, device: i32, level: GraphOptimizationLevel, dims: &[(&str, i64)]) -> ort::Result<Session> {
    let mut b = Session::builder()?
        .with_optimization_level(level)?
        .with_intra_op_spinning(false)?
        .with_config_entry("session.inter_op.allow_spinning", "0")?;
    if backend == "directml" {
        b = b.with_parallel_execution(false)?.with_memory_pattern(false)?;
    }
    for (name, v) in dims {
        b = b.with_dimension_override(*name, *v)?;
    }
    if let Some(ep) = provider(backend, device) {
        b = b.with_execution_providers([ep])?;
    }
    b.commit_from_memory(bytes)
}

fn run_ms(s: &mut Session, shape: (usize, usize, usize, usize)) -> f64 {
    let x = Tensor::from_array(Array4::<f32>::zeros(shape)).unwrap();
    let t = Instant::now();
    let out = s.run(ort::inputs!["x" => x]).unwrap();
    let _ = out[0].try_extract_tensor::<f32>().unwrap();
    t.elapsed().as_secs_f64() * 1000.0
}

fn main() -> ort::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let backend = a.get(2).map(String::as_str).unwrap_or("directml");
    let device: i32 = a.get(3).and_then(|d| d.parse().ok()).unwrap_or(0);
    let det = std::fs::read(&a[0]).expect("det model");
    let rec = std::fs::read(&a[1]).expect("rec model");
    let levels = [
        ("disable", GraphOptimizationLevel::Disable),
        ("level1", GraphOptimizationLevel::Level1),
        ("level2", GraphOptimizationLevel::Level2),
        ("level3", GraphOptimizationLevel::Level3),
    ];
    println!("{backend} device {device}");
    println!("{:<8} {:<16} {:>10} {:>10} {:>10}", "level", "session", "build ms", "first ms", "steady ms");
    for (name, level) in levels {
        let level_of = || match name {
            "disable" => GraphOptimizationLevel::Disable,
            "level1" => GraphOptimizationLevel::Level1,
            "level2" => GraphOptimizationLevel::Level2,
            _ => GraphOptimizationLevel::Level3,
        };
        let _ = level;
        #[allow(clippy::type_complexity)]
        let cases: [(&str, &[u8], &[(&str, i64)], (usize, usize, usize, usize)); 4] = [
            ("det dynamic", &det, &[], (1, 3, 544, 960)),
            ("det 544x960", &det, &[("DynamicDimension.0", 1), ("DynamicDimension.1", 544), ("DynamicDimension.2", 960)], (1, 3, 544, 960)),
            ("rec dynamic", &rec, &[], (1, 3, 48, 320)),
            ("rec 1x320", &rec, &[("DynamicDimension.0", 1), ("DynamicDimension.1", 320)], (1, 3, 48, 320)),
        ];
        for (label, bytes, dims, shape) in cases {
            let t = Instant::now();
            let mut s = build(bytes, backend, device, level_of(), dims)?;
            let build_ms = t.elapsed().as_secs_f64() * 1000.0;
            let first = run_ms(&mut s, shape);
            let mut steady: Vec<f64> = (0..10).map(|_| run_ms(&mut s, shape)).collect();
            steady.sort_by(f64::total_cmp);
            println!("{name:<8} {label:<16} {build_ms:>10.0} {first:>10.1} {:>10.1}", steady[steady.len() / 2]);
        }
    }
    Ok(())
}
