//! Does building DirectML sessions on several threads cut load time?
//!
//! parallel_build REC.onnx [DEVICE]
//!
//! Builds the recognizer's three warm-up widths (320, 640, 960) pinned, first one after another
//! and then all at once on their own threads, and checks each session still runs.
use ndarray::Array4;
use ort::{
    execution_providers::DirectMLExecutionProvider,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use std::time::Instant;

fn build(bytes: &[u8], device: i32, width: i64) -> ort::Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_op_spinning(false)?
        .with_parallel_execution(false)?
        .with_memory_pattern(false)?
        .with_dimension_override("DynamicDimension.0", 1)?
        .with_dimension_override("DynamicDimension.1", width)?
        .with_execution_providers([DirectMLExecutionProvider::default().with_device_id(device).build().error_on_failure()])?
        .commit_from_memory(bytes)
}

fn check(s: &mut Session, width: usize) -> f64 {
    let x = Tensor::from_array(Array4::<f32>::zeros((1, 3, 48, width))).unwrap();
    let t = Instant::now();
    let out = s.run(ort::inputs!["x" => x]).unwrap();
    let _ = out[0].try_extract_tensor::<f32>().unwrap();
    t.elapsed().as_secs_f64() * 1000.0
}

fn main() -> ort::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let device: i32 = a.get(1).and_then(|d| d.parse().ok()).unwrap_or(0);
    let bytes = std::sync::Arc::new(std::fs::read(&a[0]).expect("rec model"));
    let widths = [320usize, 640, 960];
    for round in 0..2 {
        let t = Instant::now();
        let mut sessions: Vec<Session> = widths.iter().map(|&w| build(&bytes, device, w as i64)).collect::<ort::Result<_>>()?;
        let sequential = t.elapsed().as_secs_f64() * 1000.0;
        let runs: Vec<f64> = sessions.iter_mut().zip(widths).map(|(s, w)| check(s, w)).collect();
        println!("round {round} sequential: {sequential:.0} ms, first runs {runs:.1?}");
        drop(sessions);

        let t = Instant::now();
        let handles: Vec<_> = widths
            .iter()
            .map(|&w| {
                let bytes = bytes.clone();
                std::thread::spawn(move || build(&bytes, device, w as i64))
            })
            .collect();
        let mut sessions: Vec<Session> = handles.into_iter().map(|h| h.join().unwrap()).collect::<ort::Result<_>>()?;
        let parallel = t.elapsed().as_secs_f64() * 1000.0;
        let runs: Vec<f64> = sessions.iter_mut().zip(widths).map(|(s, w)| check(s, w)).collect();
        println!("round {round} parallel:   {parallel:.0} ms, first runs {runs:.1?}");
    }
    Ok(())
}
