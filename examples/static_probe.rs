//! DirectML run time of a dynamic-shape session against one with every dimension pinned.
//!
//! static_probe DET.onnx REC.onnx DEVICE
//!
//! With `PRIME` set, each dynamic session first runs one other shape. That is what exposes ONNX
//! Runtime 1.22's DirectML behaviour: a dynamic session is fast for the first shape it sees and
//! 3–4× slower for every run once it has seen a second one. `REC=1x320,4x960` picks the
//! recognizer shapes to time.
use ndarray::Array4;
use ort::{
    execution_providers::DirectMLExecutionProvider,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use std::time::Instant;

fn build(path: &str, device: i32, dims: &[(&str, i64)]) -> ort::Result<Session> {
    let mut b = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_parallel_execution(false)?
        .with_memory_pattern(false)?;
    for (name, v) in dims {
        b = b.with_dimension_override(*name, *v)?;
    }
    b.with_execution_providers([DirectMLExecutionProvider::default().with_device_id(device).build()])?
        .commit_from_file(path)
}

/// Median of 10 timed runs after 2 warm-ups.
fn time(s: &mut Session, shape: (usize, usize, usize, usize)) -> f64 {
    let mut ms = vec![];
    for i in 0..12 {
        let x = Tensor::from_array(Array4::<f32>::zeros(shape)).unwrap();
        let t = Instant::now();
        let out = s.run(ort::inputs!["x" => x]).unwrap();
        let _ = out[0].try_extract_tensor::<f32>().unwrap();
        if i > 1 {
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
    }
    ms.sort_by(f64::total_cmp);
    ms[ms.len() / 2]
}

fn main() -> ort::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let device: i32 = a[2].parse().unwrap();
    let prime = std::env::var_os("PRIME").is_some();
    for (h, w) in [(416usize, 960usize), (544, 960), (736, 1280)] {
        let mut dynamic = build(&a[0], device, &[])?;
        if prime {
            time(&mut dynamic, (1, 3, 64, 64));
        }
        let dims = [("DynamicDimension.0", 1), ("DynamicDimension.1", h as i64), ("DynamicDimension.2", w as i64)];
        let mut fixed = build(&a[0], device, &dims)?;
        let (d, f) = (time(&mut dynamic, (1, 3, h, w)), time(&mut fixed, (1, 3, h, w)));
        println!("det {h}x{w}: dynamic {d:.1} ms | static {f:.1} ms");
    }
    let shapes: Vec<(usize, usize)> = std::env::var("REC")
        .map(|v| {
            v.split(',')
                .map(|p| {
                    let (n, w) = p.split_once('x').unwrap();
                    (n.parse().unwrap(), w.parse().unwrap())
                })
                .collect()
        })
        .unwrap_or(vec![(1, 320), (8, 640), (8, 960)]);
    for (n, w) in shapes {
        let mut dynamic = build(&a[1], device, &[])?;
        if prime {
            time(&mut dynamic, (1, 3, 48, 320));
        }
        let mut fixed = build(&a[1], device, &[("DynamicDimension.0", n as i64), ("DynamicDimension.1", w as i64)])?;
        let (d, f) = (time(&mut dynamic, (n, 3, 48, w)), time(&mut fixed, (n, 3, 48, w)));
        println!("rec {n}x{w}: dynamic {d:.1} ms | static {f:.1} ms");
    }
    Ok(())
}
