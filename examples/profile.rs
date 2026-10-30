//! Where the time goes: detection alone, then the full pipeline, per image.
//!
//! profile MODEL_DIR TIER BACKEND RUNS IMAGE...   (PPOCR_MAX_SIDE, PPOCR_DEVICE, PPOCR_BATCH)
use ppocr::{Backend, Config, ModelPaths, OcrOptions, PpOcr, Tier};
use std::time::Instant;

fn median(runs: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut ms: Vec<f64> = (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    ms.sort_by(f64::total_cmp);
    ms[ms.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, tier, backend, runs, images @ ..] = args.as_slice() else {
        return Err("usage: profile MODEL_DIR TIER BACKEND RUNS IMAGE...".into());
    };
    fn env<T: std::str::FromStr>(k: &str) -> Option<T> {
        std::env::var(k).ok().and_then(|v| v.parse().ok())
    }
    let tier = match tier.as_str() {
        "small" => Tier::Small,
        "tiny" => Tier::Tiny,
        _ => Tier::Medium,
    };
    let backend = match backend.as_str() {
        "cpu" => Backend::Cpu,
        "directml" => Backend::DirectMl,
        "cuda" => Backend::Cuda,
        _ => Backend::Auto,
    };
    let runs: usize = runs.parse()?;
    let mut ocr = PpOcr::with_config(
        ModelPaths::from_dir(dir, tier),
        Config {
            backend,
            detection_max_side: env("PPOCR_MAX_SIDE").unwrap_or(960),
            device_id: env("PPOCR_DEVICE"),
            max_batch_size: env("PPOCR_BATCH").unwrap_or(8),
            ..Default::default()
        },
    )?;
    println!("{:?} {}", ocr.backend_info().selected, tier.name());
    if std::env::var_os("COLD").is_some() {
        // First touch of each image right after load: includes building any new shape sessions.
        for path in images {
            let image = image::open(path)?.to_rgb8();
            let t = Instant::now();
            let lines = ocr.ocr(&image, OcrOptions::default())?.len();
            println!("cold {path}: {lines} lines {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
        }
    }
    for path in images {
        let image = image::open(path)?.to_rgb8();
        let options = OcrOptions::default();
        let boxes = ocr.detect(&image, options)?.len();
        let det = median(runs, || {
            ocr.detect(&image, options).unwrap();
        });
        let full = median(runs, || {
            ocr.ocr(&image, options).unwrap();
        });
        let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy();
        println!(
            "{name:>10} {}x{} lines {boxes:2}  det {det:6.1} ms  rec {:6.1} ms  total {full:6.1} ms",
            image.width(),
            image.height(),
            full - det
        );
    }
    Ok(())
}
