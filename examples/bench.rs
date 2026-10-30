//! The common benchmark: cold start, warm latency and peak memory, one engine per process.
//!
//! bench [--engine ppocr|meiki] [--models DIR] [--tier small|medium|tiny]
//!       [--backend auto|cpu|cuda|coreml|directml] [--threads N] [--runs N] [--json] IMAGE...
//!
//! Cold start is process start to the first image's result: engine load (runtime, sessions and
//! any accelerator probe) plus that first `ocr`, less image decoding. Warm latency is every later
//! `ocr` call, `--runs` per image (default 20), as p50/p95 per image and over all images. Peak
//! memory is the process's peak resident set (unix only). Character error rate is reported when
//! an image has a sibling `<stem>.truth.json` (Kimochi's scripts/fixtures/overlay-alignment). The
//! last line, `BENCH {...}`, is the same numbers as JSON for collecting across machines.
//!
//! Use the fixed image set so numbers compare across machines, from a Kimochi checkout:
//!   bench --models MODEL_DIR scripts/fixtures/overlay-alignment/*.png
use ppocr::{Backend, Config, ModelPaths, OcrOptions, PpOcr, Tier};
use std::time::Instant;

fn backend(name: &str) -> Result<Backend, String> {
    Ok(match name {
        "auto" => Backend::Auto,
        "cpu" => Backend::Cpu,
        "cuda" => Backend::Cuda,
        "coreml" => Backend::CoreMl,
        "directml" => Backend::DirectMl,
        _ => return Err(format!("unknown backend {name}")),
    })
}
fn meiki_backend(b: Backend) -> meikiocr::Backend {
    match b {
        Backend::Cpu => meikiocr::Backend::Cpu,
        Backend::DirectMl => meikiocr::Backend::DirectMl,
        Backend::Cuda => meikiocr::Backend::Cuda,
        Backend::CoreMl => meikiocr::Backend::CoreMl,
        Backend::Auto => meikiocr::Backend::Auto,
    }
}

/// Either engine, reduced to "selected backend" and "image to text".
enum Engine {
    Pp(PpOcr),
    Meiki(meikiocr::MeikiOcr),
}
impl Engine {
    fn selected(&self) -> String {
        match self {
            Engine::Pp(e) => format!("{:?}", e.backend_info().selected),
            Engine::Meiki(e) => format!("{:?}", e.backend_info().selected),
        }
    }
    fn fallbacks(&self) -> Vec<String> {
        match self {
            Engine::Pp(e) => e.backend_info().fallback_reasons.clone(),
            Engine::Meiki(e) => e.backend_info().fallback_reasons.clone(),
        }
    }
    fn ocr(&mut self, image: &image::RgbImage) -> String {
        match self {
            Engine::Pp(e) => {
                let mut lines = e.ocr(image, OcrOptions::default()).unwrap();
                ppocr::japanese::fix_lines(&mut lines);
                lines.into_iter().map(|l| l.text).collect::<Vec<_>>().join("\n")
            }
            Engine::Meiki(e) => e
                .ocr(image, meikiocr::OcrOptions::default())
                .unwrap()
                .into_iter()
                .map(|l| l.text)
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

/// Whitespace removed and full-width ASCII folded, as a dictionary lookup treats them alike.
fn normalise(s: &str) -> Vec<char> {
    s.chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| match c as u32 {
            0xFF01..=0xFF5E => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
            _ => c,
        })
        .collect()
}
fn cer(truth: &str, got: &str) -> f64 {
    let (a, b) = (normalise(truth), normalise(got));
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let next = (prev + usize::from(ca != cb)).min(row[j] + 1).min(row[j + 1] + 1);
            prev = row[j + 1];
            row[j + 1] = next;
        }
    }
    row[b.len()] as f64 / a.len().max(1) as f64
}
fn truth(path: &str) -> Option<String> {
    let stem = path.strip_suffix(".png").unwrap_or(path);
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{stem}.truth.json")).ok()?).ok()?;
    Some(
        json["lines"]
            .as_array()?
            .iter()
            .filter_map(|l| l["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"),
    )
}
/// Nearest-rank percentile of sorted samples.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// The process's peak resident set in MiB.
#[cfg(unix)]
fn peak_rss_mib() -> Option<f64> {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return None;
    }
    // ru_maxrss is bytes on macOS, KiB elsewhere.
    let bytes = if cfg!(target_os = "macos") { 1.0 } else { 1024.0 };
    Some(usage.ru_maxrss as f64 * bytes / (1024.0 * 1024.0))
}
#[cfg(not(unix))]
fn peak_rss_mib() -> Option<f64> {
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let (mut engine_name, mut models, mut tier, mut backend_name) =
        ("ppocr".to_string(), "models".to_string(), "small".to_string(), "auto".to_string());
    let (mut threads, mut runs, mut json_only) = (None::<usize>, 20usize, false);
    let mut images = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--engine" => engine_name = value()?,
            "--models" => models = value()?,
            "--tier" => tier = value()?,
            "--backend" => backend_name = value()?,
            "--threads" => threads = Some(value()?.parse()?),
            "--runs" => runs = value()?.parse()?,
            "--json" => json_only = true,
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg}").into()),
            _ => images.push(arg),
        }
    }
    if images.is_empty() || runs == 0 {
        return Err("usage: bench [--engine ppocr|meiki] [--models DIR] [--tier T] [--backend B] \
                    [--threads N] [--runs N] [--json] IMAGE..."
            .into());
    }
    let backend = backend(&backend_name)?;
    let decode = Instant::now();
    let decoded: Vec<(String, image::RgbImage)> = images
        .iter()
        .map(|p| Ok::<_, image::ImageError>((p.clone(), image::open(p)?.to_rgb8())))
        .collect::<Result<_, _>>()?;
    let decode_ms = decode.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let mut engine = match engine_name.as_str() {
        "ppocr" => Engine::Pp(PpOcr::with_config(
            ModelPaths::from_dir(&models, match tier.as_str() {
                "medium" => Tier::Medium,
                "tiny" => Tier::Tiny,
                _ => Tier::Small,
            }),
            Config { backend, intra_threads: threads, ..Default::default() },
        )?),
        "meiki" => Engine::Meiki(meikiocr::MeikiOcr::with_config(
            meikiocr::ModelPaths::from_dir(&models),
            meikiocr::Config {
                backend: meiki_backend(backend),
                intra_threads: threads,
                ..Default::default()
            },
        )?),
        other => return Err(format!("unknown engine {other}").into()),
    };
    let load_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let first_text = engine.ocr(&decoded[0].1);
    let first_ms = t.elapsed().as_secs_f64() * 1000.0;
    let cold_ms = start.elapsed().as_secs_f64() * 1000.0 - decode_ms;

    let mut all = Vec::new();
    let mut per_image = Vec::new();
    let mut cers = Vec::new();
    for (i, (path, image)) in decoded.iter().enumerate() {
        // Each image's first pass is its accuracy sample and its warm-up (new shapes).
        let text = if i == 0 { first_text.clone() } else { engine.ocr(image) };
        let mut ms: Vec<f64> = (0..runs)
            .map(|_| {
                let t = Instant::now();
                engine.ocr(image);
                t.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        all.extend_from_slice(&ms);
        ms.sort_by(f64::total_cmp);
        let error = truth(path).map(|t| cer(&t, &text));
        cers.extend(error);
        let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().to_string();
        per_image.push(serde_json::json!({
            "image": name, "width": image.width(), "height": image.height(),
            "p50_ms": percentile(&ms, 50.0), "p95_ms": percentile(&ms, 95.0), "cer": error,
        }));
    }
    all.sort_by(f64::total_cmp);
    let mean_cer = (!cers.is_empty()).then(|| cers.iter().sum::<f64>() / cers.len() as f64);
    let peak = peak_rss_mib();
    let summary = serde_json::json!({
        "engine": engine_name,
        "tier": (engine_name == "ppocr").then_some(&tier),
        "requested": backend_name,
        "selected": engine.selected(),
        "fallbacks": engine.fallbacks(),
        "threads": threads,
        "runs": runs,
        "os": std::env::consts::OS, "arch": std::env::consts::ARCH,
        "load_ms": load_ms, "first_ms": first_ms, "cold_ms": cold_ms,
        "warm_p50_ms": percentile(&all, 50.0), "warm_p95_ms": percentile(&all, 95.0),
        "warm_mean_ms": all.iter().sum::<f64>() / all.len() as f64,
        "peak_rss_mib": peak,
        "mean_cer": mean_cer,
        "images": per_image,
    });
    if !json_only {
        println!(
            "{engine_name} {} (requested {backend_name}, threads {})",
            engine.selected(),
            threads.map_or("default".into(), |n| n.to_string())
        );
        for reason in engine.fallbacks() {
            println!("  fallback: {reason}");
        }
        println!("cold {cold_ms:.0} ms (load {load_ms:.0} + first image {first_ms:.0})");
        for row in &per_image {
            println!(
                "{:>10} {:>4}x{:<4} p50 {:7.1} ms  p95 {:7.1} ms  CER {}",
                row["image"].as_str().unwrap(),
                row["width"],
                row["height"],
                row["p50_ms"].as_f64().unwrap(),
                row["p95_ms"].as_f64().unwrap(),
                row["cer"].as_f64().map_or("-".into(), |c| format!("{:.1}%", c * 100.0)),
            );
        }
        println!(
            "warm p50 {:.1} ms  p95 {:.1} ms  | peak RSS {}  | mean CER {}",
            percentile(&all, 50.0),
            percentile(&all, 95.0),
            peak.map_or("n/a".into(), |m| format!("{m:.0} MiB")),
            mean_cer.map_or("-".into(), |c| format!("{:.1}%", c * 100.0)),
        );
    }
    println!("BENCH {summary}");
    Ok(())
}
