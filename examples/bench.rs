//! The cross-platform benchmark: one process, one configuration, comparable numbers.
//!
//! bench MODEL_DIR BACKEND RUNS IMAGE...
//!
//! BACKEND is auto, cpu, coreml, cuda or directml. Environment: PPOCR_TIER (small), PPOCR_MAX_SIDE
//! (960), PPOCR_BATCH (8), PPOCR_THREADS, PPOCR_DEVICE, PPOCR_CACHE (Core ML compiled-model cache
//! directory) and BENCH_LABEL (a name for this configuration in the summary).
//!
//! Reports, per image: the first call after load (which builds any per-shape session it needs),
//! then detection-only and full-pipeline p50/p95 over RUNS warm calls, and the character error
//! rate against a sibling `<stem>.truth.json` (Kimochi's overlay-alignment fixtures). Then the
//! load time, the mean warm latency and the process's peak resident memory, also as one JSON line
//! (`BENCH {...}`) for collecting runs across machines. Run a fresh process per configuration:
//! the load time is only cold the first time a process loads ONNX Runtime and the models.
use ppocr::{Backend, Config, ModelPaths, OcrOptions, PpOcr, Tier};
use std::time::Instant;

fn env<T: std::str::FromStr>(k: &str) -> Option<T> {
    std::env::var(k).ok().and_then(|v| v.parse().ok())
}

fn backend(name: &str) -> Result<Backend, String> {
    Ok(match name {
        "auto" => Backend::Auto,
        "cpu" => Backend::Cpu,
        "coreml" => Backend::CoreMl,
        "cuda" => Backend::Cuda,
        "directml" => Backend::DirectMl,
        _ => return Err(format!("unknown backend {name}")),
    })
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// p50 and p95 of `runs` timed calls.
fn percentiles(runs: usize, mut f: impl FnMut()) -> (f64, f64) {
    let mut samples: Vec<f64> = (0..runs.max(1))
        .map(|_| {
            let t = Instant::now();
            f();
            ms(t)
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    let at = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize];
    (at(0.5), at(0.95))
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

fn distance(a: &[char], b: &[char]) -> usize {
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
    row[b.len()]
}

fn truth(image: &str) -> Option<String> {
    let path = std::path::Path::new(image).with_extension("truth.json");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    Some(
        v["lines"]
            .as_array()?
            .iter()
            .filter_map(|l| l["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// Peak resident set size of this process, in MiB.
#[cfg(unix)]
fn peak_rss_mib() -> Option<f64> {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return None;
    }
    // ru_maxrss is bytes on macOS and KiB on Linux.
    let bytes = if cfg!(target_os = "macos") {
        usage.ru_maxrss as f64
    } else {
        usage.ru_maxrss as f64 * 1024.0
    };
    Some(bytes / (1024.0 * 1024.0))
}
#[cfg(not(unix))]
fn peak_rss_mib() -> Option<f64> {
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, backend_name, runs, images @ ..] = args.as_slice() else {
        return Err("usage: bench MODEL_DIR BACKEND RUNS IMAGE...".into());
    };
    let runs: usize = runs.parse()?;
    let tier = match std::env::var("PPOCR_TIER").as_deref() {
        Ok("medium") => Tier::Medium,
        Ok("tiny") => Tier::Tiny,
        _ => Tier::Small,
    };
    let config = Config {
        backend: backend(backend_name)?,
        detection_max_side: env("PPOCR_MAX_SIDE").unwrap_or(960),
        max_batch_size: env("PPOCR_BATCH").unwrap_or(8),
        intra_threads: env("PPOCR_THREADS"),
        device_id: env("PPOCR_DEVICE"),
        cache_dir: std::env::var_os("PPOCR_CACHE").map(Into::into),
    };
    let label = std::env::var("BENCH_LABEL").unwrap_or_else(|_| backend_name.clone());
    let decoded: Vec<_> = images
        .iter()
        .map(|p| Ok::<_, image::ImageError>((p.as_str(), image::open(p)?.to_rgb8())))
        .collect::<Result<_, _>>()?;

    let t = Instant::now();
    let mut ocr = PpOcr::with_config(ModelPaths::from_dir(dir, tier), config.clone())?;
    let load = ms(t);
    let info = ocr.backend_info().clone();
    println!(
        "{label}: {} tier, {:?} selected, load {load:.0} ms, max side {}, batch {}",
        tier.name(),
        info.selected,
        config.detection_max_side,
        config.max_batch_size
    );
    for reason in &info.fallback_reasons {
        println!("  fallback: {reason}");
    }

    let options = OcrOptions::default();
    let read = |ocr: &mut PpOcr, image| {
        let mut lines = ocr.ocr(image, options).unwrap();
        ppocr::japanese::fix_lines(&mut lines);
        lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n")
    };
    let mut first_total = 0.0;
    let mut cold_first = Vec::new();
    for (_, image) in &decoded {
        let t = Instant::now();
        let _ = ocr.ocr(image, options)?;
        cold_first.push(ms(t));
        first_total += ms(t);
    }
    println!(
        "{:>10} {:>9} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6}",
        "image", "size", "lines", "first", "det p50", "det p95", "p50", "p95", "CER"
    );
    let (mut p50_sum, mut p95_sum, mut cer_sum, mut cer_n) = (0.0, 0.0, 0.0, 0);
    for ((path, image), first) in decoded.iter().zip(&cold_first) {
        let text = read(&mut ocr, image);
        let (dp50, dp95) = percentiles(runs, || {
            ocr.detect(image, options).unwrap();
        });
        let (p50, p95) = percentiles(runs, || {
            ocr.ocr(image, options).unwrap();
        });
        p50_sum += p50;
        p95_sum += p95;
        let cer = truth(path).map(|t| {
            let (t, g) = (normalise(&t), normalise(&text));
            distance(&t, &g) as f64 / t.len().max(1) as f64 * 100.0
        });
        if let Some(c) = cer {
            cer_sum += c;
            cer_n += 1;
        }
        let name = std::path::Path::new(path).file_stem().unwrap_or_default().to_string_lossy();
        println!(
            "{name:>10} {:>9} {:>5} {first:8.1} {dp50:8.1} {dp95:8.1} {p50:8.1} {p95:8.1} {:>6}",
            format!("{}x{}", image.width(), image.height()),
            text.lines().count(),
            cer.map_or("-".into(), |c| format!("{c:.1}%")),
        );
        if std::env::var_os("BENCH_VERBOSE").is_some() {
            println!("            {}", text.replace('\n', " | "));
        }
    }
    let n = decoded.len().max(1) as f64;
    let rss = peak_rss_mib();
    let cer_mean = (cer_n > 0).then(|| cer_sum / cer_n as f64);
    println!(
        "{label}: load {load:.0} ms, first calls {first_total:.0} ms, warm mean p50 {:.1} ms p95 {:.1} ms, CER {}, peak RSS {}",
        p50_sum / n,
        p95_sum / n,
        cer_mean.map_or("-".into(), |c| format!("{c:.2}%")),
        rss.map_or("-".into(), |r| format!("{r:.0} MiB")),
    );
    println!(
        "BENCH {}",
        serde_json::json!({
            "label": label,
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "tier": tier.name(),
            "selected": format!("{:?}", info.selected),
            "load_ms": load.round(),
            "first_calls_ms": first_total.round(),
            "warm_p50_ms": (p50_sum / n * 10.0).round() / 10.0,
            "warm_p95_ms": (p95_sum / n * 10.0).round() / 10.0,
            "cer_percent": cer_mean.map(|c| (c * 100.0).round() / 100.0),
            "peak_rss_mib": rss.map(f64::round),
            "images": decoded.len(),
            "runs": runs,
        })
    );
    Ok(())
}
