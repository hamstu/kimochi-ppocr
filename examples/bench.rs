//! The repeatable benchmark every platform runs, so numbers compare across machines.
//!
//! bench MODEL_DIR TIER BACKEND RUNS IMAGE...
//!
//! Environment: `PPOCR_MAX_SIDE` (detector input, default 960), `PPOCR_DEVICE` (adapter index),
//! `PPOCR_THREADS` (CPU intra-op threads), `PPOCR_BATCH` (dynamic-shape batch size), `BENCH_LABEL`
//! (a free-text tag copied into the JSON line), `BENCH_JSON` (write one JSON object per
//! configuration, as a line, to this file, appending).
//!
//! Reports, per configuration:
//! - cold start: process start to the engine being loaded (sessions built, warm-up shapes run),
//!   then the first OCR of every image right after load (new shapes still cost sessions);
//! - warm latency: p50 / p95 / max of `RUNS` full OCR passes (detect + recognize) per image,
//!   and the same across all images pooled;
//! - peak memory: the process's peak resident set, before and after;
//! - accuracy: character error rate against the sibling `<stem>.truth.json`, when present
//!   (Kimochi's `scripts/fixtures/overlay-alignment`), raw and normalised (whitespace removed,
//!   full-width ASCII folded), so a GPU backend's parity with the CPU can be checked.
use ppocr::{Backend, Config, ModelPaths, OcrOptions, PpOcr, Tier};
use std::time::Instant;

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
/// (raw, normalised) character error rate of `got` against `truth`.
fn cer(truth: &str, got: &str) -> (f64, f64) {
    let raw = |s: &str| s.chars().filter(|c| *c != '\n').collect::<Vec<_>>();
    let (t, g) = (raw(truth), raw(got));
    let (nt, ng) = (normalise(truth), normalise(got));
    (
        distance(&t, &g) as f64 / t.len().max(1) as f64,
        distance(&nt, &ng) as f64 / nt.len().max(1) as f64,
    )
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

/// Peak resident set size of this process, in MiB.
fn peak_rss_mib() -> f64 {
    #[cfg(windows)]
    {
        #[repr(C)]
        #[allow(non_snake_case)]
        struct ProcessMemoryCounters {
            cb: u32,
            PageFaultCount: u32,
            PeakWorkingSetSize: usize,
            WorkingSetSize: usize,
            QuotaPeakPagedPoolUsage: usize,
            QuotaPagedPoolUsage: usize,
            QuotaPeakNonPagedPoolUsage: usize,
            QuotaNonPagedPoolUsage: usize,
            PagefileUsage: usize,
            PeakPagefileUsage: usize,
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> isize;
            fn K32GetProcessMemoryInfo(process: isize, counters: *mut ProcessMemoryCounters, cb: u32) -> i32;
        }
        let mut counters = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            PageFaultCount: 0,
            PeakWorkingSetSize: 0,
            WorkingSetSize: 0,
            QuotaPeakPagedPoolUsage: 0,
            QuotaPagedPoolUsage: 0,
            QuotaPeakNonPagedPoolUsage: 0,
            QuotaNonPagedPoolUsage: 0,
            PagefileUsage: 0,
            PeakPagefileUsage: 0,
        };
        // SAFETY: the struct is the documented PROCESS_MEMORY_COUNTERS layout and `cb` is its size.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
        if ok != 0 {
            return counters.PeakWorkingSetSize as f64 / (1024.0 * 1024.0);
        }
        f64::NAN
    }
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        status
            .lines()
            .find_map(|l| l.strip_prefix("VmHWM:"))
            .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<f64>().ok())
            .map(|kb| kb / 1024.0)
            .unwrap_or(f64::NAN)
    }
    #[cfg(target_os = "macos")]
    {
        #[repr(C)]
        struct Timeval {
            tv_sec: i64,
            tv_usec: i32,
        }
        #[repr(C)]
        struct Rusage {
            ru_utime: Timeval,
            ru_stime: Timeval,
            ru_maxrss: i64,
            rest: [i64; 13],
        }
        unsafe extern "C" {
            fn getrusage(who: i32, usage: *mut Rusage) -> i32;
        }
        let mut usage = Rusage { ru_utime: Timeval { tv_sec: 0, tv_usec: 0 }, ru_stime: Timeval { tv_sec: 0, tv_usec: 0 }, ru_maxrss: 0, rest: [0; 13] };
        // SAFETY: RUSAGE_SELF is 0 and the struct matches macOS's `struct rusage` layout.
        if unsafe { getrusage(0, &mut usage) } == 0 {
            return usage.ru_maxrss as f64 / (1024.0 * 1024.0); // bytes on macOS
        }
        f64::NAN
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        f64::NAN
    }
}

fn env<T: std::str::FromStr>(k: &str) -> Option<T> {
    std::env::var(k).ok().and_then(|v| v.parse().ok())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, tier_name, backend_name, runs, images @ ..] = args.as_slice() else {
        return Err("usage: bench MODEL_DIR TIER BACKEND RUNS IMAGE...".into());
    };
    if images.is_empty() {
        return Err("at least one IMAGE is required".into());
    }
    let tier = match tier_name.as_str() {
        "small" => Tier::Small,
        "tiny" => Tier::Tiny,
        _ => Tier::Medium,
    };
    let backend = match backend_name.as_str() {
        "cpu" => Backend::Cpu,
        "directml" => Backend::DirectMl,
        "cuda" => Backend::Cuda,
        "coreml" => Backend::CoreMl,
        _ => Backend::Auto,
    };
    let runs: usize = runs.parse()?;
    let config = Config {
        backend,
        detection_max_side: env("PPOCR_MAX_SIDE").unwrap_or(960),
        device_id: env("PPOCR_DEVICE"),
        intra_threads: env("PPOCR_THREADS"),
        max_batch_size: env("PPOCR_BATCH").unwrap_or(8),
    };
    let label = std::env::var("BENCH_LABEL").unwrap_or_default();

    let rss_before = peak_rss_mib();
    let load_start = Instant::now();
    let mut ocr = PpOcr::with_config(ModelPaths::from_dir(dir, tier), config.clone())?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let info = ocr.backend_info().clone();
    println!(
        "ppocr {} requested {:?} selected {:?} max_side {} threads {:?} batch {} | load {load_ms:.0} ms (process {:.0} ms)",
        tier.name(),
        info.requested,
        info.selected,
        config.detection_max_side,
        config.intra_threads,
        config.max_batch_size,
        started.elapsed().as_secs_f64() * 1000.0,
    );
    for reason in &info.fallback_reasons {
        println!("  fallback: {reason}");
    }

    // Cold: the first OCR of each image right after load.
    let decoded: Vec<_> = images
        .iter()
        .map(|p| image::open(p).map(|i| i.to_rgb8()).map_err(|e| format!("{p}: {e}")))
        .collect::<Result<_, _>>()?;
    let mut cold = Vec::with_capacity(images.len());
    let mut texts = Vec::with_capacity(images.len());
    for image in &decoded {
        let t = Instant::now();
        let mut lines = ocr.ocr(image, OcrOptions::default())?;
        cold.push(t.elapsed().as_secs_f64() * 1000.0);
        ppocr::japanese::fix_lines(&mut lines);
        texts.push(lines.iter().map(|l| l.text.clone()).collect::<Vec<_>>().join("\n"));
    }
    let first_scan_ms = cold[0];

    // Warm: `runs` full passes per image.
    let mut per_image = Vec::with_capacity(images.len());
    let mut pooled = Vec::with_capacity(images.len() * runs);
    for image in &decoded {
        let mut ms: Vec<f64> = (0..runs)
            .map(|_| {
                let t = Instant::now();
                ocr.ocr(image, OcrOptions::default()).unwrap();
                t.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        pooled.extend_from_slice(&ms);
        ms.sort_by(f64::total_cmp);
        per_image.push(ms);
    }
    pooled.sort_by(f64::total_cmp);
    let rss_after = peak_rss_mib();

    println!("{:>10} {:>9} {:>5} {:>8} {:>8} {:>8} {:>8}  {}", "image", "size", "lines", "cold", "p50", "p95", "max", "CER raw/norm");
    let mut cer_sum = (0.0, 0.0);
    let mut cer_n = 0usize;
    let mut rows = Vec::new();
    for (i, path) in images.iter().enumerate() {
        let image = &decoded[i];
        let ms = &per_image[i];
        let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().into_owned();
        let truth_path = std::path::Path::new(path).with_extension("truth.json");
        let truth: Option<String> = std::fs::read_to_string(&truth_path).ok().and_then(|s| {
            let v: serde_json::Value = serde_json::from_str(&s).ok()?;
            Some(
                v["lines"]
                    .as_array()?
                    .iter()
                    .map(|l| l["text"].as_str().unwrap_or_default().to_string())
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        });
        let lines = texts[i].lines().count();
        let (p50, p95, max) = (percentile(ms, 50.0), percentile(ms, 95.0), ms[ms.len() - 1]);
        print!(
            "{name:>10} {:>4}x{:<4} {lines:>5} {:>8.1} {p50:>8.1} {p95:>8.1} {max:>8.1}",
            image.width(),
            image.height(),
            cold[i]
        );
        let mut row = serde_json::json!({
            "image": name, "width": image.width(), "height": image.height(), "lines": lines,
            "cold_ms": cold[i], "p50_ms": p50, "p95_ms": p95, "max_ms": max,
        });
        if let Some(truth) = &truth {
            let (raw, norm) = cer(truth, &texts[i]);
            print!("  {:5.1}% / {:5.1}%", raw * 100.0, norm * 100.0);
            cer_sum.0 += raw;
            cer_sum.1 += norm;
            cer_n += 1;
            row["cer_raw"] = raw.into();
            row["cer_norm"] = norm.into();
        }
        println!();
        if std::env::var_os("BENCH_VERBOSE").is_some() {
            println!("      text: {}", texts[i].replace('\n', " | "));
        }
        rows.push(row);
    }
    let (p50, p95) = (percentile(&pooled, 50.0), percentile(&pooled, 95.0));
    let mean_cer = (cer_n > 0).then(|| (cer_sum.0 / cer_n as f64, cer_sum.1 / cer_n as f64));
    print!(
        "summary: load {load_ms:.0} ms, first scan {first_scan_ms:.0} ms, warm p50 {p50:.1} ms p95 {p95:.1} ms over {} runs, peak RSS {rss_after:.0} MiB (before load {rss_before:.0})",
        pooled.len()
    );
    if let Some((raw, norm)) = mean_cer {
        print!(", mean CER {:.1}% / {:.1}%", raw * 100.0, norm * 100.0);
    }
    println!();

    if let Ok(path) = std::env::var("BENCH_JSON") {
        let record = serde_json::json!({
            "label": label,
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "tier": tier.name(),
            "requested": format!("{:?}", info.requested),
            "selected": format!("{:?}", info.selected),
            "fallback": info.fallback_reasons,
            "max_side": config.detection_max_side,
            "threads": config.intra_threads,
            "batch": config.max_batch_size,
            "runs": runs,
            "load_ms": load_ms,
            "first_scan_ms": first_scan_ms,
            "p50_ms": p50,
            "p95_ms": p95,
            "peak_rss_mib": rss_after,
            "mean_cer_raw": mean_cer.map(|c| c.0),
            "mean_cer_norm": mean_cer.map(|c| c.1),
            "images": rows,
        });
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(file, "{record}")?;
    }
    Ok(())
}
