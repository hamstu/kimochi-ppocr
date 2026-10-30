//! Side-by-side speed and accuracy against MeikiOCR.
//!
//! compare MEIKI_DIR PPOCR_DIR TIER BACKEND RUNS IMAGE...
//!
//! For each IMAGE, a sibling `<stem>.truth.json` (Kimochi's overlay-alignment fixtures) supplies
//! the expected lines. Character error rate is reported raw and normalised (whitespace removed,
//! full-width ASCII folded), since a dictionary lookup treats those forms alike.
use meikiocr::{MeikiOcr, ModelPaths as MeikiPaths};
use ppocr::{Backend, Config, ModelPaths, OcrOptions, PpOcr, Tier};
use std::time::Instant;

fn backend(name: &str) -> Backend {
    match name {
        "cpu" => Backend::Cpu,
        "directml" => Backend::DirectMl,
        "cuda" => Backend::Cuda,
        "coreml" => Backend::CoreMl,
        _ => Backend::Auto,
    }
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
fn cer(truth: &str, got: &str) -> (f64, f64) {
    let raw = |s: &str| s.chars().filter(|c| *c != '\n').collect::<Vec<_>>();
    let (t, g) = (raw(truth), raw(got));
    let (nt, ng) = (normalise(truth), normalise(got));
    (
        distance(&t, &g) as f64 / t.len().max(1) as f64,
        distance(&nt, &ng) as f64 / nt.len().max(1) as f64,
    )
}
fn timed(runs: usize, mut f: impl FnMut() -> String) -> (String, f64, f64) {
    let text = f(); // warm-up, also the accuracy sample
    let mut ms: Vec<f64> = (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    ms.sort_by(f64::total_cmp);
    (text, ms[ms.len() / 2], ms[(ms.len() * 9 / 10).min(ms.len() - 1)])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [meiki_dir, pp_dir, tier, backend_name, runs, images @ ..] = args.as_slice() else {
        return Err("usage: compare MEIKI_DIR PPOCR_DIR TIER BACKEND RUNS IMAGE...".into());
    };
    let tier = match tier.as_str() {
        "small" => Tier::Small,
        "tiny" => Tier::Tiny,
        _ => Tier::Medium,
    };
    let backend = backend(backend_name);
    let runs: usize = runs.parse()?;
    let max_side: u32 = std::env::var("PPOCR_MAX_SIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(960);
    let t = Instant::now();
    let mut meiki = MeikiOcr::with_config(
        MeikiPaths::from_dir(meiki_dir),
        meikiocr::Config { backend: meiki_backend(backend), ..Default::default() },
    )?;
    let meiki_load = t.elapsed().as_millis();
    let t = Instant::now();
    let mut pp = PpOcr::with_config(
        ModelPaths::from_dir(pp_dir, tier),
        Config {
            backend,
            detection_max_side: max_side,
            device_id: std::env::var("PPOCR_DEVICE").ok().and_then(|v| v.parse().ok()),
            ..Default::default()
        },
    )?;
    let pp_load = t.elapsed().as_millis();
    println!(
        "meiki {:?} load {meiki_load} ms | ppocr {} {:?} max_side {max_side} load {pp_load} ms",
        meiki.backend_info().selected,
        tier.name(),
        pp.backend_info().selected
    );
    if !pp.backend_info().fallback_reasons.is_empty() {
        println!("ppocr fallback: {:?}", pp.backend_info().fallback_reasons);
    }
    let verbose = std::env::var("COMPARE_VERBOSE").is_ok();
    let mut totals = [0.0f64; 6];
    for path in images {
        let image = image::open(path)?.to_rgb8();
        let truth_path = path.replace(".png", ".truth.json");
        let truth: Option<String> = std::fs::read_to_string(&truth_path).ok().map(|s| {
            let v: serde_json::Value = serde_json::from_str(&s).unwrap();
            v["lines"].as_array().unwrap().iter()
                .map(|l| l["text"].as_str().unwrap_or_default().to_string())
                .collect::<Vec<_>>().join("\n")
        });
        let (mt, mmed, mp90) = timed(runs, || {
            meiki.ocr(&image, meikiocr::OcrOptions::default()).unwrap()
                .iter().map(|l| l.text.clone()).filter(|t| !t.is_empty()).collect::<Vec<_>>().join("\n")
        });
        let (pt, pmed, pp90) = timed(runs, || {
            let mut lines = pp.ocr(&image, OcrOptions::default()).unwrap();
            ppocr::japanese::fix_lines(&mut lines);
            lines
                .iter().map(|l| l.text.clone()).collect::<Vec<_>>().join("\n")
        });
        let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy();
        print!("{name:>10} {}x{}  meiki {mmed:6.1} ms (p90 {mp90:6.1})  ppocr {pmed:6.1} ms (p90 {pp90:6.1})", image.width(), image.height());
        if let Some(truth) = &truth {
            let (mr, mn) = cer(truth, &mt);
            let (pr, pn) = cer(truth, &pt);
            print!("  CER meiki {:5.1}%/{:5.1}%  ppocr {:5.1}%/{:5.1}%", mr * 100.0, mn * 100.0, pr * 100.0, pn * 100.0);
            for (k, v) in [mmed, pmed, mr, pr, mn, pn].into_iter().enumerate() {
                totals[k] += v;
            }
        }
        println!();
        if verbose {
            if let Some(truth) = &truth {
                println!("  truth: {}", truth.replace('\n', " | "));
            }
            println!("  meiki: {}", mt.replace('\n', " | "));
            println!("  ppocr: {}", pt.replace('\n', " | "));
        }
    }
    let n = images.len() as f64;
    println!(
        "mean: meiki {:.1} ms, ppocr {:.1} ms | CER raw meiki {:.1}% ppocr {:.1}% | normalised meiki {:.1}% ppocr {:.1}%",
        totals[0] / n, totals[1] / n, totals[2] / n * 100.0, totals[3] / n * 100.0, totals[4] / n * 100.0, totals[5] / n * 100.0
    );
    Ok(())
}
