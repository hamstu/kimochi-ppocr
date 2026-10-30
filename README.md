# ppocr (Rust)

Native inference for PaddleOCR's [PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0) text detection and recognition models through ONNX Runtime, with no Python, Paddle or OpenCV. The output types match [kimochi-meikiocr](https://github.com/hamstu/kimochi-meikiocr)'s, per-character boxes included.

Used by [Kimochi](https://github.com/hamstu/kimochi-tauri).

```toml
[dependencies]
ppocr = { git = "https://github.com/hamstu/kimochi-ppocr" }
```

```rust,no_run
use ppocr::{ModelPaths, OcrOptions, PpOcr, Tier};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ocr = PpOcr::new(ModelPaths::from_dir("models", Tier::Small))?;
    let image = image::open("screenshot.png")?.to_rgb8();
    let mut lines = ocr.ocr(&image, OcrOptions::default())?;
    ppocr::japanese::fix_lines(&mut lines); // optional, for Japanese text
    for line in lines {
        println!("{} {:?}", line.text, line.bbox);
    }
    Ok(())
}
```

Models are not included. PaddlePaddle publishes each tier (`medium`, `small` or `tiny`) on Hugging Face under Apache-2.0 as `PaddlePaddle/PP-OCRv6_<tier>_det_onnx` and `PaddlePaddle/PP-OCRv6_<tier>_rec_onnx`. Every file there is named `inference.*`, so store them as `PP-OCRv6_<tier>_det.onnx`, `PP-OCRv6_<tier>_rec.onnx` and `PP-OCRv6_<tier>_rec.yml` (the recognizer's `inference.yml`, which holds the character dictionary). At run time, point `ORT_DYLIB_PATH` at an ONNX Runtime 1.22+ shared library. Cargo features `coreml`, `cuda` and `directml` (all on by default) enable those execution providers, with automatic fallback to CPU.

At load, the recognizer graph is rewritten in memory to end in `ArgMax` and `ReduceMax` over the class axis, so each run returns one class and score per time step instead of 18,710 probabilities. Downloaded files are not changed. On DirectML and Core ML, every input shape gets its own pinned session: ONNX Runtime 1.22's DirectML path runs a dynamic session's first shape at full speed, then 2–4× slower once it has seen a second shape (measured on an RTX 3080 Ti and a GTX 1080). The dimension names those sessions pin come straight from the model file, the warm-up sessions are built on their own threads on DirectML (about 650 ms each on a GTX 1080, so this cuts the cold start from 2–4 s to under 1 s), and the detector keeps sessions for the last four capture sizes. With `Backend::Auto`, only providers the loaded ONNX Runtime was built with are tried.

On an RTX 3080 Ti (DirectML), the small tier reads a frame of game text in about 22 ms, against MeikiOCR's 85–107 ms; on a GTX 1080 about 40 ms against 135 ms. On CPU it is slower than MeikiOCR (about 110–130 ms against 70–80 ms on a recent desktop CPU; 235 ms against 155 ms on an i7-7700K).

```sh
cargo test
cargo run --release --example ocr -- models screenshot.png auto small
# The cross-platform benchmark: cold start, warm p50/p95, peak memory and accuracy against
# Kimochi's overlay-alignment fixtures (their `<name>.truth.json`), one JSON line per run.
BENCH_JSON=bench.jsonl cargo run --release --example bench -- models small auto 20 fixtures/*.png
```

`examples/load_cost.rs` shows what each session kind costs to build and run per graph optimisation level, and `examples/parallel_build.rs` whether concurrent session builds help on a given adapter.

## Licence

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)), matching PaddleOCR.
