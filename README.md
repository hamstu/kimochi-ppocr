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

At load, the recognizer graph is rewritten in memory to end in `ArgMax` and `ReduceMax` over the class axis, so each run returns one class and score per time step instead of 18,710 probabilities. Downloaded files are not changed. On DirectML and Core ML, every input shape gets its own pinned session: ONNX Runtime 1.22's DirectML path runs a dynamic session's first shape at full speed, then 3–4× slower once it has seen a second shape.

Core ML compiles each of those shapes on first use, seconds per recognizer width. Set `Config::cache_dir` to a directory the crate can own and the compiled models are kept there, keyed by model content and input shape, so later loads skip compilation; the least recently built beyond 24 entries, and entries for other model files, are deleted at load.

On an RTX 3080 Ti (DirectML), the small tier reads a frame of game text in about 22 ms, against MeikiOCR's 85–107 ms. On CPU it is slower than MeikiOCR (about 110–130 ms against 70–80 ms). On an Apple M5 Max (Core ML on the GPU) it reads one in about 10 ms against MeikiOCR's 42 ms; loading takes 6.4 s the first time and 1.7 s from the cache.

`examples/bench.rs` is the cross-machine benchmark: cold start, warm p50/p95, peak memory and character error rate on Kimochi's overlay-alignment fixtures, for PP-OCR or MeikiOCR, ending in one `BENCH {json}` line. Run one process per configuration.

```sh
cargo test
cargo run --release --example ocr -- models screenshot.png auto small
cargo run --release --example bench -- --models models --backend cpu --threads 6 ../kimochi/scripts/fixtures/overlay-alignment/*.png
cargo run --release --example bench -- --models models --backend coreml --cache /tmp/ppocr-coreml ../kimochi/scripts/fixtures/overlay-alignment/*.png
```

## Licence

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)), matching PaddleOCR.
