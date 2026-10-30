# ppocr (Rust)

Native inference for PaddleOCR's [PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR) text
detection and recognition models through ONNX Runtime, with no Python, Paddle or OpenCV. The
output types match [kimochi-meikiocr](https://github.com/hamstu/kimochi-meikiocr)'s (`TextLine`,
`Character`, `BoundingBox`), including per-character boxes, so hosts can treat the two alike.

Used by [Kimochi](https://github.com/hamstu/kimochi-tauri).

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

## Models

Not included. Download a tier's ONNX exports from Hugging Face (Apache-2.0) and store them under
distinct names, since PaddlePaddle names every file `inference.onnx`:

| Store as | From |
| --- | --- |
| `PP-OCRv6_<tier>_det.onnx` | `PaddlePaddle/PP-OCRv6_<tier>_det_onnx` → `inference.onnx` |
| `PP-OCRv6_<tier>_rec.onnx` | `PaddlePaddle/PP-OCRv6_<tier>_rec_onnx` → `inference.onnx` |
| `PP-OCRv6_<tier>_rec.yml` | `PaddlePaddle/PP-OCRv6_<tier>_rec_onnx` → `inference.yml` (the character dictionary) |

`<tier>` is `medium`, `small` or `tiny`. At run time, point `ORT_DYLIB_PATH` at an ONNX Runtime
1.22+ shared library (or call `ort::init_from`). Features `coreml`, `cuda` and `directml` (all on
by default) enable those execution providers, with automatic fallback to CPU.

## How it runs

- **Detection** follows PaddleOCR's `DetResizeForTest` (longest side limited to
  `Config::detection_max_side`, default 960, multiples of 32) and `DBPostProcess`, with
  axis-aligned boxes: for an axis-aligned kernel, Paddle's minimum-area rectangle and polygon
  offset reduce exactly to growing the rectangle by `area × unclip_ratio / perimeter`.
- **Recognition** crops each box (rotating crops at least 1.5× taller than wide, so vertical text
  reads left to right), resizes to height 48, and decodes CTC greedily. Character boxes come from
  the time steps where each character peaks.
- **Compact recognizer outputs.** At load, the recognizer graph is rewritten in memory (`onnx.rs`)
  to end in `ArgMax` and `ReduceMax` over the class axis, so a run returns each step's best class
  and probability instead of 18,710 probabilities per step. The files on disk are unchanged; if
  the rewrite is refused, the stock graph is used.
- **DirectML and Core ML get one session per input shape.** ONNX Runtime 1.22's DirectML provider
  runs a dynamic-shape session's first input shape at full speed, then takes a 3–4× slower path
  for every run once the session has seen a second shape. Pinned sessions stay fast, so the
  detector gets one session per capture size (a game's capture size rarely changes) and the
  recognizer reads one line per run at a few bucketed widths. CPU and CUDA keep one dynamic
  session.

## Performance (RTX 3080 Ti, ORT 1.22 DirectML, warm)

Six rendered Japanese game-text fixtures from Kimochi (dialogue, katakana, large, pixel, small,
vertical); normalised character error rate ignores whitespace and full-width/half-width ASCII.

| Engine | Mean per frame | Normalised CER |
| --- | --- | --- |
| MeikiOCR (DirectML) | 85–107 ms | 6.9% |
| PP-OCRv6 small | 22 ms | 0.0% |
| PP-OCRv6 medium | 35 ms | 0.0% |

On CPU (Ryzen 7 7800X3D), small takes ~110–130 ms and medium ~390 ms, against MeikiOCR's ~70–80 ms.
A capture size or line width seen for the first time on DirectML costs one session build
(~150–250 ms).

```sh
cargo test
cargo run --release --example ocr -- models screenshot.png auto small
cargo run --release --example compare -- MEIKI_DIR PPOCR_DIR small directml 20 fixture.png ...
cargo run --release --example profile -- PPOCR_DIR small directml 15 fixture.png ...
```

## Licence

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)), matching PaddleOCR.
