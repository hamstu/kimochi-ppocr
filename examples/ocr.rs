use ppocr::{Backend, Config, ModelPaths, OcrOptions, PpOcr, Tier};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let models = args
        .next()
        .ok_or("usage: ocr MODEL_DIR IMAGE [cpu|auto|coreml|cuda|directml] [medium|small|tiny]")?;
    let image = args.next().ok_or("missing image path")?;
    let backend = match args.next().as_deref().unwrap_or("auto") {
        "cpu" => Backend::Cpu,
        "auto" => Backend::Auto,
        "coreml" => Backend::CoreMl,
        "cuda" => Backend::Cuda,
        "directml" => Backend::DirectMl,
        _ => return Err("unknown backend".into()),
    };
    let tier = match args.next().as_deref().unwrap_or("medium") {
        "medium" => Tier::Medium,
        "small" => Tier::Small,
        "tiny" => Tier::Tiny,
        _ => return Err("unknown tier".into()),
    };
    let mut ocr = PpOcr::with_config(
        ModelPaths::from_dir(models, tier),
        Config {
            backend,
            ..Default::default()
        },
    )?;
    eprintln!("{:?}", ocr.backend_info());
    println!(
        "{}",
        serde_json::to_string_pretty(&ocr.ocr_file(image, OcrOptions::default())?)?
    );
    Ok(())
}
