//! Print a model's inputs and outputs: inspect MODEL.onnx...
fn main() -> Result<(), Box<dyn std::error::Error>> {
    for path in std::env::args().skip(1) {
        let session = ort::session::Session::builder()?.commit_from_file(&path)?;
        println!("{path}");
        for i in &session.inputs {
            println!("  in  {} {:?}", i.name, i.input_type);
        }
        for o in &session.outputs {
            println!("  out {} {:?}", o.name, o.output_type);
        }
    }
    Ok(())
}
