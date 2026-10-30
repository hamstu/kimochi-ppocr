//! In-memory rewrite of the recognizer: append `ArgMax` and `ReduceMax` over the class axis, so a
//! run returns each time step's best class and its probability (`[N, T]` each) instead of every
//! class's (`[N, T, 18710]`, 75 KB per step). On DirectML that copy back from the GPU, and the CPU
//! scan over it, cost more than the recognizer itself. Downloaded files are never changed.
//!
//! Only the top-level fields of `ModelProto` and `GraphProto` are parsed; everything else is
//! copied verbatim.
use crate::{Error, Result};

fn invalid() -> Error {
    Error::ModelContract("invalid ONNX protobuf".into())
}

fn varint(bytes: &[u8], offset: &mut usize) -> Result<u64> {
    let mut value = 0;
    for shift in (0..64).step_by(7) {
        let byte = *bytes.get(*offset).ok_or_else(invalid)?;
        *offset += 1;
        value |= u64::from(byte & 127) << shift;
        if byte < 128 {
            return Ok(value);
        }
    }
    Err(invalid())
}

struct Field<'a> {
    number: u64,
    /// The payload of a length-delimited field, or the varint value's bytes otherwise.
    value: &'a [u8],
    /// The whole field, key included, for copying it unchanged.
    encoded: &'a [u8],
}

fn fields(bytes: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut offset = 0;
    let mut out = Vec::new();
    while offset < bytes.len() {
        let start = offset;
        let key = varint(bytes, &mut offset)?;
        let value_start = offset;
        let (value_start, end) = match key & 7 {
            0 => {
                varint(bytes, &mut offset)?;
                (value_start, offset)
            }
            1 => (value_start, value_start + 8),
            2 => {
                let len = varint(bytes, &mut offset)? as usize;
                (offset, offset.checked_add(len).ok_or_else(invalid)?)
            }
            5 => (value_start, value_start + 4),
            _ => return Err(invalid()),
        };
        if end > bytes.len() {
            return Err(invalid());
        }
        offset = end;
        out.push(Field {
            number: key >> 3,
            value: &bytes[value_start..end],
            encoded: &bytes[start..end],
        });
    }
    Ok(out)
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 128 {
        out.push((v as u8) | 128);
        v >>= 7;
    }
    out.push(v as u8);
}
fn put_bytes(out: &mut Vec<u8>, number: u64, bytes: &[u8]) {
    put_varint(out, number << 3 | 2);
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}
fn put_int(out: &mut Vec<u8>, number: u64, v: i64) {
    put_varint(out, number << 3);
    put_varint(out, v as u64);
}

fn int_attribute(name: &str, value: i64) -> Vec<u8> {
    let mut a = Vec::new();
    put_bytes(&mut a, 1, name.as_bytes());
    put_int(&mut a, 3, value);
    put_int(&mut a, 20, 2); // AttributeProto.INT
    a
}
fn ints_attribute(name: &str, values: &[i64]) -> Vec<u8> {
    let mut a = Vec::new();
    put_bytes(&mut a, 1, name.as_bytes());
    for &v in values {
        put_int(&mut a, 8, v);
    }
    put_int(&mut a, 20, 7); // AttributeProto.INTS
    a
}
fn node(op: &str, input: &str, output: &str, attributes: &[Vec<u8>]) -> Vec<u8> {
    let mut n = Vec::new();
    put_bytes(&mut n, 1, input.as_bytes());
    put_bytes(&mut n, 2, output.as_bytes());
    put_bytes(&mut n, 3, output.as_bytes());
    put_bytes(&mut n, 4, op.as_bytes());
    for a in attributes {
        put_bytes(&mut n, 5, a);
    }
    n
}
/// A graph output of `elem_type` (TensorProto.FLOAT = 1, INT64 = 7) with unspecified shape.
fn output(name: &str, elem_type: i64) -> Vec<u8> {
    let mut tensor = Vec::new();
    put_int(&mut tensor, 1, elem_type);
    let mut ty = Vec::new();
    put_bytes(&mut ty, 1, &tensor);
    let mut v = Vec::new();
    put_bytes(&mut v, 1, name.as_bytes());
    put_bytes(&mut v, 2, &ty);
    v
}

fn string(bytes: &[u8]) -> Result<&str> {
    std::str::from_utf8(bytes).map_err(|_| invalid())
}

/// The default-domain opset the model imports.
fn opset(model: &[Field<'_>]) -> Result<i64> {
    for f in model.iter().filter(|f| f.number == 8) {
        let set = fields(f.value)?;
        let domain = set.iter().find(|f| f.number == 1).map(|f| string(f.value)).transpose()?;
        if matches!(domain, None | Some("") | Some("ai.onnx"))
            && let Some(v) = set.iter().find(|f| f.number == 2) {
                let mut o = 0;
                return Ok(varint(v.value, &mut o)? as i64);
            }
    }
    Err(Error::ModelContract("model imports no default opset".into()))
}

/// The class count from a graph output's declared last dimension, when it is fixed.
fn last_dim(value_info: &[u8]) -> Result<Option<i64>> {
    let get = |bytes, n| -> Result<Option<&[u8]>> {
        Ok(fields(bytes)?.into_iter().find(|f| f.number == n).map(|f| f.value))
    };
    let Some(ty) = get(value_info, 2)? else { return Ok(None) };
    let Some(tensor) = get(ty, 1)? else { return Ok(None) };
    let Some(shape) = get(tensor, 2)? else { return Ok(None) };
    let Some(dim) = fields(shape)?.into_iter().rfind(|f| f.number == 1) else {
        return Ok(None);
    };
    match fields(dim.value)?.into_iter().find(|f| f.number == 1) {
        Some(v) => {
            let mut o = 0;
            Ok(Some(varint(v.value, &mut o)? as i64))
        }
        None => Ok(None),
    }
}

pub(crate) const CLASSES: &str = "ppocr_best_class";
pub(crate) const SCORES: &str = "ppocr_best_score";

/// The recognizer with compact outputs (`CLASSES` int64, `SCORES` float, both `[N, T]`),
/// and the class count the original output declared.
pub(crate) fn compact_recognizer(bytes: &[u8]) -> Result<(Vec<u8>, Option<i64>)> {
    let model = fields(bytes)?;
    let opset = opset(&model)?;
    let graph_field = model.iter().find(|f| f.number == 7).ok_or_else(invalid)?;
    let graph = fields(graph_field.value)?;
    let outputs: Vec<_> = graph.iter().filter(|f| f.number == 12).collect();
    let [probs] = outputs.as_slice() else {
        return Err(Error::ModelContract(format!("recognizer has {} outputs", outputs.len())));
    };
    let probs_name = fields(probs.value)?
        .into_iter()
        .find(|f| f.number == 1)
        .map(|f| string(f.value).map(str::to_string))
        .transpose()?
        .ok_or_else(invalid)?;
    let classes = last_dim(probs.value)?;

    let mut new_graph = Vec::with_capacity(graph_field.value.len() + 256);
    for f in graph.iter().filter(|f| f.number != 12) {
        new_graph.extend_from_slice(f.encoded);
    }
    let keep = int_attribute("keepdims", 0);
    put_bytes(
        &mut new_graph,
        1,
        &node("ArgMax", &probs_name, CLASSES, &[int_attribute("axis", 2), keep.clone()]),
    );
    if opset >= 18 {
        // ReduceMax takes `axes` as an input from opset 18; this graph has no such constant.
        return Err(Error::ModelContract(format!("recognizer opset {opset} is not supported")));
    }
    put_bytes(
        &mut new_graph,
        1,
        &node("ReduceMax", &probs_name, SCORES, &[ints_attribute("axes", &[2]), keep]),
    );
    put_bytes(&mut new_graph, 12, &output(CLASSES, 7));
    put_bytes(&mut new_graph, 12, &output(SCORES, 1));

    let mut out = Vec::with_capacity(bytes.len() + 256);
    for f in &model {
        if f.number == 7 {
            put_bytes(&mut out, 7, &new_graph);
        } else {
            out.extend_from_slice(f.encoded);
        }
    }
    Ok((out, classes))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn encodings_round_trip_through_the_parser() {
        let mut buf = Vec::new();
        put_int(&mut buf, 3, 300);
        put_bytes(&mut buf, 12, &output("x", 7));
        let f = fields(&buf).unwrap();
        assert_eq!((f[0].number, f[1].number), (3, 12));
        let mut o = 0;
        assert_eq!(varint(f[0].value, &mut o).unwrap(), 300);
        assert_eq!(f[0].encoded.len() + f[1].encoded.len(), buf.len());
        assert!(fields(&[0x0a, 0x05, 1]).is_err()); // truncated payload
    }
}
