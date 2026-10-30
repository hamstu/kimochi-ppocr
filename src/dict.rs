//! The recognizer's classes, read from `PostProcess.character_dict` in its `inference.yml`.
//!
//! The published dictionaries are one flat YAML sequence of plain or single-quoted scalars,
//! so this reads exactly that shape and rejects anything else rather than guessing.
use crate::{Error, Result};

fn invalid(line: usize, why: &str) -> Error {
    Error::ModelContract(format!("character_dict line {line}: {why}"))
}

/// One sequence item's scalar: plain (`- a`) or single-quoted (`- ''''` is `'`).
fn scalar(raw: &str, line: usize) -> Result<String> {
    if let Some(body) = raw.strip_prefix('\'') {
        let body = body
            .strip_suffix('\'')
            .ok_or_else(|| invalid(line, "unterminated quote"))?;
        if body.replace("''", "").contains('\'') {
            return Err(invalid(line, "unescaped quote"));
        }
        return Ok(body.replace("''", "'"));
    }
    if raw.starts_with('"') || raw.is_empty() {
        return Err(invalid(line, "unsupported scalar"));
    }
    Ok(raw.to_string())
}

pub(crate) fn character_dict(yaml: &str) -> Result<Vec<String>> {
    let mut lines = yaml.lines().enumerate();
    let (indent, _) = lines
        .by_ref()
        .find_map(|(_, l)| {
            let trimmed = l.trim_start();
            (trimmed == "character_dict:").then(|| (l.len() - trimmed.len(), ()))
        })
        .ok_or_else(|| Error::ModelContract("inference.yml has no character_dict".into()))?;
    let mut out = Vec::new();
    for (n, l) in lines {
        let trimmed = l.trim_start();
        let depth = l.len() - trimmed.len();
        // The sequence is either indented under the key or level with it (PyYAML's default).
        let Some(item) = trimmed.strip_prefix("- ") else {
            if depth <= indent && !trimmed.is_empty() {
                break;
            }
            return Err(invalid(n + 1, "expected a sequence item"));
        };
        if depth < indent {
            break;
        }
        let value = scalar(item.trim_end_matches('\r'), n + 1)?;
        if value.chars().count() != 1 {
            return Err(invalid(n + 1, "entry is not a single character"));
        }
        out.push(value);
    }
    if out.is_empty() {
        return Err(Error::ModelContract("character_dict is empty".into()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reads_plain_and_quoted_items_and_stops_at_the_next_key() {
        let yaml = "PostProcess:\n  name: CTCLabelDecode\n  character_dict:\n  - '!'\n  - ''''\n  - a\n  - 日\nPreProcess:\n  x: 1\n";
        assert_eq!(character_dict(yaml).unwrap(), ["!", "'", "a", "日"]);
    }
    #[test]
    fn rejects_shapes_it_does_not_understand() {
        assert!(character_dict("character_dict:\n  - \"\\u3000\"\n").is_err());
        assert!(character_dict("character_dict:\n  - ab\n").is_err());
        assert!(character_dict("name: x\n").is_err());
    }
}
