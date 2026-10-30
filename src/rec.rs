//! PP-OCR CTC recognition: `RecResizeImg` (height 48, padded to the batch's widest line),
//! then greedy `CTCLabelDecode`. Classes are `blank`, the dictionary, then ASCII space.
//!
//! CTC places each character's peak at the time steps where it is read, so character boxes
//! are reconstructed from those positions: boundaries fall midway between neighbouring peaks,
//! and no box extends more than [`HALF_WIDTH`] line heights from its peak.
use crate::image_ops::{self, Region};
use crate::{BoundingBox, Character, Detection, Error, Result, TextLine};
use image::RgbImage;
use ndarray::{Array4, Axis};
use crate::runtime::{Model, tensor};

pub(crate) const HEIGHT: usize = 48;
const MIN_WIDTH: usize = 320;
const MAX_WIDTH: usize = 3200;
/// Batch widths are rounded up to this on dynamic-shape backends.
const WIDTH_STEP: usize = 64;
/// Input widths on per-shape backends (one session each), and those built at load.
pub(crate) const WIDTHS: &[usize] = &[320, 640, 960, 1280, 1920, MAX_WIDTH];
pub(crate) const WARM_WIDTHS: &[usize] = &[320, 640, 960];
/// Crops at least this much taller than wide are vertical text, read rotated.
const VERTICAL_RATIO: f32 = 1.5;
const HALF_WIDTH: f32 = 0.6;

fn width_for(ratio: f32) -> usize {
    ((HEIGHT as f32 * ratio).ceil() as usize).max(1)
}

/// The recognizer's class count must match the dictionary plus blank and space. A compact
/// recognizer (`onnx.rs`) no longer outputs that axis, so its original declaration is used.
pub(crate) fn check_classes(model: &mut Model, dictionary: usize) -> Result<()> {
    let classes = if model.compact {
        model.declared_classes.map(|c| c as usize)
    } else {
        let shape =
            model.run(Array4::zeros((1, 3, HEIGHT, MIN_WIDTH)), |shape, _| Ok(shape.to_vec()))?;
        (shape.len() == 3).then(|| shape[2])
    };
    if classes != Some(dictionary + 2) {
        return Err(Error::ModelContract(format!(
            "recognizer has {classes:?} classes, dictionary {dictionary} (+ blank and space)"
        )));
    }
    Ok(())
}

/// The best class and its probability in one time step's scores.
fn best(row: &[f32]) -> (usize, f32) {
    // A plain scan: this runs over ~18.7k classes per time step, so no closures or total_cmp.
    let (mut class, mut p) = (0, row.first().copied().unwrap_or(0.0));
    for (k, &v) in row.iter().enumerate().skip(1) {
        if v > p {
            (class, p) = (k, v);
        }
    }
    (class, p)
}

/// Characters as (class, first step, last step, confidence) from one line's best class per step.
fn decode(steps: impl Iterator<Item = (usize, f32)>) -> Vec<(usize, usize, usize, f32)> {
    let mut out: Vec<(usize, usize, usize, f32)> = Vec::new();
    let mut previous = 0;
    for (t, (class, p)) in steps.enumerate() {
        if class != 0 {
            match out.last_mut() {
                Some(last) if class == previous => {
                    last.2 = t;
                    last.3 = last.3.max(p);
                }
                _ => out.push((class, t, t, p)),
            }
        }
        previous = class;
    }
    out
}

/// Model-space spans `[start, end)` along the line for characters centred at `centres`.
fn spans(centres: &[f32], length: f32, thickness: f32) -> Vec<(f32, f32)> {
    let limit = HALF_WIDTH * thickness;
    let n = centres.len();
    (0..n)
        .map(|i| {
            let c = centres[i];
            let left_gap = if i > 0 { (c - centres[i - 1]) / 2.0 } else { f32::INFINITY };
            let right_gap = if i + 1 < n { (centres[i + 1] - c) / 2.0 } else { f32::INFINITY };
            // An edge character mirrors its inner neighbour's gap.
            let left = left_gap.min(if left_gap.is_finite() { left_gap } else { right_gap });
            let right = right_gap.min(if right_gap.is_finite() { right_gap } else { left_gap });
            let (left, right) = if n == 1 { (limit, limit) } else { (left.min(limit), right.min(limit)) };
            ((c - left).max(0.0), (c + right).min(length))
        })
        .collect()
}

/// A model-space span along the line, as a box in source pixels.
fn to_source(region: Region, (a, b): (f32, f32)) -> BoundingBox {
    let [x1, y1, x2, y2] = region.bbox;
    let (a, b) = (a.floor() as u32, (b.ceil() as u32).max(a.floor() as u32 + 1));
    if region.rotated {
        [x1, (y1 + a).min(y2 - 1), x2, (y1 + b).min(y2)]
    } else {
        [(x1 + a).min(x2 - 1), y1, (x1 + b).min(x2), y2]
    }
}

pub(crate) fn recognize(
    model: &mut Model,
    dictionary: &[String],
    image: &RgbImage,
    boxes: &[Detection],
    threshold: f32,
    max_batch: usize,
) -> Result<Vec<TextLine>> {
    let regions: Vec<_> = boxes
        .iter()
        .map(|d| {
            let [x1, y1, x2, y2] = d.bbox;
            if x2 <= x1 || y2 <= y1 || x2 > image.width() || y2 > image.height() {
                return Err(Error::InvalidInput(format!("box {:?} is outside the image", d.bbox)));
            }
            let (w, h) = ((x2 - x1) as f32, (y2 - y1) as f32);
            Ok(Region { bbox: d.bbox, rotated: h >= VERTICAL_RATIO * w })
        })
        .collect::<Result<_>>()?;
    let ratio = |r: &Region| {
        let (w, h) = r.size();
        w as f32 / h as f32
    };
    let mut order: Vec<usize> = (0..regions.len()).collect();
    order.sort_by(|&a, &b| ratio(&regions[a]).total_cmp(&ratio(&regions[b])));
    let mut lines: Vec<Option<TextLine>> = vec![None; regions.len()];
    // Per-shape backends read one line per run at a bucketed width: batching saves little there
    // (DirectML: 1×960 in 5.6 ms, 4×960 in 15.8 ms), and every batch size would be another session.
    let per_shape = model.per_shape();
    let max_batch = if per_shape { 1 } else { max_batch };
    for batch in order.chunks(max_batch) {
        let widest = batch.iter().map(|&i| ratio(&regions[i])).fold(0.0, f32::max);
        let needed = width_for(widest).max(MIN_WIDTH);
        let width = if per_shape {
            WIDTHS.iter().copied().find(|&w| w >= needed).unwrap_or(MAX_WIDTH)
        } else {
            needed.div_ceil(WIDTH_STEP).saturating_mul(WIDTH_STEP).min(MAX_WIDTH)
        };
        let mut input = Array4::zeros((batch.len(), 3, HEIGHT, width));
        let mut resized = Vec::with_capacity(batch.len());
        for (slot, &i) in batch.iter().enumerate() {
            let w = width_for(ratio(&regions[i])).min(width);
            image_ops::resize_normalize(
                image,
                regions[i],
                w,
                HEIGHT,
                [0.5; 3],
                [0.5; 3],
                input.index_axis_mut(Axis(0), slot),
            );
            resized.push(w);
        }
        let classes = dictionary.len() + 2;
        let compact = model.compact;
        let (steps, decoded) = model.run_outputs(input, |outputs| {
            let contract = |shape: &[usize]| Error::ModelContract(format!("recognition output {shape:?}"));
            if compact {
                let (shape, best_class) = tensor::<i64>(outputs, 0)?;
                let (score_shape, score) = tensor::<f32>(outputs, 1)?;
                if shape.len() != 2 || shape[0] != batch.len() || score_shape != shape {
                    return Err(contract(&shape));
                }
                if best_class.iter().any(|&c| c < 0 || c as usize >= classes) {
                    return Err(Error::ModelContract("recognizer class out of range".into()));
                }
                let steps = shape[1];
                let lines = best_class
                    .chunks_exact(steps)
                    .zip(score.chunks_exact(steps))
                    .map(|(c, p)| decode(c.iter().map(|&c| c as usize).zip(p.iter().copied())))
                    .collect::<Vec<_>>();
                return Ok((steps, lines));
            }
            let (shape, probs) = tensor::<f32>(outputs, 0)?;
            if shape.len() != 3 || shape[0] != batch.len() || shape[2] != classes {
                return Err(contract(&shape));
            }
            let steps = shape[1];
            let lines = probs
                .chunks_exact(steps * classes)
                .map(|line| decode(line.chunks_exact(classes).map(best)))
                .collect::<Vec<_>>();
            Ok((steps, lines))
        })?;
        let mut decoded = decoded.into_iter();
        for (slot, &i) in batch.iter().enumerate() {
            let region = regions[i];
            let (length, thickness) = region.size();
            // Time step to model-space position along the source crop.
            let step = width as f32 / steps as f32 * length as f32 / resized[slot] as f32;
            let mut found = decoded.next().unwrap_or_default();
            let class_char = |class: usize| -> char {
                dictionary
                    .get(class - 1)
                    .and_then(|s| s.chars().next())
                    .unwrap_or(' ')
            };
            // Paddle's text keeps leading/trailing ASCII spaces; an overlay has no use for them.
            while found.first().is_some_and(|f| class_char(f.0) == ' ') {
                found.remove(0);
            }
            while found.last().is_some_and(|f| class_char(f.0) == ' ') {
                found.pop();
            }
            let centres: Vec<f32> = found
                .iter()
                .map(|&(_, a, b, _)| (a + b + 1) as f32 / 2.0 * step)
                .collect();
            let spans = spans(&centres, length as f32, thickness as f32);
            let characters: Vec<Character> = found
                .iter()
                .zip(spans)
                .map(|(&(class, _, _, confidence), span)| Character {
                    character: class_char(class),
                    bbox: to_source(region, span),
                    confidence,
                })
                .collect();
            let score = if characters.is_empty() {
                0.0
            } else {
                characters.iter().map(|c| c.confidence).sum::<f32>() / characters.len() as f32
            };
            lines[i] = Some(TextLine {
                text: characters.iter().map(|c| c.character).collect(),
                bbox: region.bbox,
                detection_confidence: Some(boxes[i].confidence),
                characters,
                is_vertical: region.rotated,
            })
            .filter(|line| threshold == 0.0 || (score >= threshold && !line.text.is_empty()));
        }
    }
    Ok(lines.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn one_hot(seq: &[usize], classes: usize) -> Vec<f32> {
        seq.iter()
            .flat_map(|&c| (0..classes).map(move |k| if k == c { 0.9 } else { 0.0 }))
            .collect()
    }
    #[test]
    fn ctc_collapses_repeats_and_keeps_blank_separated_doubles() {
        // a a _ a b b _ -> "a a b": the blank splits the doubled a.
        let probs = one_hot(&[1, 1, 0, 1, 2, 2, 0], 3);
        let chars: Vec<_> = decode(probs.as_chunks::<3>().0.iter().map(|row| best(row))).iter().map(|c| (c.0, c.1, c.2)).collect();
        assert_eq!(chars, [(1, 0, 1), (1, 3, 3), (2, 4, 5)]);
    }
    #[test]
    fn spans_meet_between_peaks_and_are_capped_by_line_height() {
        let s = spans(&[10.0, 30.0, 200.0], 220.0, 20.0);
        assert_eq!(s[0], (0.0, 20.0));
        assert_eq!(s[1], (20.0, 42.0)); // capped at 0.6 * 20 on the far side
        assert_eq!(s[2], (188.0, 212.0));
    }
    #[test]
    fn rotated_spans_run_down_the_source_column() {
        let region = Region { bbox: [100, 50, 130, 250], rotated: true };
        assert_eq!(to_source(region, (10.0, 40.0)), [100, 60, 130, 90]);
        let flat = Region { bbox: [100, 50, 300, 80], rotated: false };
        assert_eq!(to_source(flat, (10.0, 40.0)), [110, 50, 140, 80]);
    }
}
