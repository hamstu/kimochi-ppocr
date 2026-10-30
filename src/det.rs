//! PP-OCR DB text detection: `DetResizeForTest` (limit_type max), then `DBPostProcess`.
//!
//! Boxes are axis-aligned. For an axis-aligned text kernel, Paddle's minimum-area rectangle,
//! polygon offset (`unclip`) and second rectangle reduce exactly to growing the kernel's
//! rectangle by `area * unclip_ratio / perimeter` on every side, so neither OpenCV nor a
//! polygon clipper is needed. Rotated text gets its upright bounding box.
use crate::image_ops::{self, Region};
use crate::{Detection, Error, OcrOptions, Result};
use image::RgbImage;
use ndarray::Array4;
use crate::runtime::Model;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const MAX_CANDIDATES: usize = 3000;
const MIN_SIZE: f32 = 3.0;

/// The detector input size: the longest side limited to `max_side`, both sides multiples of 32.
pub(crate) fn input_size(width: u32, height: u32, max_side: u32) -> (usize, usize) {
    let longest = width.max(height) as f64;
    let ratio = if longest > max_side as f64 { max_side as f64 / longest } else { 1.0 };
    let round = |v: u32| (((v as f64 * ratio / 32.0).round() as usize) * 32).max(32);
    (round(width), round(height))
}

struct Component {
    min: (usize, usize),
    max: (usize, usize),
}

/// 8-connected components of `pred > threshold`, in raster order of their first pixel.
fn components(pred: &[f32], w: usize, h: usize, threshold: f32) -> Vec<Component> {
    let mut seen = vec![false; w * h];
    let mut stack = Vec::new();
    let mut out = Vec::new();
    for start in 0..w * h {
        if seen[start] || pred[start] <= threshold {
            continue;
        }
        if out.len() == MAX_CANDIDATES {
            break;
        }
        seen[start] = true;
        stack.push(start);
        let mut c = Component {
            min: (start % w, start / w),
            max: (start % w, start / w),
        };
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            c.min = (c.min.0.min(x), c.min.1.min(y));
            c.max = (c.max.0.max(x), c.max.1.max(y));
            for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let j = ny * w + nx;
                    if !seen[j] && pred[j] > threshold {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        out.push(c);
    }
    out
}

/// Boxes in map coordinates `[x1, y1, x2, y2]` with their scores, per `DBPostProcess`.
fn boxes(pred: &[f32], w: usize, h: usize, options: OcrOptions) -> Vec<([f32; 4], f32)> {
    let mut out = Vec::new();
    for c in components(pred, w, h, options.detection_threshold) {
        // minAreaRect of the contour's pixel centres.
        let (bw, bh) = ((c.max.0 - c.min.0) as f32, (c.max.1 - c.min.1) as f32);
        if bw.min(bh) < MIN_SIZE {
            continue;
        }
        // box_score_fast: mean probability over the box's pixel span.
        let mut sum = 0.0;
        for y in c.min.1..=c.max.1 {
            sum += pred[y * w + c.min.0..=y * w + c.max.0].iter().sum::<f32>();
        }
        let score = sum / ((c.max.0 - c.min.0 + 1) * (c.max.1 - c.min.1 + 1)) as f32;
        if score < options.box_threshold {
            continue;
        }
        let d = bw * bh * options.unclip_ratio / (2.0 * (bw + bh));
        if (bw + 2.0 * d).min(bh + 2.0 * d) < MIN_SIZE + 2.0 {
            continue;
        }
        let (x1, y1) = (c.min.0 as f32 - d, c.min.1 as f32 - d);
        let (x2, y2) = (c.max.0 as f32 + d, c.max.1 as f32 + d);
        out.push(([x1, y1, x2, y2], score));
    }
    out
}

/// PaddleOCR's `sorted_boxes`: top to bottom, then left to right within 10 px of the same top.
/// Pages that are mostly vertical columns read right to left instead.
fn reading_order(detections: &mut [Detection]) {
    let tall = detections
        .iter()
        .filter(|d| (d.bbox[3] - d.bbox[1]) as f32 >= 1.5 * (d.bbox[2] - d.bbox[0]) as f32)
        .count();
    if tall * 2 > detections.len() {
        detections.sort_by_key(|d| (std::cmp::Reverse(d.bbox[2]), d.bbox[1]));
        return;
    }
    detections.sort_by_key(|d| (d.bbox[1], d.bbox[0]));
    for i in 0..detections.len().saturating_sub(1) {
        for j in (0..=i).rev() {
            let (a, b) = (&detections[j], &detections[j + 1]);
            if b.bbox[1].abs_diff(a.bbox[1]) < 10 && b.bbox[0] < a.bbox[0] {
                detections.swap(j, j + 1);
            } else {
                break;
            }
        }
    }
}

pub(crate) fn detect(
    model: &mut Model,
    image: &RgbImage,
    max_side: u32,
    options: OcrOptions,
) -> Result<Vec<Detection>> {
    image_ops::validate(image)?;
    let (rw, rh) = input_size(image.width(), image.height(), max_side);
    let mut input = Array4::zeros((1, 3, rh, rw));
    let region = Region {
        bbox: [0, 0, image.width(), image.height()],
        rotated: false,
    };
    image_ops::resize_normalize(
        image,
        region,
        rw,
        rh,
        MEAN,
        STD,
        input.index_axis_mut(ndarray::Axis(0), 0),
    );
    let found = model.run(input, |shape, pred| {
        if shape != [1, 1, rh, rw] {
            return Err(Error::ModelContract(format!(
                "detection map {shape:?} for input {rh}x{rw}"
            )));
        }
        Ok(boxes(pred, rw, rh, options))
    })?;
    let (sx, sy) = (image.width() as f32 / rw as f32, image.height() as f32 / rh as f32);
    let mut detections: Vec<_> = found
        .into_iter()
        .filter_map(|([x1, y1, x2, y2], score)| {
            let clamp = |v: f32, max: u32| v.clamp(0.0, max as f32);
            let bbox = [
                clamp((x1 * sx).floor(), image.width()) as u32,
                clamp((y1 * sy).floor(), image.height()) as u32,
                clamp((x2 * sx).ceil(), image.width()) as u32,
                clamp((y2 * sy).ceil(), image.height()) as u32,
            ];
            (bbox[2] > bbox[0] && bbox[3] > bbox[1]).then_some(Detection {
                bbox,
                confidence: score,
            })
        })
        .collect();
    reading_order(&mut detections);
    Ok(detections)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn input_keeps_small_images_and_limits_large_ones_to_multiples_of_32() {
        assert_eq!(input_size(1920, 1080, 960), (960, 544));
        assert_eq!(input_size(300, 70, 960), (288, 64));
        assert_eq!(input_size(10, 10, 960), (32, 32));
    }
    #[test]
    fn a_rectangular_kernel_grows_by_the_unclip_distance() {
        let (w, h) = (40, 20);
        let mut pred = vec![0.0; w * h];
        for y in 5..=10 {
            for x in 4..=25 {
                pred[y * w + x] = 0.9;
            }
        }
        let found = boxes(&pred, w, h, OcrOptions::default());
        assert_eq!(found.len(), 1);
        let ([x1, y1, x2, y2], score) = found[0];
        // 21×5 kernel: d = 105 * 1.4 / 52.
        let d = 105.0 * 1.4 / 52.0;
        assert!((x1 - (4.0 - d)).abs() < 1e-4 && (y2 - (10.0 + d)).abs() < 1e-4);
        assert!(x2 > x1 && y2 > y1 && (score - 0.9).abs() < 1e-4);
    }
    #[test]
    fn weak_or_thin_kernels_are_dropped() {
        let (w, h) = (30, 10);
        let mut pred = vec![0.0; w * h];
        for x in 2..20 {
            pred[3 * w + x] = 0.9; // one pixel tall
        }
        assert!(boxes(&pred, w, h, OcrOptions::default()).is_empty());
    }
    #[test]
    fn same_row_reads_left_to_right_and_columns_right_to_left() {
        let d = |bbox| Detection { bbox, confidence: 1.0 };
        let mut rows = vec![d([200, 12, 300, 40]), d([10, 10, 100, 40]), d([10, 60, 100, 90])];
        reading_order(&mut rows);
        assert_eq!(rows.iter().map(|d| d.bbox[0]).collect::<Vec<_>>(), [10, 200, 10]);
        let mut columns = vec![d([10, 10, 40, 300]), d([100, 10, 130, 300])];
        reading_order(&mut columns);
        assert_eq!(columns[0].bbox[0], 100);
    }
}
