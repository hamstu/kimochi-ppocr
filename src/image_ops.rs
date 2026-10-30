use crate::{BoundingBox, Error, Result};
use image::RgbImage;
use ndarray::{ArrayViewMut3, s};

pub(crate) fn validate(image: &RgbImage) -> Result<()> {
    if image.width() == 0 || image.height() == 0 {
        return Err(Error::InvalidInput(
            "image dimensions must be nonzero".into(),
        ));
    }
    Ok(())
}

/// A region of the source image, optionally read rotated 90° counter-clockwise
/// (PaddleOCR's `np.rot90` for tall crops, so vertical text reads left to right).
#[derive(Clone, Copy)]
pub(crate) struct Region {
    pub bbox: BoundingBox,
    pub rotated: bool,
}
impl Region {
    /// Width and height as the model sees them.
    pub fn size(self) -> (usize, usize) {
        let [x1, y1, x2, y2] = self.bbox;
        let (w, h) = ((x2 - x1) as usize, (y2 - y1) as usize);
        if self.rotated { (h, w) } else { (w, h) }
    }
}

/// Precomputed OpenCV `INTER_LINEAR` taps: half-pixel centres, edge-clamped.
fn taps(src: usize, dst: usize) -> Vec<(usize, usize, f32)> {
    let scale = src as f64 / dst as f64;
    (0..dst)
        .map(|i| {
            let f = ((i as f64 + 0.5) * scale - 0.5).max(0.0);
            let a = (f.floor() as usize).min(src - 1);
            let b = (a + 1).min(src - 1);
            (a, b, (f - a as f64) as f32)
        })
        .collect()
}

/// Resize `region` to `out_w`×`out_h` and write `(bgr / 255 - mean) / std` into the top-left
/// of `out` (CHW, BGR channel order as the Paddle models expect). The rest of `out` is untouched.
pub(crate) fn resize_normalize(
    image: &RgbImage,
    region: Region,
    out_w: usize,
    out_h: usize,
    mean: [f32; 3],
    std: [f32; 3],
    mut out: ArrayViewMut3<f32>,
) {
    let (w, h) = region.size();
    let [x1, y1, ..] = region.bbox;
    let xs = taps(w, out_w);
    let ys = taps(h, out_h);
    let stride = image.width() as usize * 3;
    let raw = image.as_raw();
    // Model-space (u, v) to source pixel; rotation maps u to source y and v to source x reversed.
    let fetch = |u: usize, v: usize| -> [f32; 3] {
        let (sx, sy) = if region.rotated {
            (x1 as usize + (h - 1 - v), y1 as usize + u)
        } else {
            (x1 as usize + u, y1 as usize + v)
        };
        let i = sy * stride + sx * 3;
        [raw[i] as f32, raw[i + 1] as f32, raw[i + 2] as f32]
    };
    let scale: [f32; 3] = std::array::from_fn(|c| 1.0 / (255.0 * std[c]));
    let bias: [f32; 3] = std::array::from_fn(|c| -mean[c] / std[c]);
    let mut planes = out.slice_mut(s![.., ..out_h, ..out_w]);
    for (y, &(ya, yb, dy)) in ys.iter().enumerate() {
        for (x, &(xa, xb, dx)) in xs.iter().enumerate() {
            let (p00, p01, p10, p11) = (fetch(xa, ya), fetch(xb, ya), fetch(xa, yb), fetch(xb, yb));
            for c in 0..3 {
                // RGB source channel 2 - c gives BGR order. Round like OpenCV's uint8 output.
                let k = 2 - c;
                let top = p00[k] + (p01[k] - p00[k]) * dx;
                let bottom = p10[k] + (p11[k] - p10[k]) * dx;
                let value = (top + (bottom - top) * dy).round();
                planes[[c, y, x]] = value * scale[c] + bias[c];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    #[test]
    fn rgb_becomes_bgr_and_rotation_reads_top_to_bottom() {
        // A 1×2 column: red above blue. Rotated, it is a 2×1 row: red then blue.
        let mut img = RgbImage::new(1, 2);
        img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        img.put_pixel(0, 1, image::Rgb([0, 0, 255]));
        let mut out = Array3::zeros((3, 1, 2));
        let region = Region { bbox: [0, 0, 1, 2], rotated: true };
        resize_normalize(&img, region, 2, 1, [0.0; 3], [1.0; 3], out.view_mut());
        // Channel 2 is red (BGR), channel 0 is blue.
        assert_eq!(out[[2, 0, 0]], 1.0);
        assert_eq!(out[[0, 0, 1]], 1.0);
        assert_eq!(out[[2, 0, 1]], 0.0);
    }
}
