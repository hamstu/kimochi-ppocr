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
///
/// Separable: each source row the output needs is interpolated horizontally once (into BGR
/// floats), and consecutive output rows share those rows, so a pixel costs one vertical lerp
/// instead of four fetches and three. The arithmetic is the same as the direct form, so the
/// rounded values are identical; a 1280x720 frame to 960x544 fell from 13 ms to 2 ms.
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
    // Model-space (u, v) to the source pixel's first byte: `base(v) + step * u`. Rotation maps u
    // to source y (so `step` is a row) and v to source x reversed.
    let (step, base): (usize, Box<dyn Fn(usize) -> usize>) = if region.rotated {
        (stride, Box::new(move |v| y1 as usize * stride + (x1 as usize + h - 1 - v) * 3))
    } else {
        (3, Box::new(move |v| (y1 as usize + v) * stride + x1 as usize * 3))
    };
    let taps_x: Vec<(usize, usize, f32)> = xs.iter().map(|&(a, b, dx)| (a * step, b * step, dx)).collect();
    // Source row `v` resized horizontally, in BGR order.
    let horizontal = |v: usize, row: &mut [[f32; 3]]| {
        let start = base(v);
        for (dst, &(a, b, dx)) in row.iter_mut().zip(&taps_x) {
            let pa = &raw[start + a..start + a + 3];
            let pb = &raw[start + b..start + b + 3];
            *dst = std::array::from_fn(|c| {
                let k = 2 - c;
                let (pa, pb) = (pa[k] as f32, pb[k] as f32);
                pa + (pb - pa) * dx
            });
        }
    };
    let scale: [f32; 3] = std::array::from_fn(|c| 1.0 / (255.0 * std[c]));
    let bias: [f32; 3] = std::array::from_fn(|c| -mean[c] / std[c]);
    let mut planes = out.slice_mut(s![.., ..out_h, ..out_w]);
    let (mut top, mut bottom) = (vec![[0.0f32; 3]; out_w], vec![[0.0f32; 3]; out_w]);
    let (mut top_v, mut bottom_v) = (usize::MAX, usize::MAX);
    for (y, &(ya, yb, dy)) in ys.iter().enumerate() {
        // Rows advance monotonically, so the previous bottom row often becomes the new top row.
        if ya != top_v {
            if ya == bottom_v {
                std::mem::swap(&mut top, &mut bottom);
                top_v = ya;
                bottom_v = usize::MAX;
            } else {
                horizontal(ya, &mut top);
                top_v = ya;
            }
        }
        if yb != bottom_v {
            if yb == top_v {
                bottom.copy_from_slice(&top);
            } else {
                horizontal(yb, &mut bottom);
            }
            bottom_v = yb;
        }
        for c in 0..3 {
            let mut plane_row = planes.slice_mut(s![c, y, ..]);
            let plane_row = plane_row.as_slice_mut().expect("contiguous output row");
            for (x, dst) in plane_row.iter_mut().enumerate() {
                let (t, b) = (top[x][c], bottom[x][c]);
                // Round like OpenCV's uint8 output.
                let value = (t + (b - t) * dy).round();
                *dst = value * scale[c] + bias[c];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    /// `cargo test --release -- --ignored --nocapture resize_cost`: the detector's resize of a
    /// 1280x720 frame to 960x544 and a recognizer crop, in milliseconds.
    #[test]
    #[ignore]
    fn resize_cost() {
        let mut img = RgbImage::new(1280, 720);
        for (i, p) in img.pixels_mut().enumerate() {
            *p = image::Rgb([(i % 251) as u8, (i % 241) as u8, (i % 239) as u8]);
        }
        let mut out = Array3::zeros((3, 544, 960));
        let region = Region { bbox: [0, 0, 1280, 720], rotated: false };
        let t = std::time::Instant::now();
        for _ in 0..20 {
            resize_normalize(&img, region, 960, 544, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], out.view_mut());
        }
        println!("det resize 1280x720 -> 960x544: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0 / 20.0);
        let mut crop = Array3::zeros((3, 48, 960));
        let line = Region { bbox: [80, 480, 900, 520], rotated: false };
        let t = std::time::Instant::now();
        for _ in 0..100 {
            resize_normalize(&img, line, 960, 48, [0.5; 3], [0.5; 3], crop.view_mut());
        }
        println!("rec resize 820x40 -> 960x48: {:.3} ms", t.elapsed().as_secs_f64() * 1000.0 / 100.0);
    }
    /// The direct four-tap form the separable version replaced; the two must agree exactly.
    fn resize_direct(image: &RgbImage, region: Region, out_w: usize, out_h: usize, mean: [f32; 3], std: [f32; 3], mut out: ArrayViewMut3<f32>) {
        let (w, h) = region.size();
        let [x1, y1, ..] = region.bbox;
        let xs = taps(w, out_w);
        let ys = taps(h, out_h);
        let stride = image.width() as usize * 3;
        let raw = image.as_raw();
        let fetch = |u: usize, v: usize| -> [f32; 3] {
            let (sx, sy) = if region.rotated { (x1 as usize + (h - 1 - v), y1 as usize + u) } else { (x1 as usize + u, y1 as usize + v) };
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
                    let k = 2 - c;
                    let top = p00[k] + (p01[k] - p00[k]) * dx;
                    let bottom = p10[k] + (p11[k] - p10[k]) * dx;
                    planes[[c, y, x]] = (top + (bottom - top) * dy).round() * scale[c] + bias[c];
                }
            }
        }
    }
    #[test]
    fn separable_resize_matches_the_direct_form_bit_for_bit() {
        let mut img = RgbImage::new(97, 61);
        for (i, p) in img.pixels_mut().enumerate() {
            *p = image::Rgb([(i * 7 % 251) as u8, (i * 13 % 241) as u8, (i * 31 % 239) as u8]);
        }
        for (region, out_w, out_h) in [
            (Region { bbox: [0, 0, 97, 61], rotated: false }, 64, 32),
            (Region { bbox: [5, 7, 80, 20], rotated: false }, 320, 48),
            (Region { bbox: [10, 3, 22, 58], rotated: true }, 96, 48),
            (Region { bbox: [0, 0, 97, 61], rotated: false }, 160, 96), // upscale: rows repeat
        ] {
            let mut a = Array3::zeros((3, out_h, out_w));
            let mut b = Array3::zeros((3, out_h, out_w));
            resize_normalize(&img, region, out_w, out_h, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], a.view_mut());
            resize_direct(&img, region, out_w, out_h, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], b.view_mut());
            assert_eq!(a, b, "{:?} {out_w}x{out_h}", region.bbox);
        }
    }
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
