use numpy::ndarray::Array3;
use numpy::{PyArray3, PyReadonlyArray3, IntoPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Convert a (H, W, 3) BGR uint8 image to RGB by swapping channels 0 and 2.
#[pyfunction]
pub fn bgr_to_rgb<'py>(py: Python<'py>, frame: PyReadonlyArray3<'py, u8>) -> Bound<'py, PyArray3<u8>> {
    let arr = frame.as_array();
    let (h, w, _c) = arr.dim();
    let mut out = Array3::<u8>::zeros((h, w, 3));

    // Parallel over rows
    out.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..w {
                row[[j, 0]] = arr[[i, j, 2]]; // R ← B
                row[[j, 1]] = arr[[i, j, 1]]; // G ← G
                row[[j, 2]] = arr[[i, j, 0]]; // B ← R
            }
        });

    out.into_pyarray_bound(py)
}

/// Convert a (H, W, 3) RGB uint8 image to BGR.
#[pyfunction]
pub fn rgb_to_bgr<'py>(py: Python<'py>, frame: PyReadonlyArray3<'py, u8>) -> Bound<'py, PyArray3<u8>> {
    // Channel swap is symmetric: RGB→BGR is the same as BGR→RGB.
    bgr_to_rgb(py, frame)
}

/// Resize a (H, W, 3) uint8 image using bilinear interpolation.
#[pyfunction]
pub fn resize_bilinear<'py>(
    py: Python<'py>,
    frame: PyReadonlyArray3<'py, u8>,
    new_h: usize,
    new_w: usize,
) -> Bound<'py, PyArray3<u8>> {
    let arr = frame.as_array();
    let (h, w, c) = arr.dim();

    if h == new_h && w == new_w {
        return arr.to_owned().into_pyarray_bound(py);
    }

    let row_ratio = h as f64 / new_h as f64;
    let col_ratio = w as f64 / new_w as f64;

    let mut out = Array3::<u8>::zeros((new_h, new_w, c));

    out.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let y_f = i as f64 * row_ratio;
            let y0 = (y_f as usize).min(h.saturating_sub(2));
            let y1 = y0 + 1;
            let fy = (y_f - y0 as f64) as f32;

            for j in 0..new_w {
                let x_f = j as f64 * col_ratio;
                let x0 = (x_f as usize).min(w.saturating_sub(2));
                let x1 = x0 + 1;
                let fx = (x_f - x0 as f64) as f32;

                for k in 0..c {
                    let tl = arr[[y0, x0, k]] as f32;
                    let tr = arr[[y0, x1, k]] as f32;
                    let bl = arr[[y1, x0, k]] as f32;
                    let br = arr[[y1, x1, k]] as f32;

                    let top = tl * (1.0 - fx) + tr * fx;
                    let bot = bl * (1.0 - fx) + br * fx;
                    let val = top * (1.0 - fy) + bot * fy;
                    row[[j, k]] = val.clamp(0.0, 255.0) as u8;
                }
            }
        });

    out.into_pyarray_bound(py)
}

/// Normalize (H, W, 3) uint8 to float32 in [0, 1].
#[pyfunction]
pub fn normalize_frame<'py>(py: Python<'py>, frame: PyReadonlyArray3<'py, u8>) -> Bound<'py, PyArray3<f32>> {
    let arr = frame.as_array();
    let out = arr.mapv(|v| v as f32 / 255.0);
    out.into_pyarray_bound(py)
}
