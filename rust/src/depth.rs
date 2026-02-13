use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Normalize a float32 depth map to uint8 [0, 255].
#[pyfunction]
pub fn normalize_depth_map<'py>(
    py: Python<'py>,
    depth: PyReadonlyArray2<'py, f32>,
) -> Bound<'py, PyArray2<u8>> {
    let arr = depth.as_array();
    let d_min = arr.iter().cloned().fold(f32::INFINITY, f32::min);
    let d_max = arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let range = d_max - d_min;
    if range < 1e-9 {
        return Array2::<u8>::zeros(arr.dim()).into_pyarray_bound(py);
    }

    let out = arr.mapv(|v| ((v - d_min) / range * 255.0).clamp(0.0, 255.0) as u8);
    out.into_pyarray_bound(py)
}

/// JET LUT (BGR order to match OpenCV convention)
fn build_jet_lut() -> [[u8; 3]; 256] {
    let mut lut = [[0u8; 3]; 256];
    for i in 0..256 {
        let t = i as f32 / 255.0;

        // Red
        let r = if t < 0.375 {
            0.0
        } else if t < 0.625 {
            (t - 0.375) / 0.25
        } else if t < 0.875 {
            1.0
        } else {
            1.0 - (t - 0.875) / 0.25
        };

        // Green
        let g = if t < 0.125 {
            0.0
        } else if t < 0.375 {
            (t - 0.125) / 0.25
        } else if t < 0.625 {
            1.0
        } else if t < 0.875 {
            1.0 - (t - 0.625) / 0.25
        } else {
            0.0
        };

        // Blue
        let b = if t < 0.125 {
            0.5 + t / 0.125 * 0.5
        } else if t < 0.375 {
            1.0
        } else if t < 0.625 {
            1.0 - (t - 0.375) / 0.25
        } else {
            0.0
        };

        lut[i] = [
            (b.clamp(0.0, 1.0) * 255.0) as u8,
            (g.clamp(0.0, 1.0) * 255.0) as u8,
            (r.clamp(0.0, 1.0) * 255.0) as u8,
        ];
    }
    lut
}

/// Apply JET colormap to a single-channel uint8 image → (H, W, 3) BGR uint8.
#[pyfunction]
pub fn depth_to_colormap_jet<'py>(
    py: Python<'py>,
    depth_u8: PyReadonlyArray2<'py, u8>,
) -> Bound<'py, PyArray3<u8>> {
    let arr = depth_u8.as_array();
    let (h, w) = arr.dim();
    let lut = build_jet_lut();

    let mut out = Array3::<u8>::zeros((h, w, 3));

    out.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..w {
                let idx = arr[[i, j]] as usize;
                row[[j, 0]] = lut[idx][0];
                row[[j, 1]] = lut[idx][1];
                row[[j, 2]] = lut[idx][2];
            }
        });

    out.into_pyarray_bound(py)
}

/// Back-project a (H, W) depth map to (H*W, 3) XYZ point cloud.
#[pyfunction]
pub fn depth_to_pointcloud<'py>(
    py: Python<'py>,
    depth: PyReadonlyArray2<'py, f32>,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
) -> Bound<'py, PyArray2<f32>> {
    let arr = depth.as_array();
    let (h, w) = arr.dim();
    let n = h * w;

    let mut out = Array2::<f32>::zeros((n, 3));

    out.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(idx, mut point)| {
            let i = idx / w;
            let j = idx % w;
            let z = arr[[i, j]];
            point[0] = (j as f32 - cx) * z / fx;
            point[1] = (i as f32 - cy) * z / fy;
            point[2] = z;
        });

    out.into_pyarray_bound(py)
}
