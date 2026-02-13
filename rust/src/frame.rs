use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;

/// Generate a (H, W, 3) uint8 test frame with a moving green square.
#[pyfunction]
pub fn generate_dummy_frame<'py>(py: Python<'py>, h: usize, w: usize, t: f64) -> Bound<'py, PyArray3<u8>> {
    let mut frame = Array3::<u8>::zeros((h, w, 3));

    let x = ((t.sin() * 0.4 + 0.5) * (w as f64 - 80.0)) as usize;
    let y = ((t.cos() * 0.4 + 0.5) * (h as f64 - 80.0)) as usize;

    let x_end = (x + 80).min(w);
    let y_end = (y + 80).min(h);

    for i in y..y_end {
        for j in x..x_end {
            frame[[i, j, 1]] = 255; // Green channel
        }
    }

    frame.into_pyarray_bound(py)
}
