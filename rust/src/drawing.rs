use numpy::IntoPyArray;
use numpy::{PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

/// Draw rectangles on a (H, W, 3) uint8 image. Returns a new array.
///
/// boxes: (N, 4) float32 [x1, y1, x2, y2]
/// color: (R, G, B) tuple
/// thickness: line thickness in pixels
#[pyfunction]
#[pyo3(signature = (frame, boxes, color=(0, 255, 0), thickness=2))]
pub fn draw_bboxes_on_frame<'py>(
    py: Python<'py>,
    frame: PyReadonlyArray3<'py, u8>,
    boxes: PyReadonlyArray2<'py, f32>,
    color: (u8, u8, u8),
    thickness: usize,
) -> Bound<'py, PyArray3<u8>> {
    let arr = frame.as_array();
    let (h, w, _c) = arr.dim();
    let mut out = arr.to_owned();
    let b = boxes.as_array();
    let n = b.nrows();

    for idx in 0..n {
        let x1 = (b[[idx, 0]] as isize).clamp(0, w as isize - 1) as usize;
        let y1 = (b[[idx, 1]] as isize).clamp(0, h as isize - 1) as usize;
        let x2 = (b[[idx, 2]] as isize).clamp(0, w as isize - 1) as usize;
        let y2 = (b[[idx, 3]] as isize).clamp(0, h as isize - 1) as usize;

        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        let pixel = [color.0, color.1, color.2];

        // Top edge
        for dy in 0..thickness.min(y2 - y1) {
            let y = y1 + dy;
            for x in x1..=x2 {
                out[[y, x, 0]] = pixel[0];
                out[[y, x, 1]] = pixel[1];
                out[[y, x, 2]] = pixel[2];
            }
        }
        // Bottom edge
        for dy in 0..thickness.min(y2 - y1) {
            let y = y2 - dy;
            if y < h {
                for x in x1..=x2 {
                    out[[y, x, 0]] = pixel[0];
                    out[[y, x, 1]] = pixel[1];
                    out[[y, x, 2]] = pixel[2];
                }
            }
        }
        // Left edge
        for dx in 0..thickness.min(x2 - x1) {
            let x = x1 + dx;
            for y in y1..=y2 {
                out[[y, x, 0]] = pixel[0];
                out[[y, x, 1]] = pixel[1];
                out[[y, x, 2]] = pixel[2];
            }
        }
        // Right edge
        for dx in 0..thickness.min(x2 - x1) {
            let x = x2 - dx;
            if x < w {
                for y in y1..=y2 {
                    out[[y, x, 0]] = pixel[0];
                    out[[y, x, 1]] = pixel[1];
                    out[[y, x, 2]] = pixel[2];
                }
            }
        }
    }

    out.into_pyarray_bound(py)
}
