use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Non-maximum suppression. Returns indices of kept boxes.
///
/// boxes: (N, 4) float32 [x1, y1, x2, y2]
/// scores: (N,) float32
#[pyfunction]
#[pyo3(signature = (boxes, scores, iou_threshold=0.5))]
pub fn nms_boxes<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'py, f32>,
    scores: PyReadonlyArray1<'py, f32>,
    iou_threshold: f32,
) -> Bound<'py, PyArray1<i64>> {
    let b = boxes.as_array();
    let s = scores.as_array();
    let n = b.nrows();

    if n == 0 {
        return Array1::<i64>::zeros(0).into_pyarray_bound(py);
    }

    // Sort by score descending
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b_idx| s[b_idx].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal));

    // Pre-compute areas
    let areas: Vec<f32> = (0..n)
        .map(|i| (b[[i, 2]] - b[[i, 0]]) * (b[[i, 3]] - b[[i, 1]]))
        .collect();

    let mut suppressed = vec![false; n];
    let mut keep = Vec::with_capacity(n);

    for &i in &order {
        if suppressed[i] {
            continue;
        }
        keep.push(i as i64);

        for &j in &order {
            if suppressed[j] || j == i {
                continue;
            }
            let xx1 = b[[i, 0]].max(b[[j, 0]]);
            let yy1 = b[[i, 1]].max(b[[j, 1]]);
            let xx2 = b[[i, 2]].min(b[[j, 2]]);
            let yy2 = b[[i, 3]].min(b[[j, 3]]);

            let inter = (xx2 - xx1).max(0.0) * (yy2 - yy1).max(0.0);
            let iou = inter / (areas[i] + areas[j] - inter + 1e-9);

            if iou > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    Array1::from_vec(keep).into_pyarray_bound(py)
}

/// Keep only detections with score >= min_score.
/// Returns (boxes, scores, class_ids) — all filtered.
#[pyfunction]
#[pyo3(signature = (boxes, scores, class_ids, min_score=0.25))]
pub fn filter_detections_by_score<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'py, f32>,
    scores: PyReadonlyArray1<'py, f32>,
    class_ids: PyReadonlyArray1<'py, i32>,
    min_score: f32,
) -> PyResult<(
    Bound<'py, numpy::PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<i32>>,
)> {
    let b = boxes.as_array();
    let s = scores.as_array();
    let c = class_ids.as_array();

    let mask: Vec<usize> = (0..s.len()).filter(|&i| s[i] >= min_score).collect();
    let k = mask.len();

    let mut out_b = numpy::ndarray::Array2::<f32>::zeros((k, 4));
    let mut out_s = Array1::<f32>::zeros(k);
    let mut out_c = Array1::<i32>::zeros(k);

    for (idx, &i) in mask.iter().enumerate() {
        for j in 0..4 {
            out_b[[idx, j]] = b[[i, j]];
        }
        out_s[idx] = s[i];
        out_c[idx] = c[i];
    }

    Ok((
        out_b.into_pyarray_bound(py),
        out_s.into_pyarray_bound(py),
        out_c.into_pyarray_bound(py),
    ))
}
