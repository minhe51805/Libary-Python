mod image_ops;
mod nms;
mod depth;
mod drawing;
mod frame;

use pyo3::prelude::*;

/// Rust acceleration module for scanlt3d.
#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // image_ops
    m.add_function(wrap_pyfunction!(image_ops::bgr_to_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(image_ops::rgb_to_bgr, m)?)?;
    m.add_function(wrap_pyfunction!(image_ops::resize_bilinear, m)?)?;
    m.add_function(wrap_pyfunction!(image_ops::normalize_frame, m)?)?;

    // nms
    m.add_function(wrap_pyfunction!(nms::nms_boxes, m)?)?;
    m.add_function(wrap_pyfunction!(nms::filter_detections_by_score, m)?)?;

    // depth
    m.add_function(wrap_pyfunction!(depth::normalize_depth_map, m)?)?;
    m.add_function(wrap_pyfunction!(depth::depth_to_colormap_jet, m)?)?;
    m.add_function(wrap_pyfunction!(depth::depth_to_pointcloud, m)?)?;

    // drawing
    m.add_function(wrap_pyfunction!(drawing::draw_bboxes_on_frame, m)?)?;

    // frame
    m.add_function(wrap_pyfunction!(frame::generate_dummy_frame, m)?)?;

    Ok(())
}
