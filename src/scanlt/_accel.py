"""Acceleration bridge — imports Rust extension when available, falls back to NumPy."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Try to load the compiled Rust extension (_rust_core).
# If not available (e.g. source install without Rust toolchain), every
# function falls back to a pure-Python / NumPy implementation.
# ---------------------------------------------------------------------------
try:
    from ._rust_core import (  # type: ignore[import-not-found]
        bgr_to_rgb as _rs_bgr_to_rgb,
        rgb_to_bgr as _rs_rgb_to_bgr,
        resize_bilinear as _rs_resize_bilinear,
        normalize_frame as _rs_normalize_frame,
        nms_boxes as _rs_nms_boxes,
        filter_detections_by_score as _rs_filter_detections_by_score,
        normalize_depth_map as _rs_normalize_depth_map,
        depth_to_colormap_jet as _rs_depth_to_colormap_jet,
        depth_to_pointcloud as _rs_depth_to_pointcloud,
        draw_bboxes_on_frame as _rs_draw_bboxes_on_frame,
        generate_dummy_frame as _rs_generate_dummy_frame,
    )

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

# ---------------------------------------------------------------------------
# Public flag
# ---------------------------------------------------------------------------
RUST_AVAILABLE: bool = _RUST_AVAILABLE

# ===== Image Ops ==========================================================


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a (H, W, 3) BGR uint8 image to RGB."""
    if _RUST_AVAILABLE:
        return _rs_bgr_to_rgb(frame)
    return frame[:, :, ::-1].copy()


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert a (H, W, 3) RGB uint8 image to BGR."""
    if _RUST_AVAILABLE:
        return _rs_rgb_to_bgr(frame)
    return frame[:, :, ::-1].copy()


def resize_bilinear(frame: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Resize (H, W, 3) uint8 image using bilinear interpolation."""
    if _RUST_AVAILABLE:
        return _rs_resize_bilinear(frame, new_h, new_w)

    h, w = frame.shape[:2]
    if h == new_h and w == new_w:
        return frame.copy()

    row_ratio = h / new_h
    col_ratio = w / new_w

    row_idx = np.arange(new_h) * row_ratio
    col_idx = np.arange(new_w) * col_ratio

    row_floor = np.clip(np.floor(row_idx).astype(np.intp), 0, h - 2)
    col_floor = np.clip(np.floor(col_idx).astype(np.intp), 0, w - 2)

    row_frac = (row_idx - row_floor).reshape(-1, 1, 1).astype(np.float32)
    col_frac = (col_idx - col_floor).reshape(1, -1, 1).astype(np.float32)

    top_left = frame[np.ix_(row_floor, col_floor)]
    top_right = frame[np.ix_(row_floor, col_floor + 1)]
    bot_left = frame[np.ix_(row_floor + 1, col_floor)]
    bot_right = frame[np.ix_(row_floor + 1, col_floor + 1)]

    top = top_left * (1 - col_frac) + top_right * col_frac
    bot = bot_left * (1 - col_frac) + bot_right * col_frac
    result = top * (1 - row_frac) + bot * row_frac
    return np.clip(result, 0, 255).astype(np.uint8)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize (H, W, 3) uint8 → float32 in [0, 1]."""
    if _RUST_AVAILABLE:
        return _rs_normalize_frame(frame)
    return frame.astype(np.float32) / 255.0


# ===== NMS =================================================================


def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Non-maximum suppression. Returns indices of kept boxes.

    Parameters
    ----------
    boxes : (N, 4) float32 array of [x1, y1, x2, y2]
    scores : (N,) float32 array
    iou_threshold : IoU threshold for suppression
    """
    if _RUST_AVAILABLE:
        return _rs_nms_boxes(boxes, scores, iou_threshold)

    if len(boxes) == 0:
        return np.empty(0, dtype=np.intp)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)

        order = rest[iou <= iou_threshold]

    return np.array(keep, dtype=np.intp)


def filter_detections_by_score(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    min_score: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only detections with ``score >= min_score``."""
    if _RUST_AVAILABLE:
        return _rs_filter_detections_by_score(boxes, scores, class_ids, min_score)

    mask = scores >= min_score
    return boxes[mask], scores[mask], class_ids[mask]


# ===== Depth ===============================================================


def normalize_depth_map(depth: np.ndarray) -> np.ndarray:
    """Normalize a float32 depth map to uint8 [0, 255]."""
    if _RUST_AVAILABLE:
        return _rs_normalize_depth_map(depth)

    d = depth.astype(np.float32, copy=False)
    d_min = d.min()
    d_max = d.max()
    if d_max - d_min < 1e-9:
        return np.zeros(d.shape, dtype=np.uint8)
    d_norm = (d - d_min) / (d_max - d_min) * 255.0
    return d_norm.astype(np.uint8)


def depth_to_colormap_jet(depth_u8: np.ndarray) -> np.ndarray:
    """Apply JET colormap to a single-channel uint8 image → (H, W, 3) BGR uint8."""
    if _RUST_AVAILABLE:
        return _rs_depth_to_colormap_jet(depth_u8)

    # Build a compact 256-entry JET LUT (BGR order, matching OpenCV convention).
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        # Red channel
        if t < 0.375:
            r = 0.0
        elif t < 0.625:
            r = (t - 0.375) / 0.25
        elif t < 0.875:
            r = 1.0
        else:
            r = 1.0 - (t - 0.875) / 0.25
        # Green channel
        if t < 0.125:
            g = 0.0
        elif t < 0.375:
            g = (t - 0.125) / 0.25
        elif t < 0.625:
            g = 1.0
        elif t < 0.875:
            g = 1.0 - (t - 0.625) / 0.25
        else:
            g = 0.0
        # Blue channel
        if t < 0.125:
            b = 0.5 + t / 0.125 * 0.5
        elif t < 0.375:
            b = 1.0
        elif t < 0.625:
            b = 1.0 - (t - 0.375) / 0.25
        else:
            b = 0.0
        lut[i] = [int(b * 255), int(g * 255), int(r * 255)]  # BGR

    return lut[depth_u8]


def depth_to_pointcloud(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Back-project a (H, W) depth map to (H*W, 3) XYZ point cloud.

    Parameters
    ----------
    depth : (H, W) float32
    fx, fy : focal lengths in pixels
    cx, cy : principal point
    """
    if _RUST_AVAILABLE:
        return _rs_depth_to_pointcloud(depth, fx, fy, cx, cy)

    h, w = depth.shape[:2]
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    z = depth.astype(np.float32, copy=False)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


# ===== Drawing =============================================================


def draw_bboxes_on_frame(
    frame: np.ndarray,
    boxes: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw rectangles on a (H, W, 3) uint8 image. Returns a copy.

    Parameters
    ----------
    boxes : (N, 4) int or float array of [x1, y1, x2, y2]
    color : RGB tuple
    thickness : line thickness in pixels
    """
    if _RUST_AVAILABLE:
        return _rs_draw_bboxes_on_frame(frame, boxes, color, thickness)

    out = frame.copy()
    h, w = out.shape[:2]
    for box in boxes:
        x1 = int(max(0, min(w - 1, box[0])))
        y1 = int(max(0, min(h - 1, box[1])))
        x2 = int(max(0, min(w - 1, box[2])))
        y2 = int(max(0, min(h - 1, box[3])))
        # Top / bottom horizontal lines
        out[y1 : y1 + thickness, x1 : x2 + 1] = color
        out[max(0, y2 - thickness + 1) : y2 + 1, x1 : x2 + 1] = color
        # Left / right vertical lines
        out[y1 : y2 + 1, x1 : x1 + thickness] = color
        out[y1 : y2 + 1, max(0, x2 - thickness + 1) : x2 + 1] = color
    return out


# ===== Dummy Frame =========================================================


def generate_dummy_frame(h: int, w: int, t: float) -> np.ndarray:
    """Generate a (H, W, 3) uint8 test frame with a moving green square."""
    if _RUST_AVAILABLE:
        return _rs_generate_dummy_frame(h, w, t)

    import math

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    x = int((math.sin(t) * 0.4 + 0.5) * (w - 80))
    y = int((math.cos(t) * 0.4 + 0.5) * (h - 80))
    frame[y : y + 80, x : x + 80, 1] = 255
    return frame
