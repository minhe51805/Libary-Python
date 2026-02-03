"""Convert depth maps and pixel coordinates to 3D world coordinates."""

from __future__ import annotations

from typing import Optional

import numpy as np


def pixel_to_3d(
    x: float,
    y: float,
    depth: float,
    camera_matrix: Optional[np.ndarray] = None,
    image_width: int = 640,
    image_height: int = 480,
) -> tuple[float, float, float]:
    """Convert 2D pixel coordinates + depth to 3D world coordinates.
    
    Args:
        x: Pixel x-coordinate
        y: Pixel y-coordinate
        depth: Depth value at that pixel (in meters)
        camera_matrix: Optional 3x3 camera intrinsic matrix. If None, uses default webcam values.
        image_width: Image width in pixels (used for default camera matrix)
        image_height: Image height in pixels (used for default camera matrix)
    
    Returns:
        Tuple of (X, Y, Z) coordinates in meters relative to camera.
        X is horizontal (positive = right), Y is vertical (positive = down),
        Z is depth (positive = away from camera)
    
    Notes:
        Default camera matrix assumes typical webcam with:
        - Focal length ~= 0.8 * image_width
        - Principal point at image center
    """
    if camera_matrix is None:
        # Default intrinsic matrix for typical webcam
        # Focal length is roughly 0.8 * image width for typical webcams
        fx = fy = 0.8 * image_width
        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Convert pixel coordinates to 3D
    # Using pinhole camera model: x = (X/Z)*fx + cx, y = (Y/Z)*fy + cy
    # Solving for X, Y: X = (x - cx) * Z / fx, Y = (y - cy) * Z / fy
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    
    return (float(X), float(Y), float(Z))


def get_bbox_depth(
    bbox: tuple[float, float, float, float],
    depth_map: np.ndarray,
    method: str = "center",
) -> float:
    """Extract depth value for a bounding box from a depth map.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        depth_map: Depth map array (H, W)
        method: Method to extract depth. Options:
            - "center": Use depth at center of bbox (default)
            - "median": Use median depth in bbox region
            - "min": Use minimum depth in bbox region (closest point)
    
    Returns:
        Depth value in meters (same units as depth_map)
    """
    x1, y1, x2, y2 = bbox
    h, w = depth_map.shape[:2]
    
    # Clip to image bounds
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))
    
    # Ensure valid box
    if x2 <= x1 or y2 <= y1:
        return 1.0  # Default depth if invalid box
    
    if method == "center":
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return float(depth_map[cy, cx])
    
    elif method == "median":
        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return 1.0
        # Filter out zero/invalid depths
        valid = region[region > 0]
        if valid.size == 0:
            return float(np.median(region))
        return float(np.median(valid))
    
    elif method == "min":
        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return 1.0
        # Filter out zero/invalid depths
        valid = region[region > 0]
        if valid.size == 0:
            return float(np.min(region))
        return float(np.min(valid))
    
    else:
        raise ValueError(f"Unknown depth extraction method: {method}")
