"""Visualization utilities for drawing 3D tracks on frames."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .track import Track3D


# Color palette for different track IDs (BGR format for OpenCV)
COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 255),    # Purple
    (255, 128, 0),    # Orange
    (0, 128, 255),    # Light Blue
    (128, 255, 0),    # Lime
]


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track3D],
    show_depth: bool = True,
    show_id: bool = True,
    show_class: bool = True,
    show_velocity: bool = False,
    show_trajectory: bool = False,
    is_bgr: bool = True,
) -> np.ndarray:
    """Draw track annotations on a frame.
    
    Args:
        frame: Input frame (H, W, 3) in BGR or RGB format
        tracks: List of Track3D objects to draw
        show_depth: Show depth (Z) value
        show_id: Show track ID
        show_class: Show class name
        show_velocity: Show velocity vector
        show_trajectory: Show trajectory path (requires trajectory history)
        is_bgr: True if frame is in BGR format (OpenCV default), False for RGB
    
    Returns:
        Annotated frame (same format as input)
    
    Notes:
        Requires opencv-python to be installed. Falls back to returning
        the original frame if OpenCV is not available.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        # OpenCV not available, return original frame
        return frame
    
    # Create a copy to avoid modifying the original
    output = frame.copy()
    
    for track in tracks:
        # Get color for this track ID
        color = COLORS[track.id % len(COLORS)]
        
        # Extract bounding box
        x1, y1, x2, y2 = track.bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw bounding box
        thickness = 3 if track.is_new else 2
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        
        # Build label text
        label_parts = []
        if show_id:
            label_parts.append(f"ID:{track.id}")
        if show_class:
            label_parts.append(track.class_name)
        if show_depth:
            label_parts.append(f"Z:{track.z:.2f}m")
        
        label = " ".join(label_parts)
        
        # Draw label background
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness_text = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness_text
            )
            
            # Position label above bbox
            label_y = max(y1 - 10, text_height + 10)
            
            # Draw background rectangle
            cv2.rectangle(
                output,
                (x1, label_y - text_height - baseline),
                (x1 + text_width, label_y + baseline),
                color,
                -1,  # Filled
            )
            
            # Draw text
            text_color = (255, 255, 255) if is_bgr else (255, 255, 255)
            cv2.putText(
                output,
                label,
                (x1, label_y),
                font,
                font_scale,
                text_color,
                thickness_text,
            )
        
        # Draw velocity vector if requested
        if show_velocity and track.velocity is not None:
            vx, vy, vz = track.velocity
            # Draw arrow showing 2D velocity (ignore Z component for visualization)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Scale velocity for visibility (assume velocity is in m/s, scale by 50 pixels per m/s)
            scale = 50
            end_x = int(center_x + vx * scale)
            end_y = int(center_y + vy * scale)
            
            cv2.arrowedLine(output, (center_x, center_y), (end_x, end_y), color, 2)
    
    return output


def draw_depth_overlay(
    frame: np.ndarray,
    depth_map: Optional[np.ndarray],
    alpha: float = 0.4,
) -> np.ndarray:
    """Draw depth map as a colored overlay on the frame.
    
    Args:
        frame: Input frame (H, W, 3) in BGR format
        depth_map: Depth map (H, W) with depth values
        alpha: Opacity of the overlay (0.0 to 1.0)
    
    Returns:
        Frame with depth overlay
    """
    if depth_map is None:
        return frame
    
    try:
        import cv2  # type: ignore
    except ImportError:
        return frame
    
    # Normalize depth map to 0-255
    d = depth_map.astype(np.float32)
    d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
    d_u8 = d_norm.astype(np.uint8)
    
    # Apply colormap
    d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
    
    # Blend with original frame
    output = cv2.addWeighted(frame, 1 - alpha, d_color, alpha, 0)
    
    return output
