"""Track data structure for 3D object tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Track3D:
    """Represents a tracked object in 3D space.
    
    Attributes:
        id: Unique identifier for the tracked object
        x: X coordinate in 3D space (meters)
        y: Y coordinate in 3D space (meters)
        z: Z coordinate (depth/distance from camera in meters)
        bbox: Bounding box in 2D image space (x1, y1, x2, y2)
        class_id: Object class ID from the detector
        class_name: Human-readable class name
        confidence: Detection confidence score (0.0 to 1.0)
        velocity: Optional velocity vector (vx, vy, vz) in m/s
        is_new: True if object just appeared in this frame
        frames_tracked: Number of consecutive frames this object has been tracked
    """
    
    id: int
    x: float
    y: float
    z: float
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    velocity: Optional[tuple[float, float, float]] = None
    is_new: bool = False
    frames_tracked: int = 1
    
    def __repr__(self) -> str:
        return (
            f"Track3D(id={self.id}, class={self.class_name}, "
            f"pos=({self.x:.2f}, {self.y:.2f}, {self.z:.2f}m), "
            f"conf={self.confidence:.2f}, frames={self.frames_tracked})"
        )
