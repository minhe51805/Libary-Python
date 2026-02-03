"""Core 3D tracking engine for OpenCV integration."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .api import Detection, Detector, DepthEstimator
from .backends import choose_backend
from .depth_to_3d import get_bbox_depth, pixel_to_3d
from .track import Track3D
from .visualization import draw_tracks


# Default class names for COCO dataset (used by YOLO models)
COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class Tracker3D:
    """3D object tracker for OpenCV workflows.
    
    This class provides an easy-to-use interface for tracking objects in 3D space
    using a camera. It integrates detection, depth estimation, and tracking into
    a single API.
    
    Example:
        >>> import cv2
        >>> import scanlt
        >>> 
        >>> tracker = scanlt.Tracker3D()
        >>> cap = cv2.VideoCapture(0)
        >>> 
        >>> while True:
        >>>     ret, frame = cap.read()
        >>>     annotated_frame, tracks = tracker.process(frame)
        >>>     
        >>>     for track in tracks:
        >>>         print(f"Object {track.id}: Z={track.z:.2f}m")
        >>>     
        >>>     cv2.imshow("Tracking", annotated_frame)
        >>>     if cv2.waitKey(1) & 0xFF == ord('q'):
        >>>         break
    """
    
    def __init__(
        self,
        profile: str = "fast",
        backend: str = "auto",
        conf_threshold: float = 0.25,
        max_objects: int = 50,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        class_names: Optional[list[str]] = None,
    ):
        """Initialize the 3D tracker.
        
        Args:
            profile: Model profile ("fast", "balanced", "quality")
            backend: Computing backend ("auto", "cpu", "cuda", "mps", "dml")
            conf_threshold: Minimum confidence for detections
            max_objects: Maximum number of objects to track
            max_age: Maximum frames to keep a track without updates
            min_hits: Minimum consecutive detections before confirming a track
            iou_threshold: IOU threshold for matching detections to tracks
            class_names: Optional list of class names. If None, uses COCO names.
        """
        self.profile = profile
        self.backend_choice = choose_backend(backend)
        self.conf_threshold = conf_threshold
        self.max_objects = max_objects
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.class_names = class_names or COCO_CLASS_NAMES
        
        # Initialize detector
        self._detector: Optional[Detector] = None
        self._depth_estimator: Optional[DepthEstimator] = None
        self._init_models()
        
        # Tracking state
        self._tracks: dict[int, _TrackState] = {}
        self._next_id = 0
        self._frame_count = 0
        
        # Timing for velocity estimation
        self._last_time = time.perf_counter()
        
    def _init_models(self) -> None:
        """Initialize detection and depth models."""
        try:
            from .model_zoo import ensure_model, get_default_yolo_seg_specs
            from .onnx_yolo_seg import OnnxYoloSegDetector
            
            specs = get_default_yolo_seg_specs()
            if self.profile in specs:
                model_path = ensure_model(specs[self.profile])
                self._detector = OnnxYoloSegDetector(
                    str(model_path),
                    backend=self.backend_choice.name
                )
        except Exception:
            # If model loading fails, use a dummy detector
            self._detector = None
    
    def configure(
        self,
        conf_threshold: Optional[float] = None,
        max_objects: Optional[int] = None,
        max_age: Optional[int] = None,
        min_hits: Optional[int] = None,
        iou_threshold: Optional[float] = None,
    ) -> None:
        """Update tracker configuration.
        
        Args:
            conf_threshold: Minimum confidence for detections
            max_objects: Maximum number of objects to track
            max_age: Maximum frames to keep a track without updates
            min_hits: Minimum consecutive detections before confirming a track
            iou_threshold: IOU threshold for matching detections to tracks
        """
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if max_objects is not None:
            self.max_objects = max_objects
        if max_age is not None:
            self.max_age = max_age
        if min_hits is not None:
            self.min_hits = min_hits
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
    
    def reset(self) -> None:
        """Reset tracking state (clear all tracks)."""
        self._tracks.clear()
        self._next_id = 0
        self._frame_count = 0
        self._last_time = time.perf_counter()
    
    def process(
        self,
        frame: np.ndarray,
        draw_annotations: bool = True,
    ) -> tuple[np.ndarray, list[Track3D]]:
        """Process a frame and return annotated frame with tracks.
        
        Args:
            frame: Input frame (H, W, 3) in BGR or RGB format
            draw_annotations: Whether to draw track annotations on the frame
        
        Returns:
            Tuple of (annotated_frame, tracks) where:
            - annotated_frame: Frame with drawn annotations
            - tracks: List of Track3D objects for confirmed tracks
        """
        # Calculate time delta for velocity estimation
        current_time = time.perf_counter()
        dt = current_time - self._last_time
        self._last_time = current_time
        
        # Detect objects
        detections: list[Detection] = []
        if self._detector is not None:
            detections = self._detector.predict(frame)
        
        # Filter by confidence
        detections = [d for d in detections if d.score >= self.conf_threshold]
        
        # Create a simple depth map (placeholder - in real usage, integrate depth estimator)
        # For now, use a dummy depth map
        h, w = frame.shape[:2]
        depth_map = np.ones((h, w), dtype=np.float32) * 2.0  # Default 2 meters
        
        # Update tracks
        self._update_tracks(detections, depth_map, dt)
        
        # Get confirmed tracks
        confirmed_tracks = self._get_confirmed_tracks()
        
        # Draw annotations if requested
        annotated_frame = frame
        if draw_annotations:
            # Convert BGR to RGB if needed for visualization
            annotated_frame = draw_tracks(
                frame,
                confirmed_tracks,
                show_depth=True,
                show_id=True,
                show_class=True,
                is_bgr=True,
            )
        
        self._frame_count += 1
        
        return annotated_frame, confirmed_tracks
    
    def _update_tracks(
        self,
        detections: list[Detection],
        depth_map: np.ndarray,
        dt: float,
    ) -> None:
        """Update tracks with new detections."""
        # Match detections to existing tracks using IOU
        matched_pairs, unmatched_dets, unmatched_tracks = self._match_detections(detections)
        
        # Update matched tracks
        for det_idx, track_id in matched_pairs:
            detection = detections[det_idx]
            track_state = self._tracks[track_id]
            track_state.update(detection, depth_map, dt)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            if len(self._tracks) >= self.max_objects:
                break
            detection = detections[det_idx]
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                detection=detection,
                depth_map=depth_map,
                class_names=self.class_names,
            )
        
        # Mark unmatched tracks as missing
        for track_id in unmatched_tracks:
            self._tracks[track_id].mark_missing()
        
        # Remove old tracks
        to_remove = []
        for track_id, track_state in self._tracks.items():
            if track_state.age > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self._tracks[track_id]
    
    def _match_detections(
        self,
        detections: list[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Match detections to tracks using IOU.
        
        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(detections) == 0:
            return [], [], list(self._tracks.keys())
        
        if len(self._tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Compute IOU matrix
        iou_matrix = np.zeros((len(detections), len(self._tracks)))
        track_ids = list(self._tracks.keys())
        
        for i, detection in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track_state = self._tracks[track_id]
                iou = self._compute_iou(detection.xyxy, track_state.bbox)
                iou_matrix[i, j] = iou
        
        # Greedy matching
        matched_pairs = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_ids)))
        
        while len(unmatched_dets) > 0 and len(unmatched_tracks) > 0:
            # Find best match
            max_iou = -1
            best_det = -1
            best_track = -1
            
            for i in unmatched_dets:
                for j in unmatched_tracks:
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        best_det = i
                        best_track = j
            
            if max_iou < self.iou_threshold:
                break
            
            matched_pairs.append((best_det, track_ids[best_track]))
            unmatched_dets.remove(best_det)
            unmatched_tracks.remove(best_track)
        
        unmatched_track_ids = [track_ids[i] for i in unmatched_tracks]
        
        return matched_pairs, unmatched_dets, unmatched_track_ids
    
    def _compute_iou(
        self,
        bbox1: tuple[float, float, float, float],
        bbox2: tuple[float, float, float, float],
    ) -> float:
        """Compute IOU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    def _get_confirmed_tracks(self) -> list[Track3D]:
        """Get list of confirmed tracks."""
        confirmed = []
        for track_state in self._tracks.values():
            if track_state.hits >= self.min_hits:
                confirmed.append(track_state.to_track3d())
        return confirmed


class _TrackState:
    """Internal state for a single track."""
    
    def __init__(
        self,
        track_id: int,
        detection: Detection,
        depth_map: np.ndarray,
        class_names: list[str],
    ):
        self.track_id = track_id
        self.bbox = detection.xyxy
        self.class_id = detection.class_id
        self.confidence = detection.score
        self.class_names = class_names
        
        # Get 3D position
        depth = get_bbox_depth(detection.xyxy, depth_map, method="center")
        x1, y1, x2, y2 = detection.xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        h, w = depth_map.shape[:2]
        self.x, self.y, self.z = pixel_to_3d(cx, cy, depth, None, w, h)
        
        # Tracking state
        self.hits = 1
        self.age = 0
        self.frames_tracked = 1
        self.is_new = True
        
        # Velocity tracking
        self.velocity: Optional[tuple[float, float, float]] = None
        self.last_pos = (self.x, self.y, self.z)
    
    def update(
        self,
        detection: Detection,
        depth_map: np.ndarray,
        dt: float,
    ) -> None:
        """Update track with new detection."""
        self.bbox = detection.xyxy
        self.confidence = detection.score
        self.class_id = detection.class_id
        
        # Update 3D position
        depth = get_bbox_depth(detection.xyxy, depth_map, method="center")
        x1, y1, x2, y2 = detection.xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        h, w = depth_map.shape[:2]
        self.x, self.y, self.z = pixel_to_3d(cx, cy, depth, None, w, h)
        
        # Calculate velocity
        if dt > 0:
            vx = (self.x - self.last_pos[0]) / dt
            vy = (self.y - self.last_pos[1]) / dt
            vz = (self.z - self.last_pos[2]) / dt
            self.velocity = (vx, vy, vz)
        
        self.last_pos = (self.x, self.y, self.z)
        
        # Update tracking state
        self.hits += 1
        self.age = 0
        self.frames_tracked += 1
        self.is_new = False
    
    def mark_missing(self) -> None:
        """Mark track as missing in current frame."""
        self.age += 1
        self.is_new = False
    
    def to_track3d(self) -> Track3D:
        """Convert to Track3D object."""
        class_name = self.class_names[self.class_id] if self.class_id < len(self.class_names) else f"class_{self.class_id}"
        
        return Track3D(
            id=self.track_id,
            x=self.x,
            y=self.y,
            z=self.z,
            bbox=self.bbox,
            class_id=self.class_id,
            class_name=class_name,
            confidence=self.confidence,
            velocity=self.velocity,
            is_new=self.is_new,
            frames_tracked=self.frames_tracked,
        )
