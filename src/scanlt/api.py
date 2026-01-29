from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    score: float
    class_id: int


@dataclass(frozen=True)
class Result:
    frame: np.ndarray
    detections: list[Detection]
    depth: Optional[np.ndarray]
    fps: float


class Detector(Protocol):
    def predict(self, frame: np.ndarray) -> list[Detection]: ...


class DepthEstimator(Protocol):
    def predict(self, frame: np.ndarray, detections: Optional[list[Detection]] = None) -> np.ndarray: ...


class FrameSource(Protocol):
    def __iter__(self): ...


def _now_s() -> float:
    import time

    return time.perf_counter()


class _DummyCamera:
    def __init__(self, size: tuple[int, int] = (480, 640)):
        self.h, self.w = size

    def __iter__(self):
        t0 = _now_s()
        while True:
            t = _now_s() - t0
            frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            x = int((np.sin(t) * 0.4 + 0.5) * (self.w - 80))
            y = int((np.cos(t) * 0.4 + 0.5) * (self.h - 80))
            frame[y : y + 80, x : x + 80, 1] = 255
            yield frame


class _NoopDetector:
    def predict(self, frame: np.ndarray) -> list[Detection]:
        return []


class _NoopDepth:
    def predict(self, frame: np.ndarray, detections: Optional[list[Detection]] = None) -> np.ndarray:
        h, w = frame.shape[:2]
        return np.zeros((h, w), dtype=np.float32)


def run(
    *,
    source: Optional[FrameSource] = None,
    detector: Optional[Detector] = None,
    depth: Optional[DepthEstimator] = None,
    on_result: Optional[Callable[[Result], None]] = None,
    target_fps: float = 20.0,
    max_frames: Optional[int] = None,
) -> None:
    """Run the realtime loop.

    - No OpenCV required.
    - Default source is a dummy camera (so import/run never fails).
    - Provide your own source/detector/depth for real usage.
    """

    if source is None:
        source = _DummyCamera()
    if detector is None:
        detector = _NoopDetector()
    if depth is None:
        depth = _NoopDepth()

    frame_interval = 1.0 / max(target_fps, 1e-6)

    t_last = _now_s()
    fps = 0.0
    n = 0
    for frame in source:
        t0 = _now_s()
        dets = detector.predict(frame)
        depth_map = depth.predict(frame, dets) if depth is not None else None

        t1 = _now_s()
        dt = max(t1 - t_last, 1e-9)
        inst_fps = 1.0 / dt
        fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)
        t_last = t1

        if on_result is not None:
            on_result(Result(frame=frame, detections=dets, depth=depth_map, fps=fps))

        n += 1
        if max_frames is not None and n >= max_frames:
            break

        # Simple pacing (best-effort)
        elapsed = _now_s() - t0
        sleep_s = frame_interval - elapsed
        if sleep_s > 0:
            import time

            time.sleep(sleep_s)
