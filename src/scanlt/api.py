from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Iterator

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
    def __iter__(self) -> Iterator[np.ndarray]: ...


class WebcamSource:
    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        convert_bgr_to_rgb: bool = True,
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.convert_bgr_to_rgb = convert_bgr_to_rgb

    def __iter__(self) -> Iterator[np.ndarray]:
        try:
            import cv2  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "WebcamSource requires opencv-python. Install with: pip install 'scanlt3d[opencv]'"
            ) from e

        cap = cv2.VideoCapture(self.device_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open webcam device_id={self.device_id}")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Failed to read frame from webcam")

                if self.convert_bgr_to_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                yield frame
        finally:
            cap.release()


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
    show_preview: bool = True,
    window_name: str = "scanlt",
    show_depth: bool = True,
) -> None:

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
    preview_cv2 = None
    if show_preview:
        try:
            import cv2  # type: ignore

            preview_cv2 = cv2
        except Exception:
            preview_cv2 = None

    def _draw_detections_rgb(img: np.ndarray, detections: list[Detection]) -> np.ndarray:
        if preview_cv2 is None:
            return img

        cv2 = preview_cv2
        out = img.copy()
        h, w = out.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            x1i = int(max(0, min(w - 1, x1)))
            y1i = int(max(0, min(h - 1, y1)))
            x2i = int(max(0, min(w - 1, x2)))
            y2i = int(max(0, min(h - 1, y2)))
            cv2.rectangle(out, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            label = f"{det.class_id}:{det.score:.2f}"
            cv2.putText(out, label, (x1i, max(0, y1i - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return out

    for frame in source:
        t0 = _now_s()
        dets = detector.predict(frame)
        depth_map = depth.predict(frame, dets) if depth is not None else None

        t1 = _now_s()
        dt = max(t1 - t_last, 1e-9)
        inst_fps = 1.0 / dt
        fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)
        t_last = t1

        res = Result(frame=frame, detections=dets, depth=depth_map, fps=fps)
        if on_result is not None:
            on_result(res)

        if preview_cv2 is not None:
            cv2 = preview_cv2
            vis_rgb = _draw_detections_rgb(frame, dets)
            cv2.putText(
                vis_rgb,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

            panels = []
            # OpenCV expects BGR
            panels.append(cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))

            if show_depth and depth_map is not None:
                d = depth_map
                if d.dtype != np.float32:
                    d = d.astype(np.float32, copy=False)
                d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
                d_u8 = d_norm.astype(np.uint8)
                d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
                panels.append(d_color)

            vis = panels[0] if len(panels) == 1 else cv2.hconcat(panels)

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        n += 1
        if max_frames is not None and n >= max_frames:
            break

        # Simple pacing (best-effort)
        elapsed = _now_s() - t0
        sleep_s = frame_interval - elapsed
        if sleep_s > 0:
            import time

            time.sleep(sleep_s)

    if preview_cv2 is not None:
        try:
            preview_cv2.destroyWindow(window_name)
        except Exception:
            pass
