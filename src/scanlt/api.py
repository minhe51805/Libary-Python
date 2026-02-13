from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Iterator

import numpy as np

from ._accel import (
    bgr_to_rgb,
    rgb_to_bgr,
    generate_dummy_frame,
    draw_bboxes_on_frame,
    normalize_depth_map,
    depth_to_colormap_jet,
)


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    mask: Optional[np.ndarray] = None


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
                    frame = bgr_to_rgb(frame)

                yield frame
        finally:
            cap.release()


def demo_webcam(
    *,
    profile: str = "fast",
    device_id: int = 0,
    width: int = 640,
    height: int = 480,
    backend: str = "auto",
    target_fps: float = 20.0,
) -> None:
    """Run webcam demo with instance segmentation mask.

    - Downloads a YOLOv8-seg ONNX model (profile: fast/balanced/quality) on first run and caches it.
    - Requires OpenCV for webcam + preview.
    """

    from .model_zoo import ensure_model, get_default_yolo_seg_specs
    from .onnx_yolo_seg import OnnxYoloSegDetector

    specs = get_default_yolo_seg_specs()
    if profile not in specs:
        raise ValueError(f"Unknown profile '{profile}'. Choose one of: {', '.join(specs.keys())}")

    model_path = ensure_model(specs[profile])
    det = OnnxYoloSegDetector(str(model_path), backend=backend)

    src = WebcamSource(device_id=device_id, width=width, height=height, convert_bgr_to_rgb=True)

    run(
        source=src,
        detector=det,
        depth=None,
        target_fps=target_fps,
        show_preview=True,
        show_depth=False,
        window_name="scanlt3d",
    )


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
            yield generate_dummy_frame(self.h, self.w, t)


class _NoopDetector:
    def predict(self, frame: np.ndarray) -> list[Detection]:
        return []


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
    show_depth: bool = False,
) -> None:
    """Run the realtime loop.

    Notes:
    - If `source` is None, a dummy source is used (never fails).
    - If `detector` is None, a noop detector is used (no detections).
    - If `depth` is None, depth is disabled (no fake depth panel).
    """

    if source is None:
        source = _DummyCamera()
    if detector is None:
        detector = _NoopDetector()

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

    def _overlay_mask_rgb(out: np.ndarray, mask01: np.ndarray, color: tuple[int, int, int]) -> None:
        # out: RGB uint8
        if mask01.dtype != np.float32:
            m = mask01.astype(np.float32, copy=False)
        else:
            m = mask01
        if m.max() > 1.0:
            m = m / 255.0
        m = np.clip(m, 0.0, 1.0)
        if m.ndim == 3:
            m = m[..., 0]

        alpha = 0.45
        for c, col in enumerate(color):
            out[..., c] = (out[..., c] * (1.0 - alpha * m) + col * (alpha * m)).astype(np.uint8)

    def _draw_detections_rgb(img: np.ndarray, detections: list[Detection]) -> np.ndarray:
        if preview_cv2 is None:
            return img

        cv2 = preview_cv2
        out = img.copy()
        h, w = out.shape[:2]

        # Show a clear hint if no detector configured
        if detector is None or isinstance(detector, _NoopDetector):
            cv2.putText(
                out,
                "No detector configured (pass detector=...)",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            x1i = int(max(0, min(w - 1, x1)))
            y1i = int(max(0, min(h - 1, y1)))
            x2i = int(max(0, min(w - 1, x2)))
            y2i = int(max(0, min(h - 1, y2)))

            # If detector attached a mask
            mask = det.mask
            if mask is not None:
                _overlay_mask_rgb(out, mask, (0, 255, 0))

            cv2.rectangle(out, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            label = f"{det.class_id}:{det.score:.2f}"
            cv2.putText(
                out,
                label,
                (x1i, max(0, y1i - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        return out

    for frame in source:
        t0 = _now_s()
        dets = detector.predict(frame) if detector is not None else []
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

            panels = [rgb_to_bgr(vis_rgb)]

            if show_depth and depth_map is not None:
                d_u8 = normalize_depth_map(depth_map)
                d_color = depth_to_colormap_jet(d_u8)
                panels.append(d_color)

            vis = panels[0] if len(panels) == 1 else cv2.hconcat(panels)

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        n += 1
        if max_frames is not None and n >= max_frames:
            break

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
