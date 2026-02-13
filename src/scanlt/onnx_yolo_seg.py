from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .api import Detection
from .backends import choose_backend


@dataclass(frozen=True)
class YoloSegConfig:
    img_size: int = 640
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    max_det: int = 50


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float, max_det: int) -> list[int]:
    # boxes: (N,4) xyxy
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep


def _letterbox_rgb(img: np.ndarray, new_size: int) -> tuple[np.ndarray, float, int, int]:
    # img: HWC RGB uint8
    h0, w0 = img.shape[:2]
    r = min(new_size / h0, new_size / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))

    pad_w = new_size - new_unpad[0]
    pad_h = new_size - new_unpad[1]
    dw = pad_w // 2
    dh = pad_h // 2

    try:
        import cv2  # type: ignore

        resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        out = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
        out[dh : dh + new_unpad[1], dw : dw + new_unpad[0]] = resized
        return out, r, dw, dh
    except Exception:
        # Fallback: resize via Pillow (always available as a dependency)
        from PIL import Image as _PILImage

        resized = np.array(_PILImage.fromarray(img).resize(new_unpad))
        out = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
        out[dh : dh + new_unpad[1], dw : dw + new_unpad[0]] = resized
        return out, r, dw, dh


class OnnxYoloSegDetector:
    """YOLOv8 segmentation ONNX detector.

    This implementation supports common YOLOv8-seg ONNX export layouts.
    """

    def __init__(
        self,
        model_path: str,
        *,
        backend: str = "auto",
        config: Optional[YoloSegConfig] = None,
    ):
        self.model_path = model_path
        self.backend_choice = choose_backend(backend)
        self.cfg = config or YoloSegConfig()

        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "onnxruntime is required for OnnxYoloSegDetector. Install with: pip install 'scanlt3d[onnx]'"
            ) from e

        providers = None
        bc = self.backend_choice.name
        if bc == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif bc == "dml":
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self._output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, frame: np.ndarray) -> list[Detection]:
        # frame: RGB uint8 HWC
        img, r, dw, dh = _letterbox_rgb(frame, self.cfg.img_size)
        inp = img.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...]

        outputs = self.session.run(None, {self.input_name: inp})

        # Identify det and proto outputs by ONNX output names first
        det = None
        proto = None
        output_map = dict(zip(self._output_names, outputs))

        if "output0" in output_map and "output1" in output_map:
            det = output_map["output0"]
            proto = output_map["output1"]
        else:
            # Fallback: shape-based heuristic
            for out in outputs:
                if out.ndim == 4 and out.shape[0] == 1:
                    proto = out
                elif out.ndim == 3 and out.shape[0] == 1 and det is None:
                    det = out

        if det is None:
            return []

        det = det[0]  # remove batch dim
        # YOLOv8 ONNX outputs (D, N) where D=attributes, N=anchors
        # We need (N, D) — rows are detections, columns are attributes
        if det.shape[0] < det.shape[1]:
            det = det.T

        # YOLOv8-seg format: [x,y,w,h, cls0..cls79, mask0..mask31]
        # No objectness score — confidence = max class score
        proto_c = int(proto.shape[1]) if proto is not None else 32
        if det.shape[1] < 4 + proto_c:
            return []

        mask_coeffs = det[:, -proto_c:]
        cls_scores = det[:, 4:-proto_c]

        if cls_scores.shape[1] == 0:
            return []

        conf = cls_scores.max(axis=1)
        class_id = cls_scores.argmax(axis=1).astype(np.int32)

        keep = conf >= self.cfg.conf_thres
        if not np.any(keep):
            return []

        det = det[keep]
        conf = conf[keep]
        class_id = class_id[keep]
        mask_coeffs = mask_coeffs[keep]

        # boxes xywh -> xyxy on letterboxed image
        xywh = det[:, 0:4]
        x = xywh[:, 0]
        y = xywh[:, 1]
        w = xywh[:, 2]
        h = xywh[:, 3]
        boxes = np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)

        # NMS
        keep_idx = _nms_xyxy(boxes, conf, self.cfg.iou_thres, self.cfg.max_det)
        boxes = boxes[keep_idx]
        conf = conf[keep_idx]
        class_id = class_id[keep_idx]
        mask_coeffs = mask_coeffs[keep_idx]

        # map boxes back to original frame
        # undo padding and scale
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= r

        # clip
        h0, w0 = frame.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0 - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0 - 1)

        # Compute masks if proto available
        masks = None
        if proto is not None:
            proto = proto[0]  # (C, Hp, Wp)
            # masks in proto space
            m = mask_coeffs @ proto.reshape(proto_c, -1)  # (N, Hp*Wp)
            m = _sigmoid(m).reshape(-1, proto.shape[1], proto.shape[2])  # (N, Hp, Wp)

            # upsample to letterbox size
            try:
                import cv2  # type: ignore

                up = []
                for i in range(m.shape[0]):
                    up.append(cv2.resize(m[i], (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_LINEAR))
                masks = np.stack(up, axis=0)
            except Exception:
                masks = m

            # remove padding and scale to original frame
            # crop padding
            masks = masks[:, dh : dh + int(round(h0 * r)), dw : dw + int(round(w0 * r))]
            # resize to original
            try:
                import cv2  # type: ignore

                up2 = []
                for i in range(masks.shape[0]):
                    up2.append(cv2.resize(masks[i], (w0, h0), interpolation=cv2.INTER_LINEAR))
                masks = np.stack(up2, axis=0)
            except Exception:
                pass

        detections: list[Detection] = []
        for i in range(boxes.shape[0]):
            det_mask = masks[i] if masks is not None else None
            det_obj = Detection(
                xyxy=(float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]), float(boxes[i, 3])),
                score=float(conf[i]),
                class_id=int(class_id[i]),
                mask=det_mask,
            )
            detections.append(det_obj)

        return detections
