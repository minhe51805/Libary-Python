from __future__ import annotations

import hashlib
import os
import pathlib
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    sha256: str
    filename: str


def _default_cache_dir() -> pathlib.Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return pathlib.Path(base) / "scanlt3d" / "cache"
    return pathlib.Path(os.path.expanduser("~/.cache")) / "scanlt3d"


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")

    headers = {}
    if tmp.exists():
        headers["Range"] = f"bytes={tmp.stat().st_size}-"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as r:
        mode = "ab" if headers else "wb"
        with tmp.open(mode) as f:
            while True:
                chunk = r.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)

    tmp.replace(dst)


def ensure_model(spec: ModelSpec, *, cache_dir: Optional[str | os.PathLike[str]] = None) -> pathlib.Path:
    cache = pathlib.Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    path = cache / "models" / spec.filename

    if path.exists():
        got = _sha256_file(path)
        if got.lower() == spec.sha256.lower() and spec.sha256:
            return path
        try:
            path.unlink()
        except Exception:
            pass

    _download(spec.url, path)

    if spec.sha256:
        got = _sha256_file(path)
        if got.lower() != spec.sha256.lower():
            raise RuntimeError(
                f"Downloaded model checksum mismatch for {spec.name}. Expected {spec.sha256}, got {got}"
            )

    return path


def get_default_yolo_seg_specs() -> dict[str, ModelSpec]:
    # URLs are placeholders until you provide a hosting location.
    # sha256 can be left empty to skip checksum validation during development.
    return {
        "fast": ModelSpec(
            name="yolov8n-seg",
            url="https://huggingface.co/pleb631/onnxmodel/resolve/main/yolo/yolov8n-seg.onnx?download=true",
            sha256="",
            filename="yolov8n-seg.onnx",
        ),
        "balanced": ModelSpec(
            name="yolov8s-seg",
            url="https://huggingface.co/pleb631/onnxmodel/resolve/main/yolo/yolov8seg/yolov8s-seg.onnx?download=true",
            sha256="",
            filename="yolov8s-seg.onnx",
        ),
        "quality": ModelSpec(
            name="yolov8m-seg",
            url="https://huggingface.co/pleb631/onnxmodel/resolve/main/yolo/yolov8seg/yolov8m-seg.onnx?download=true",
            sha256="",
            filename="yolov8m-seg.onnx",
        ),
    }
