from __future__ import annotations

from dataclasses import dataclass

from .hw import get_hardware_info


@dataclass(frozen=True)
class BackendChoice:
    name: str
    reason: str


def choose_backend(preferred: str = "auto") -> BackendChoice:
    """Pick best available backend by environment.

    This does NOT import heavyweight deps unless needed.

    Supported names:
    - auto, cpu
    - cuda (onnxruntime-gpu or torch cuda)
    - dml (onnxruntime-directml) [Windows]
    - mps (torch MPS) [macOS Apple Silicon]
    - coreml (not implemented yet; reserved)

    Users can override with env var SCANLT_BACKEND.
    """

    hw = get_hardware_info()
    forced = (hw.env_force_backend or "").strip().lower() or None
    if forced is not None:
        preferred = forced

    preferred = preferred.lower().strip()

    if preferred not in {"auto", "cpu", "cuda", "dml", "mps", "coreml"}:
        return BackendChoice("cpu", f"unknown preferred backend '{preferred}', falling back to cpu")

    if preferred == "cpu":
        return BackendChoice("cpu", "user selected cpu")

    # macOS Apple Silicon: MPS is the most realistic default for torch-based users.
    if preferred in {"auto", "mps"} and hw.is_apple_silicon:
        try:
            import torch  # type: ignore

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return BackendChoice("mps", "torch MPS available on Apple Silicon")
        except Exception:
            if preferred == "mps":
                return BackendChoice("cpu", "torch MPS not available; falling back to cpu")

    # Windows DirectML: best shot for AMD/Intel GPUs without CUDA.
    if preferred in {"auto", "dml"} and hw.is_windows:
        try:
            import onnxruntime as ort  # type: ignore

            providers = [p.lower() for p in ort.get_available_providers()]
            if any("directml" in p for p in providers):
                return BackendChoice("dml", "onnxruntime DirectML provider available")
        except Exception:
            if preferred == "dml":
                return BackendChoice("cpu", "DirectML not available; falling back to cpu")

    # CUDA (NVIDIA)
    if preferred in {"auto", "cuda"}:
        # Try ORT CUDA EP first
        try:
            import onnxruntime as ort  # type: ignore

            providers = [p.lower() for p in ort.get_available_providers()]
            if any("cuda" in p for p in providers):
                return BackendChoice("cuda", "onnxruntime CUDA provider available")
        except Exception:
            pass

        # Then torch cuda
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return BackendChoice("cuda", "torch CUDA available")
        except Exception:
            if preferred == "cuda":
                return BackendChoice("cpu", "CUDA not available; falling back to cpu")

    if preferred == "coreml":
        return BackendChoice("cpu", "coreml backend reserved (not implemented); falling back to cpu")

    return BackendChoice("cpu", "no accelerated backend detected")
