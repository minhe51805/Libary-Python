# scanLt

Realtime camera pipeline for **detection** + **monocular depth** ("3D") with pluggable backends.

- Project/package name (PyPI): `scanLt`
- Import name (Python): `scanlt`

This library **does not require OpenCV**.

## Install

### Minimal (always works)

CPU-only, no special hardware acceleration:

```bash
pip install scanlt3d
```

### Optional extras

You can install extra backends depending on your machine.

```bash
pip install "scanlt3d[onnx]"       # ONNX Runtime (recommended cross-platform)
pip install "scanlt3d[torch]"      # PyTorch (CUDA/MPS support)
pip install "scanlt3d[mediapipe]"  # (planned) camera source via MediaPipe
```

> Note: Hardware acceleration depends on which runtime is installed and available on your system (CUDA/DirectML/MPS).

## Quickstart (works immediately)

```python
import scanlt

# Runs a realtime loop with a built-in dummy camera source.
# This ensures `import + run()` never fails even if no webcam/backend is available.
scanlt.run()
```

## Use by hardware (CPU / NVIDIA / Windows iGPU / Mac M)

scanLt can auto-detect the best available backend:

```python
import scanlt

print(scanlt.choose_backend())
```

### 1) CPU (Intel/AMD)

Install:

```bash
pip install scanlt3d
# recommended runtime
pip install "scanLt[onnx]"
```

What to expect:
- Best compatibility.
- Realtime depends heavily on your detector/depth model sizes.
- Use lower resolution / run depth less frequently for higher FPS.

### 2) NVIDIA GPU (CUDA)

Two typical options:

- ONNX Runtime CUDA (best for ONNX models)
- PyTorch CUDA (best for torch models)

Install (choose one):

```bash
# Option A: PyTorch CUDA
pip install "scanLt[torch]"

# Option B: ONNX Runtime (you must install a CUDA-enabled onnxruntime build)
pip install "scanLt[onnx]"
```

Notes:
- If `choose_backend()` returns `cuda`, scanLt detected a CUDA-capable runtime.
- CUDA packaging varies by OS/driver; if CUDA runtime is not available, scanLt falls back to CPU.

### 3) Windows Intel/AMD iGPU (DirectML)

scanLt can pick `dml` **if** your ONNX Runtime installation exposes a DirectML provider.

Install:

```bash
pip install "scanLt[onnx]"
```

Notes:
- DirectML is Windows-specific.
- If DirectML is not available, scanLt falls back to CPU.

### 4) macOS Apple Silicon (M1/M2/M3)

Install:

```bash
pip install scanlt3d
pip install "scanLt[torch]"
```

Notes:
- scanLt will try to use `mps` if PyTorch MPS is available.
- If MPS is not available or unsupported by the model, it falls back to CPU.

## Force backend (override auto-detect)

You can override backend selection via environment variable:

- `SCAN3D_BACKEND=cpu`
- `SCAN3D_BACKEND=cuda`
- `SCAN3D_BACKEND=dml`
- `SCAN3D_BACKEND=mps`

Example:

```bash
# Windows PowerShell
setx SCAN3D_BACKEND cuda
```

(Then restart your terminal.)

## Provide your own detector/depth

scanLt is designed to let you plug in your own models.

### Minimal interfaces

- `Detector.predict(frame) -> list[Detection]`
- `DepthEstimator.predict(frame, detections=None) -> depth_map`

### Example

```python
import scanlt

class MyDetector:
    def predict(self, frame):
        # return list of scanlt.api.Detection
        return []

class MyDepth:
    def predict(self, frame, detections=None):
        import numpy as np
        h, w = frame.shape[:2]
        return np.zeros((h, w), dtype=np.float32)


def on_result(res):
    # res.frame: np.ndarray (H,W,3)
    # res.detections: list
    # res.depth: np.ndarray (H,W) or None
    # res.fps: float
    print(res.fps)


scanlt.run(detector=MyDetector(), depth=MyDepth(), on_result=on_result, max_frames=100)
```

## Troubleshooting

### `pip install` succeeds but `choose_backend()` is still CPU

This usually means:
- CUDA / DirectML / MPS runtime is not installed or not detected
- your model runtime doesn't support the needed execution provider

scanLt will **always** fall back to CPU to stay usable.

### I want real webcam input

Current default `run()` uses a dummy source. For real camera sources, you will plug in a `FrameSource` (or enable the MediaPipe source once implemented).
