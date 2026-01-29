# scanLt

Realtime camera pipeline for detection + monocular depth ("3D") with pluggable backends.

## Install

Base (CPU, minimal):

```bash
pip install scanLt
```

Optional:

```bash
pip install scanLt[onnx]
pip install scanLt[torch]
pip install scanLt[mediapipe]
```

## Quickstart

```python
import scanlt

scanlt.run()
```

## Provide your own detector/depth

```python
import scanlt

# detector/depth must follow the library interfaces
scan3d.run(detector=my_detector, depth=my_depth, on_result=my_callback)
```
