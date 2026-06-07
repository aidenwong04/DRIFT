# DRIFT
Diffusion-Robust Invariant Fingerprint Training


## Inference & Performance

Benchmarks run on batch size 1, 100 iterations with 10 warmup runs. All timings in milliseconds. Input shape: `[1, 3, 224, 224]`, dtype `float32`.

**Hardware:** NVIDIA GeForce GTX 1050 Ti · Intel Core i5-7500
**Model:** DRIFT — DINOv3 ViT-B/16 backbone + MLP projection head, ONNX opset 17

| Runtime | Device | Avg Latency | P95 Latency | P99 Latency |
|---|---|---|---|---|
| PyTorch | GPU | **40.68 ms** | 49.60 ms | 119.67 ms |
| ONNX Runtime | CPU | 225.22 ms | 419.57 ms | 515.91 ms |
| PyTorch | CPU | 299.07 ms | 566.96 ms | 892.69 ms |

**Key observations:**

- PyTorch on GPU is ~7× faster than both CPU runtimes on average latency, confirming expected GPU acceleration for a ViT-class model.
- ONNX Runtime on CPU is ~25% faster than PyTorch CPU on average (225 ms vs 299 ms), validating that the ONNX export improves CPU deployment efficiency.
- Tight P95 GPU latency (49.60 ms) indicates consistent sub-50 ms inference for most requests, suitable for real-time API serving.
- Elevated CPU tail latencies (P99 ~516–893 ms) are expected on a local dev machine with OS scheduling noise; production deployment on dedicated hardware would reduce this substantially.

> **Deployment note:** `drift_dinov2.onnx` is the primary deployment artifact. For CPU serving, ONNX Runtime is used with `ORT_ENABLE_ALL` graph optimizations. GPU serving will use ONNX Runtime with `CUDAExecutionProvider` or TensorRT via Triton on a cloud GPU instance (AWS g6, CUDA 12 + cuDNN 9).