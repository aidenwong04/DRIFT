## About

**DRIFT** (Diffusion-Robust Invariant Fingerprint Training) is a production-grade synthetic image attribution system that identifies which of 10 commercial AI image generators produced a given image, even after the image has been compressed, resized, screenshotted, or re-uploaded through social media platforms.

### The Problem

AI-generated images are rarely distributed in pristine form. Before reaching a viewer, they are typically JPEG-compressed, resized, re-uploaded through social platforms, or screenshotted. These post-processing transformations degrade the subtle generator-specific artifacts that attribution models rely on. Prior methods achieve high accuracy on clean images but collapse under realistic degradation; a standard ResNet-50 baseline drops from **98.93% to 71.47%** accuracy when images are degraded.

### The Approach

DRIFT addresses this robustness gap by combining two components:

- **A six-type stochastic degradation pipeline** — applied during training to simulate JPEG compression, Gaussian blur, Gaussian noise, re-upload (downscale + upscale), screenshot simulation, and random combinations thereof.
- **Supervised Contrastive Learning (SupCon)** — which trains the encoder to produce transformation-invariant fingerprint embeddings by treating all images from the same generator as positives. After contrastive training, a lightweight linear probe (768→10) is trained on top of frozen DINOv2 ViT-B/14 features for final 10-way attribution.

### Results

Evaluated on the [WILD dataset](https://arxiv.org/abs/2504.19595) closed set (10 generators, 1,500 test samples):

| Method | Clean Accuracy | Degraded Accuracy | Degradation Drop |
|---|---|---|---|
| ResNet-50 Baseline | 98.93% | 71.47% | −27.46 pp |
| CLIP Baseline | 91.40% | 69.13% | −22.27 pp |
| **DRIFT (ours)** | **95.80%** | **94.40%** | **−1.40 pp** |

DRIFT delivers a **~23-point robustness gain** over a standard supervised baseline with only a ~3-point clean-accuracy trade-off.

### Supported Generators

Adobe Firefly · DALL-E 3 · Flux.1 · Flux.1.1 Pro · Freepik · Leonardo AI · Midjourney · Stable Diffusion 3.5 · Stable Diffusion XL · Starry AI

### Production Stack

The research model is wrapped in a full production inference pipeline:

- **Inference:** DINOv3 ViT-B/16 backbone exported to ONNX, served via a FastAPI REST API (`POST /predict`)
- **Response:** Returns predicted generator, confidence score, and inference latency
- **Low-confidence handling:** Predictions below a configurable threshold return `"unknown"` rather than a forced class assignment

### Scope & Limitations

DRIFT is a **closed-set** attribution system. It identifies among the 10 generators it was trained on; images from unseen generators or non-portrait content may yield unreliable predictions.

This project is designed as a detection-side complement to provenance-based approaches like [C2PA Content Credentials](https://c2pa.org/).


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