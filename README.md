# DRIFT

![Build](https://img.shields.io/github/actions/workflow/status/aidenwong04/DRIFT/deploy.yml?branch=main&label=build)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Robust AI-generated image attribution — identifying which of 10 commercial generators produced an image, even after compression, resizing, or social media re-upload.**

[**Live Demo →**](https://drift-api-765464933028.asia-east1.run.app/docs)

---

## The Problem

AI-generated images are rarely distributed in pristine form. Before reaching a viewer, they are typically JPEG-compressed, resized, re-uploaded through social platforms, or screenshotted. These post-processing transformations degrade the subtle generator-specific artifacts that attribution models rely on. A standard ResNet-50 baseline drops from **98.93% to 71.47%** accuracy when images are degraded.

## The Approach

DRIFT addresses this robustness gap by combining two components:

- **A six-type stochastic degradation pipeline** — applied during training to simulate JPEG compression, Gaussian blur, Gaussian noise, re-upload (downscale + upscale), screenshot simulation, and random combinations thereof.
- **Supervised Contrastive Learning (SupCon)** — which trains the encoder to produce transformation-invariant fingerprint embeddings by treating all images from the same generator as positives. After contrastive training, a lightweight linear probe (768→10) is trained on top of frozen DINOv2 ViT-B/14 features for final 10-way attribution.

## Results

Evaluated on the [WILD dataset](https://arxiv.org/abs/2504.19595) closed set (10 generators, 1,500 test samples):

| Method | Clean Accuracy | Degraded Accuracy | Degradation Drop |
|---|---|---|---|
| ResNet-50 Baseline | 98.93% | 71.47% | −27.46 pp |
| CLIP Baseline | 91.40% | 69.13% | −22.27 pp |
| **DRIFT (ours)** | **95.80%** | **94.40%** | **−1.40 pp** |

DRIFT delivers a **~23-point robustness gain** over a standard supervised baseline with only a ~3-point clean-accuracy trade-off.

## Supported Generators

Adobe Firefly · DALL-E 3 · Flux.1 · Flux.1.1 Pro · Freepik · Leonardo AI · Midjourney · Stable Diffusion 3.5 · Stable Diffusion XL · Starry AI

---

## Architecture

```
Image Upload (POST /predict)
        ↓
FastAPI (Uvicorn, Python 3.12)
        ↓
DINOv3 ViT-B/16 → ONNX Runtime (CPU)
        ↓
Linear Probe (768 → 10 classes)
        ↓
JSON Response { predicted_class, confidence, latency_ms }
        ↓
Docker Container → GCP Cloud Run (asia-east1)
```

The DINOv3 backbone is exported to ONNX (opset 17) and served via ONNX Runtime with `ORT_ENABLE_ALL` graph optimizations. The linear probe is loaded separately as a PyTorch checkpoint. Both are baked into the Docker image at build time.

## Production Stack

| Layer | Tool |
|---|---|
| Model backbone | DINOv3 ViT-B/14 (HuggingFace Transformers) |
| Export format | ONNX (opset 17) |
| Inference runtime | ONNX Runtime (CPU) |
| API framework | FastAPI + Uvicorn |
| Containerisation | Docker |
| Container registry | Docker Hub |
| Cloud serving | GCP Cloud Run (serverless, asia-east1) |
| CI/CD | GitHub Actions (build → push → deploy on every commit to `main`) |

---

## Getting Started

### Prerequisites

- Python 3.12
- Docker (for containerised serving)

### Local Setup

```bash
# Clone the repo
git clone https://github.com/aidenwong04/DRIFT.git
cd DRIFT

# Install dependencies
pip install -r requirements.txt

# Run the API locally (requires models/ folder — see below)
uvicorn app.main:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) to access the Swagger UI.

### Model Files

The ONNX model and linear probe checkpoint are not committed to the repository due to file size. They are attached as assets to the [v1.0 release](https://github.com/aidenwong04/DRIFT/releases/tag/v1.0):

- `drift_dinov2.onnx`
- `drift_dinov2.onnx.data`
- `linear_probe.pth`

Download and place them in a `models/` folder at the project root.

### Docker

```bash
# Build image (downloads model files automatically)
docker build -t drift .

# Run container
docker run -p 8000:8000 drift
```

### API Usage

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@your_image.jpg"
```

**Response:**
```json
{
  "predicted_class": "Midjourney",
  "confidence": 0.94,
  "latency_ms": 231.4
}
```

---

## Inference & Performance

Benchmarks run on batch size 1, 100 iterations with 10 warmup runs. All timings in milliseconds. Input shape: `[1, 3, 224, 224]`, dtype `float32`.

**Hardware:** NVIDIA GeForce GTX 1050 Ti · Intel Core i5-7500  
**Model:** DRIFT — DINOv2 ViT-B/14 backbone + linear probe, ONNX opset 17

| Runtime | Device | Avg Latency | P95 Latency | P99 Latency |
|---|---|---|---|---|
| PyTorch | GPU | **40.68 ms** | 49.60 ms | 119.67 ms |
| ONNX Runtime | CPU | 225.22 ms | 419.57 ms | 515.91 ms |
| PyTorch | CPU | 299.07 ms | 566.96 ms | 892.69 ms |

**Key observations:**

- PyTorch on GPU is ~7× faster than both CPU runtimes, confirming expected GPU acceleration for a ViT-class model.
- ONNX Runtime on CPU is ~25% faster than PyTorch CPU (225 ms vs 299 ms), validating that the ONNX export improves CPU deployment efficiency.
- Tight P95 GPU latency (49.60 ms) indicates consistent sub-50 ms inference, suitable for real-time API serving.
- Elevated CPU tail latencies (P99 ~516–893 ms) are expected on a local dev machine; production deployment on dedicated hardware would reduce this substantially.

---

## Scope & Limitations

DRIFT is a **closed-set** attribution system. It identifies among the 10 generators it was trained on; images from unseen generators or non-portrait content may yield unreliable predictions. Low-confidence predictions (below a configurable threshold) return `"unknown"` rather than a forced class assignment.

This project is designed as a detection-side complement to provenance-based approaches like [C2PA Content Credentials](https://c2pa.org/).

---

## Production Roadmap

- **Drift monitoring** — Evidently AI integration to track feature distribution shifts in production embeddings against the WILD training baseline
- **Experiment tracking** — MLflow for logging retraining runs, hyperparameters, and model versioning
- **GPU serving** — ONNX Runtime with `CUDAExecutionProvider` or TensorRT via Triton Inference Server on a cloud GPU instance for sub-50 ms CPU latency
- **Open-set detection** — rejection mechanism for images from unseen generators using embedding distance thresholds

---

## Contributors

Developed at Boston University in collaboration with Additya Singh, Martin So, Daniel Chen, and Aohan Mei.

---

## License

MIT
