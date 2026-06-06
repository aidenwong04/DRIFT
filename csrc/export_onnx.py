import torch
import torch.nn as nn
from transformers import AutoModel
from huggingface_hub import hf_hub_download

from model import DRIFT

# ── load your saved projection head from HuggingFace ───────────────────────

device = torch.device('cpu')

# replace with your actual HF repo path
ckpt_path = hf_hub_download(
    repo_id='aidenite/drift-dinov3-vitb16',
    filename='drift_projection_head_only.pth'
)

lean = torch.load(ckpt_path, map_location=device)
backbone_name = lean['backbone_name']   # 'facebook/dinov3-vitb16-pretrain-lvd1689m'

model = DRIFT(embed_dim=128, backbone_name=backbone_name).to(device)
model.projection_head.load_state_dict(lean['projection_head_state'])
model.eval()

print(f"Loaded projection head from epoch {lean['epoch']}, "
      f"best val loss {lean['best_val_loss']:.4f}")

# ── dummy input ─────────────────────────────────────────────────────────────

# DINOv2 ViT-B/14 expects 224x224; batch size 1 for export
dummy = torch.randn(1, 3, 224, 224)

# ── verify forward pass before export ───────────────────────────────────────

with torch.no_grad():
    features, projections = model(dummy)
    print(f"features shape:    {features.shape}")     # expect (1, 768)
    print(f"projections shape: {projections.shape}")  # expect (1, 128)

# ── export ──────────────────────────────────────────────────────────────────

torch.onnx.export(
    model,
    dummy,
    'drift_dinov2.onnx',
    input_names=['pixel_values'],
    output_names=['features', 'projections'],
    dynamic_axes={
        'pixel_values': {0: 'batch_size'},
        'features':     {0: 'batch_size'},
        'projections':  {0: 'batch_size'},
    },
    opset_version=17,

)
print("Exported to drift_dinov2.onnx")

# ── verify the exported graph ────────────────────────────────────────────────

import onnx
import onnxruntime as ort
import numpy as np

onnx_model = onnx.load('drift_dinov2.onnx')
onnx.checker.check_model(onnx_model)
print("ONNX graph check passed")

ort_session = ort.InferenceSession('drift_dinov2.onnx',
                                   providers=['CPUExecutionProvider'])

ort_inputs = {'pixel_values': dummy.numpy()}
ort_features, ort_projections = ort_session.run(None, ort_inputs)

# numerical parity check — should be <1e-5
pt_features, pt_projections = model(dummy)
print(f"Max feature delta:    {np.abs(ort_features - pt_features.numpy()).max():.2e}")
print(f"Max projection delta: {np.abs(ort_projections - pt_projections.numpy()).max():.2e}")