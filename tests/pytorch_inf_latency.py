import time
import numpy as np
import torch
from huggingface_hub import hf_hub_download

import sys
import os

target_folder = os.path.abspath("../csrc")
sys.path.append(target_folder)

from model import DRIFT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = hf_hub_download(
    repo_id='aidenite/drift-dinov3-vitb16',
    filename='drift_projection_head_only.pth'
)

lean = torch.load(ckpt_path, map_location=device)
backbone_name = lean['backbone_name']   # 'facebook/dinov3-vitb16-pretrain-lvd1689m'

model = DRIFT(embed_dim=128, backbone_name=backbone_name).to(device)
model.projection_head.load_state_dict(lean['projection_head_state'])
model.eval()

batch_size = 1
input_tensor = torch.randn(batch_size, 3, 224, 224)

# warmup
with torch.inference_mode():
    for _ in range(10):
        _ = model(input_tensor)

num_iterations = 100
latencies = []

with torch.inference_mode():
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = model(input_tensor)
        latencies.append((time.perf_counter() - start) * 1000)

print(f"Average Latency: {np.mean(latencies):.2f} ms")
print(f"P95 Latency: {np.percentile(latencies, 95):.2f} ms")
print(f"P99 Latency: {np.percentile(latencies, 99):.2f} ms")