"""
gradcam_analysis.py — Grad-CAM + Fourier Spectrum analysis for DRIFT project.

Generates:
  1. Grad-CAM heatmaps overlaid on input images (one grid per generator)
  2. Fourier frequency spectrum (clean vs degraded comparison)
  3. Radial average power spectrum across all generators

Usage (DRIFT contrastive model):
    python gradcam_analysis.py \
        --drift_checkpoint .../checkpoints/best_model_20260417_155351.pth \
        --probe_checkpoint .../checkpoints/probe_best_20260418_174843.pth \
        --image_dir .../data/Data/Closed_Set \
        --out_dir .../figures/interpretability \
        --n_samples 3

Usage (ResNet baseline):
    python gradcam_analysis.py \
        --baseline_checkpoint .../checkpoints/baseline_resnet_best_20260426_174958.pth \
        --image_dir .../data/Data/Closed_Set \
        --out_dir .../figures/interpretability \
        --n_samples 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── add csrc to path so we can import DRIFT models ──────────────────────────
CSRC = Path(__file__).resolve().parent / 'csrc'
if str(CSRC) not in sys.path:
    sys.path.insert(0, str(CSRC))

from model import DRIFT, LinearProbe
from baseline_model import BaselineResNet

# ── constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Adobe Firefly", "Dall-E 3", "Flux.1", "Flux.1.1 Pro", "Freepik",
    "Leonardo AI", "Midjourney", "Stable Diffusion 3.5",
    "Stable Diffusion XL", "Starry AI",
]
NUM_CLASSES = len(CLASS_NAMES)

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
PREPROCESS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    NORMALIZE,
])

# ── Grad-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM for ResNet-50.
    Hooks into model.layer4 (the last convolutional block).
    Works for both DRIFT+LinearProbe and BaselineResNet.
    """

    def __init__(self, model: nn.Module, is_baseline: bool = False):
        self.model = model
        self.model.eval()
        self._activations = None
        self._gradients = None

        # find the right layer4 depending on model wrapper
        if is_baseline:
            target_layer = model.model.layer4  # BaselineResNet wraps resnet as self.model
        else:
            target_layer = model.backbone.layer4  # LinearProbe exposes backbone directly

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, img_tensor: torch.Tensor, target_class: int = None):
        """
        img_tensor: (1, 3, H, W) — already normalized
        target_class: if None, uses predicted class
        Returns:
            cam: np.ndarray (H, W) in [0, 1]
            pred_class: int
            confidence: float
        """
        self.model.zero_grad()
        img_tensor = img_tensor.clone().requires_grad_(True)

        # LinearProbe wraps backbone in torch.no_grad() which kills gradients.
        # Bypass by calling backbone + classifier directly so gradients flow.
        if hasattr(self.model, 'backbone') and hasattr(self.model, 'classifier'):
            features = self.model.backbone(img_tensor)
            logits = self.model.classifier(features)
        else:
            logits = self.model(img_tensor)

        pred_class = logits.argmax(dim=1).item()
        confidence = torch.softmax(logits, dim=1)[0, pred_class].item()

        if target_class is None:
            target_class = pred_class

        self.model.zero_grad()
        logits[0, target_class].backward()

        # Grad-CAM: weight each activation channel by its global-average gradient
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze(0)  # (h, w)
        cam = F.relu(cam)

        # normalise to [0, 1] and resize to input spatial dims
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        H, W = img_tensor.shape[2], img_tensor.shape[3]
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
        ).squeeze().cpu().numpy()

        return cam, pred_class, confidence

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_cam(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend Grad-CAM heatmap onto original image. Returns H×W×3 uint8 array."""
    img_np = np.array(img_pil.resize((256, 256))).astype(np.float32) / 255.0
    heatmap = plt.cm.jet(cam)[..., :3]  # (H, W, 3) RGB
    blended = (1 - alpha) * img_np + alpha * heatmap
    blended = np.clip(blended, 0, 1)
    return (blended * 255).astype(np.uint8)


# ── Fourier helpers ───────────────────────────────────────────────────────────
def compute_fft_spectrum(img_pil: Image.Image) -> np.ndarray:
    """
    Returns the 2D log-magnitude Fourier spectrum of the grayscale image.
    Shifted so DC is in the centre.
    """
    gray = np.array(img_pil.convert('L').resize((256, 256))).astype(np.float32)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(fft))
    return magnitude


def radial_average(spectrum: np.ndarray) -> np.ndarray:
    """
    Collapse a 2D centred spectrum to a 1D radial average.
    Returns array of length = half the diagonal.
    """
    H, W = spectrum.shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cx, cy)
    radial = np.array([spectrum[r == i].mean() if np.any(r == i) else 0
                       for i in range(max_r)])
    return radial


# ── image loading helpers ─────────────────────────────────────────────────────
def load_samples(image_dir: Path, n_samples: int):
    """
    Returns dict: class_name → list of (PIL.Image, path) of length n_samples.
    Picks the first n_samples .png files found per generator folder.
    """
    samples = {}
    for cls in CLASS_NAMES:
        cls_dir = image_dir / cls
        if not cls_dir.exists():
            print(f"[WARN] Directory not found: {cls_dir}")
            continue
        imgs = sorted(cls_dir.glob('*.png'))[:n_samples]
        if not imgs:
            print(f"[WARN] No .png images found in {cls_dir}")
            continue
        samples[cls] = [(Image.open(p).convert('RGB'), p) for p in imgs]
    return samples


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_gradcam_grid(samples, gradcam_fn, out_path: Path):
    """
    One big grid: rows = generators, cols = n_samples × 2 (original | CAM overlay).
    """
    classes = list(samples.keys())
    n_cls = len(classes)
    n_samp = max(len(v) for v in samples.values())
    n_cols = n_samp * 2  # original + overlay per sample

    fig, axes = plt.subplots(
        n_cls, n_cols,
        figsize=(n_cols * 2.2, n_cls * 2.2),
        squeeze=False
    )
    fig.patch.set_facecolor('#0d0d0d')

    for row, cls in enumerate(classes):
        print(f'  [{row+1}/{n_cls}] Grad-CAM: {cls} ...', flush=True)
        for col_pair, (img_pil, _) in enumerate(samples[cls]):
            img_tensor = PREPROCESS(img_pil).unsqueeze(0)
            cam, pred, conf = gradcam_fn(img_tensor)
            overlay = overlay_cam(img_pil, cam)

            col_orig = col_pair * 2
            col_over = col_pair * 2 + 1

            # original
            ax = axes[row][col_orig]
            ax.imshow(img_pil.resize((256, 256)))
            ax.axis('off')
            if col_pair == 0:
                ax.set_ylabel(cls, color='white', fontsize=7, rotation=0,
                              labelpad=60, va='center')

            # overlay
            ax2 = axes[row][col_over]
            ax2.imshow(overlay)
            ax2.set_title(
                f"pred: {CLASS_NAMES[pred][:10]}\n{conf:.2f}",
                fontsize=5.5, color='#aaffaa', pad=2
            )
            ax2.axis('off')

    # column headers
    for cp in range(n_samp):
        axes[0][cp * 2].set_title(f'Sample {cp + 1}', color='white', fontsize=7)
        axes[0][cp * 2 + 1].set_title(f'Grad-CAM {cp + 1}', color='#ffcc44', fontsize=7)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('#0d0d0d')

    plt.suptitle('Grad-CAM Attribution Analysis — DRIFT', color='white',
                 fontsize=13, y=1.01, fontweight='bold')
    plt.tight_layout(pad=0.4)
    plt.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[✓] Grad-CAM grid saved → {out_path}')


def plot_fourier_grid(samples, out_path: Path):
    """
    2D Fourier spectrum grid: rows = generators, cols = n_samples.
    """
    classes = list(samples.keys())
    n_cls = len(classes)
    n_samp = max(len(v) for v in samples.values())

    fig, axes = plt.subplots(
        n_cls, n_samp,
        figsize=(n_samp * 2.5, n_cls * 2.2),
        squeeze=False
    )
    fig.patch.set_facecolor('#0d0d0d')

    for row, cls in enumerate(classes):
        print(f'  [{row+1}/{n_cls}] Fourier: {cls} ...', flush=True)
        for col, (img_pil, _) in enumerate(samples[cls]):
            spectrum = compute_fft_spectrum(img_pil)
            ax = axes[row][col]
            ax.imshow(spectrum, cmap='inferno', aspect='auto')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(cls, color='white', fontsize=7,
                              rotation=0, labelpad=60, va='center')
            if row == 0:
                ax.set_title(f'Sample {col + 1}', color='white', fontsize=7)

    plt.suptitle('Fourier Frequency Spectrum by Generator — DRIFT',
                 color='white', fontsize=13, y=1.01, fontweight='bold')
    plt.tight_layout(pad=0.4)
    plt.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[✓] Fourier grid saved → {out_path}')


def plot_radial_spectrum(samples, out_path: Path):
    """
    1D radial-average power spectrum, one curve per generator (averaged over n_samples).
    Good for spotting high-frequency artefact differences between generators.
    """
    COLORS = plt.cm.tab10(np.linspace(0, 1, len(samples)))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#111111')

    for i, ((cls, img_list), color) in enumerate(zip(samples.items(), COLORS)):
        print(f'  [{i+1}/{len(samples)}] Radial spectrum: {cls} ...', flush=True)
        radials = [radial_average(compute_fft_spectrum(img)) for img, _ in img_list]

        min_len = min(len(r) for r in radials)
        mean_radial = np.mean([r[:min_len] for r in radials], axis=0)
        freqs = np.arange(min_len)
        ax.plot(freqs, mean_radial, label=cls, color=color, linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Spatial Frequency (cycles/pixel)', color='white', fontsize=10)
    ax.set_ylabel('Log Magnitude (mean)', color='white', fontsize=10)
    ax.set_title('Radial Average Power Spectrum by Generator',
                 color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444444')
    ax.legend(fontsize=7, loc='upper right',
              facecolor='#1a1a1a', labelcolor='white', framealpha=0.8)
    ax.grid(True, color='#333333', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[✓] Radial spectrum saved → {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Grad-CAM + Fourier analysis for DRIFT')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--drift_checkpoint', type=str,
                       help='DRIFT encoder checkpoint (.pth)')
    group.add_argument('--baseline_checkpoint', type=str,
                       help='BaselineResNet checkpoint (.pth)')
    parser.add_argument('--probe_checkpoint', type=str, default=None,
                       help='LinearProbe checkpoint (required for DRIFT)')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Root of Closed_Set (contains one folder per generator)')
    parser.add_argument('--out_dir', type=str,
                       default='./figures/interpretability')
    parser.add_argument('--n_samples', type=int, default=3,
                       help='Number of images per generator to visualise')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── load model ────────────────────────────────────────────────────────────
    if args.baseline_checkpoint:
        print('Loading BaselineResNet …')
        ckpt = torch.load(args.baseline_checkpoint, map_location=device)
        model = BaselineResNet(num_classes=NUM_CLASSES)
        model.load_state_dict(ckpt['model_state'])
        model.to(device).eval()
        gradcam = GradCAM(model, is_baseline=True)
        tag = 'baseline_resnet'

    elif args.drift_checkpoint:
        if not args.probe_checkpoint:
            parser.error('--probe_checkpoint is required when using --drift_checkpoint')
        print('Loading DRIFT + LinearProbe …')
        drift = DRIFT().to(device)
        drift_ckpt = torch.load(args.drift_checkpoint, map_location=device)
        drift.load_state_dict(drift_ckpt['model_state'])

        model = LinearProbe(drift.backbone, NUM_CLASSES).to(device)
        probe_ckpt = torch.load(args.probe_checkpoint, map_location=device)
        model.classifier.load_state_dict(probe_ckpt['classifier_state'])
        model.eval()
        gradcam = GradCAM(model, is_baseline=False)
        tag = 'drift'

    else:
        parser.error('Provide either --drift_checkpoint or --baseline_checkpoint')

    # ── load images ───────────────────────────────────────────────────────────
    print(f'Loading {args.n_samples} samples per generator from {args.image_dir} …')
    samples = load_samples(Path(args.image_dir), args.n_samples)
    if not samples:
        print('ERROR: No images loaded. Check --image_dir path.')
        return
    print(f'Loaded generators: {list(samples.keys())}')

    # wrap gradcam so it moves tensor to device automatically
    def gradcam_fn(img_tensor):
        return gradcam(img_tensor.to(device))

    # ── generate plots ────────────────────────────────────────────────────────
    print(f'\n[1/3] Generating Grad-CAM heatmaps …')
    plot_gradcam_grid(samples, gradcam_fn,
                     out_dir / f'gradcam_{tag}.png')

    print(f'\n[2/3] Generating Fourier spectrum grid …')
    plot_fourier_grid(samples,
                     out_dir / f'fourier_spectrum_{tag}.png')

    print(f'\n[3/3] Generating radial power spectrum …')
    plot_radial_spectrum(samples,
                        out_dir / f'radial_spectrum_{tag}.png')

    gradcam.remove_hooks()
    print(f'\nAll figures saved to: {out_dir}')


if __name__ == '__main__':
    main()
