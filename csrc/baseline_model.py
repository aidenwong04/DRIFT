"""
baseline_model.py — Two baseline architectures for diffusion model attribution.

BaselineResNet: fine-tuned ResNet50 with a classification head (end-to-end).
BaselineCLIP:   frozen CLIP ViT-B/32 features + a trained linear classifier.

Both expose a forward(x) -> logits interface so they drop into the same
train/eval loop without modification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# 1.  ResNet-50 end-to-end fine-tune baseline
# ---------------------------------------------------------------------------

class BaselineResNet(nn.Module):
    """
    Standard supervised ResNet-50 fine-tuned with cross-entropy.

    Unlike DRIFT's LinearProbe, the whole network is trained end-to-end —
    the backbone is NOT frozen.  This is the strongest "vanilla" baseline
    and the main number to beat.
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        feat_dim = resnet.fc.in_features          # 2048 for ResNet-50
        resnet.fc = nn.Linear(feat_dim, num_classes)
        self.model = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# 2.  CLIP ViT-B/32  frozen features + linear head baseline
# ---------------------------------------------------------------------------

class BaselineCLIP(nn.Module):
    """
    Frozen CLIP ViT-B/32 visual encoder with a trained linear classifier on top.

    CLIP features are surprisingly robust to post-processing (Cozzolino et al.,
    2024), making this a strong and fast-to-train comparison point.

    Requires: pip install openai-clip
              (or  pip install git+https://github.com/openai/CLIP.git)
    """

    def __init__(self, num_classes: int = 10, device: str = "cpu"):
        super().__init__()
        try:
            import clip  # openai-clip
        except ImportError:
            raise ImportError(
                "openai-clip is not installed.\n"
                "Run:  pip install git+https://github.com/openai/CLIP.git"
            )

        clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.encoder = clip_model.visual          # just the vision tower
        self.encoder.float()                      # clip loads in fp16 by default

        # freeze CLIP — we only train the linear head
        for param in self.encoder.parameters():
            param.requires_grad = False

        clip_feat_dim = 512                       # ViT-B/32 output dimension
        self.classifier = nn.Linear(clip_feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            features = self.encoder(x).float()
        return self.classifier(features)