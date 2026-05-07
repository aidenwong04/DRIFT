import torch
from torch import nn
from torchvision import models

from transformers import AutoModel, AutoImageProcessor

# creating our DRIFT model class, we subclass torch's nn.module to create a custom class
# and we need a __init__ and a forward methods.

class DRIFT(nn.Module):
    def __init__(self, embed_dim=128, backbone_name='facebook/dinov3-vitb16-pretrain-lvd1689m'):
        super().__init__()
        # we want to remove the last fully connected layer of resnet
        self.backbone = AutoModel.from_pretrained(backbone_name)
        feat_dim = self.backbone.config.hidden_size

        for param in self.backbone.parameters():
            param.requires_grad = False

        # in supcon, they use a small MLP to map the backbone embeddings to the contrastive space
        # they used 128 as the size of their contrastive embeddings, so we use 128 as well for now.

        self.projection_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, embed_dim)
        )

    def forward(self,x):
        # takes in x, which is a batch of samples from the dataset
        # x shape: (batch_size, 3, H, W)
        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state[:, 0] # CLS token, shape (B, 768)
        projections = self.projection_head(features) # (batch size, 128)
        projections = nn.functional.normalize(projections, dim=1)
        return features, projections


class LinearProbe(nn.Module):
    def __init__(self, backbone, num_classes, feat_dim):
        # takes the backbone, freezes the weights, and attatches a linear classifier on top of it
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False # this freezes the backbone

        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self,x):
        with torch.no_grad():
            outputs = self.backbone(pixel_values=x)
            features = outputs.last_hidden_state[:, 0]
        return self.classifier(features)
        



