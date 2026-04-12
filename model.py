import torch
from torch import nn
from torchvision import models

# creating our DRIFT model class, we subclass torch's nn.module to create a custom class
# and we need a __init__ and a forward methods.

class DRIFT(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        # we want to remove the last fully connected layer of resnet
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat_dim = resnet.fc.in_features
        resnet.fc = nn.Identity() # this replaces the last classification layer with a identity layer
        self.backbone = resnet

        # in supcon, they use a small MLP to map the backbone embeddings to the contrastive space
        # they used 128 as the size of their contrastive embeddings, so we use 128 as well for now.

        self.projection_head = nn.sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, embed_dim)
        )

    def forward(self,x):
        # takes in x, which is a batch of samples from the dataset
        # x shape: (batch_size, 3, H, W)
        features = self.backbone(x) # dimension is (batch size, 2048)
        projections = self.projection_head(features) # (batch size, 128)
        projections = nn.functional.normalize(projections, dim=1)
        return features, projections





