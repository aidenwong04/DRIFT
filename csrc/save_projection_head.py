import torch
import os
from model import DRIFT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the full checkpoint
checkpoint = torch.load('/projectnb/cs585/projects/ASUFratLeader/DRIFT_NEW/checkpoints/best_model_20260507_161029.pth', map_location=device)

# reconstruct the model and load weights
drift = DRIFT().to(device)
drift.load_state_dict(checkpoint['model_state'])

# save only what you trained
lean = {
    'projection_head_state': drift.projection_head.state_dict(),
    'epoch': checkpoint['epoch'],
    'best_val_loss': checkpoint['best_val_loss'],
    'backbone_name': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'embed_dim': 128,
}
torch.save(lean, '/projectnb/cs585/projects/ASUFratLeader/DRIFT_NEW/checkpoints/drift_projection_head_only.pth')
print(f"Saved. Size: {os.path.getsize('drift_projection_head_only.pth') / 1e6:.1f} MB")