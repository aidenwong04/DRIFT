"""
train_baseline.py — train a supervised baseline (ResNet-50 or CLIP+linear)
for diffusion model attribution on the WILD closed set.

Usage:
    # ResNet-50 end-to-end fine-tune (main baseline):
    python train_baseline.py --arch resnet

    # CLIP frozen-encoder + linear head:
    python train_baseline.py --arch clip

    # Resume a previous run:
    python train_baseline.py --arch resnet --resume checkpoints/baseline_resnet_best_XXXXXXXX.pth

Outputs (saved to checkpoints/):
    baseline_<arch>_best_<run_name>.pth   — best checkpoint (lowest val loss)
    baseline_<arch>_latest_<run_name>.pth — most recent checkpoint

Each checkpoint contains:
    epoch, model_state, optimizer_state, val_loss, val_accuracy, arch
"""

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import wandb

from dataset import WILDDataset
from baseline_model import BaselineResNet, BaselineCLIP

# ---------------------------------------------------------------------------
# Paths  (mirror the conventions in train.py / probe.py)
# ---------------------------------------------------------------------------
ROOT       = Path('/projectnb/cs585/projects/ASUFratLeader/data/Data/Closed_Set')
SPLIT_DIR  = Path('/projectnb/cs585/projects/ASUFratLeader/DRIFT/splits')
CKPT_DIR   = Path('/projectnb/cs585/projects/ASUFratLeader/DRIFT/checkpoints')
NUM_CLASSES = 10


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(arch: str, device: torch.device) -> nn.Module:
    if arch == 'resnet':
        return BaselineResNet(num_classes=NUM_CLASSES).to(device)
    elif arch == 'clip':
        return BaselineCLIP(num_classes=NUM_CLASSES, device=str(device)).to(device)
    else:
        raise ValueError(f"Unknown arch: {arch!r}. Choose 'resnet' or 'clip'.")


def trainable_params(model: nn.Module, arch: str):
    """Return only the parameters that need gradients.

    For CLIP, only the linear classifier head is trained; for ResNet the whole
    network is fine-tuned.
    """
    if arch == 'clip':
        return model.classifier.parameters()
    return model.parameters()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',       type=str, default='resnet',
                        choices=['resnet', 'clip'],
                        help='which baseline architecture to train')
    parser.add_argument('--resume',     type=str, default=None,
                        help='path to a checkpoint to resume from')
    parser.add_argument('--run_name',   type=str, default=None)
    parser.add_argument('--epochs',     type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr',         type=float, default=1e-4,
                        help='learning rate (1e-4 works well for ResNet fine-tune; '
                             'use 1e-3 for CLIP linear probe)')
    parser.add_argument('--patience',   type=int, default=5,
                        help='early-stopping patience in epochs')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    run_name = args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Arch:   {args.arch}')
    print(f'Device: {device}')
    print(f'Run:    {run_name}')

    # ------------------------------------------------------------------
    # Dataset  — use clean mode for training the baseline
    # The dataset already returns (view1, view2, model_idx); we use view1
    # as the single training image (consistent with probe.py).
    # ------------------------------------------------------------------
    full_dataset = WILDDataset(ROOT, mode='clean')
    train_idx = torch.load(SPLIT_DIR / 'train_idx.pt')
    val_idx   = torch.load(SPLIT_DIR / 'val_idx.pt')

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=4,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=4,
    )
    print(f'Train: {len(train_idx)} | Val: {len(val_idx)}')

    # ------------------------------------------------------------------
    # Model, optimizer, loss
    # ------------------------------------------------------------------
    model     = build_model(args.arch, device)
    optimizer = torch.optim.Adam(trainable_params(model, args.arch), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Optional resume
    # ------------------------------------------------------------------
    start_epoch       = 0
    best_val_loss     = float('inf')
    epochs_no_improve = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        best_val_loss = ckpt['val_loss']
        start_epoch   = ckpt['epoch'] + 1
        print(f'Resumed from {args.resume}, starting at epoch {start_epoch}')

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    wandb.init(
        project='DRIFT',
        name=f'baseline_{args.arch}_{run_name}',
        config=vars(args),
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        # ---- train ----
        model.train()
        print(f'\nEpoch {epoch}')
        for batch_idx, (images,labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'train_loss': loss.item(), 'epoch': epoch, 'batch': batch_idx})

        # ---- validate ----
        model.eval()
        val_loss, correct, total, num_batches = 0.0, 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits    = model(images)
                loss      = criterion(logits, labels)
                val_loss += loss.item()
                num_batches += 1

                preds    = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        avg_val_loss = val_loss / num_batches
        val_acc      = correct / total
        wandb.log({'val_loss': avg_val_loss, 'val_accuracy': val_acc, 'epoch': epoch})
        print(f'  Val loss: {avg_val_loss:.4f}  Val acc: {val_acc:.4f}')

        # ---- checkpointing ----
        checkpoint = {
            'epoch':           epoch,
            'arch':            args.arch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_loss':        avg_val_loss,
            'val_accuracy':    val_acc,
        }
        latest_path = CKPT_DIR / f'baseline_{args.arch}_latest_{run_name}.pth'
        torch.save(checkpoint, latest_path)

        if avg_val_loss < best_val_loss:
            best_val_loss     = avg_val_loss
            epochs_no_improve = 0
            best_path = CKPT_DIR / f'baseline_{args.arch}_best_{run_name}.pth'
            torch.save(checkpoint, best_path)
            print(f'  ✓ New best saved → {best_path}')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f'  Early stopping (no improvement for {args.patience} epochs)')
                break

    print('\nTraining complete.')


if __name__ == '__main__':
    main()