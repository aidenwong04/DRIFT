"""
eval_baseline.py — evaluate a trained baseline (ResNet-50 or CLIP+linear)
on the held-out test set.

Mirrors the interface of eval.py so both scripts produce comparable outputs
in the same eval_results/ folder structure.

Run clean and degraded to get the two headline numbers:

    python eval_baseline.py \
        --checkpoint .../checkpoints/baseline_resnet_best_XXXXXXXX.pth \
        --mode clean

    python eval_baseline.py \
        --checkpoint .../checkpoints/baseline_resnet_best_XXXXXXXX.pth \
        --mode degraded

Outputs (in eval_results/<run_name>_<mode>/):
    confusion_matrix.npy         — raw 10x10 confusion matrix
    confusion_matrix.png         — row-normalized heatmap
    classification_report.txt    — accuracy + per-class precision/recall/F1
    preds.npy, labels.npy        — raw arrays for error analysis
"""

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from dataset import WILDDataset
from baseline_model import BaselineResNet, BaselineCLIP

ROOT        = Path('/projectnb/cs585/projects/ASUFratLeader/data/Data/Closed_Set')
SPLIT_DIR   = Path('/projectnb/cs585/projects/ASUFratLeader/DRIFT/splits')
RESULTS_DIR = Path('/projectnb/cs585/projects/ASUFratLeader/DRIFT/eval_results')
NUM_CLASSES = 10


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  type=str, required=True,
                        help='path to the baseline checkpoint')
    parser.add_argument('--mode',        type=str, choices=['clean', 'degraded'],
                        required=True)
    parser.add_argument('--run_name',    type=str, default=None)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--batch_size',  type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    run_name = args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir  = RESULTS_DIR / f'{run_name}_{args.mode}'
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device:   {device}')
    print(f'Mode:     {args.mode}')
    print(f'Saving →  {out_dir}')

    # ------------------------------------------------------------------
    # Load checkpoint and infer arch
    # ------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device)
    arch = ckpt.get('arch', 'resnet')          # default to resnet for old ckpts
    print(f'Arch:     {arch}')

    if arch == 'resnet':
        model = BaselineResNet(num_classes=NUM_CLASSES)
    elif arch == 'clip':
        model = BaselineCLIP(num_classes=NUM_CLASSES, device=str(device))
    else:
        raise ValueError(f'Unknown arch in checkpoint: {arch!r}')

    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    # ------------------------------------------------------------------
    # Dataset  (mode controls whether post-processing is applied)
    # ------------------------------------------------------------------
    full_dataset = WILDDataset(ROOT, mode=args.mode)
    test_idx     = torch.load(SPLIT_DIR / 'test_idx.pt')
    test_dataset = Subset(full_dataset, test_idx)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    class_names = full_dataset.models
    print(f'Test samples: {len(test_dataset)}')
    print(f'Classes ({len(class_names)}): {class_names}')

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds  = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels)
            if batch_idx % 10 == 0:
                print(f'  batch {batch_idx}/{len(test_loader)}')

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    acc = accuracy_score(all_labels, all_preds)
    cm  = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print(f'\n=== BASELINE ({arch.upper()}) — {args.mode.upper()} RESULTS ===')
    print(f'Overall accuracy: {acc:.4f}')
    print('\nPer-class accuracy:')
    for name, a in zip(class_names, per_class_acc):
        print(f'  {name:30s} {a:.4f}')
    print('\nClassification report:')
    print(report)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    np.save(out_dir / 'confusion_matrix.npy', cm)
    np.save(out_dir / 'preds.npy',            all_preds)
    np.save(out_dir / 'labels.npy',           all_labels)

    with open(out_dir / 'classification_report.txt', 'w') as f:
        f.write(f'Arch:       {arch}\n')
        f.write(f'Mode:       {args.mode}\n')
        f.write(f'Checkpoint: {args.checkpoint}\n')
        f.write(f'Seed:       {args.seed}\n')
        f.write(f'Test samples: {len(test_dataset)}\n')
        f.write(f'Overall accuracy: {acc:.4f}\n\n')
        f.write('Per-class accuracy:\n')
        for name, a in zip(class_names, per_class_acc):
            f.write(f'  {name:30s} {a:.4f}\n')
        f.write('\n')
        f.write(report)

    # row-normalized confusion matrix heatmap
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True, fmt='.2f',
        cmap='Oranges',                         # different color from DRIFT eval
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Fraction of true class'},
        ax=ax,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Baseline ({arch}) confusion matrix — {args.mode} (acc={acc:.4f})')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png', dpi=200)
    plt.close(fig)

    print(f'\nDone. Results in {out_dir}')


if __name__ == '__main__':
    main()