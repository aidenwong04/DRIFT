"""
eval.py — evaluate a trained DRIFT encoder + LinearProbe on the held-out test set.

Run once with --mode clean and once with --mode degraded to get the two
headline numbers for the paper:

    python eval.py \
        --drift_checkpoint .../checkpoints/best_model_XXX.pth \
        --probe_checkpoint .../checkpoints/probe_best_XXX.pth \
        --mode clean
    python eval.py \
        --drift_checkpoint .../checkpoints/best_model_XXX.pth \
        --probe_checkpoint .../checkpoints/probe_best_XXX.pth \
        --mode degraded

Outputs (per run, in eval_results/<run_name>_<mode>/):
    confusion_matrix.npy        — raw confusion matrix (10x10)
    confusion_matrix.png        — row-normalized heatmap for the paper
    classification_report.txt   — accuracy + per-class precision/recall/F1
    preds.npy, labels.npy       — raw predictions (for later error analysis)

Dependencies: sklearn, seaborn, matplotlib (pip install --user if missing).
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
from model import DRIFT, LinearProbe


ROOT = Path('/projectnb/cs585/projects/ASUFratLeader/data/Data/Closed_Set')
SPLIT_DIR = Path('/projectnb/cs585/projects/ASUFratLeader/DRIFT/splits')
RESULTS_DIR = Path('/projectnb/cs585/projects/ASUFratLeader/DRIFT/eval_results')
NUM_CLASSES = 10


def set_seed(seed):
    """seed all RNGs so the degraded eval is reproducible across runs"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drift_checkpoint', type=str, required=True,
                        help='path to the trained DRIFT checkpoint (has model_state)')
    parser.add_argument('--probe_checkpoint', type=str, required=True,
                        help='path to the trained linear probe checkpoint (has classifier_state)')
    parser.add_argument('--mode', type=str, choices=['clean', 'degraded'], required=True,
                        help='clean = no post-processing applied to test images; '
                             'degraded = random post-processing applied per-image')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='keep at 0 for reproducible degraded eval '
                             '(multi-worker RNG state is non-deterministic by default)')
    args = parser.parse_args()

    set_seed(args.seed)
    run_name = args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = RESULTS_DIR / f'{run_name}_{args.mode}'
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Mode: {args.mode}')
    print(f'Seed: {args.seed}')
    print(f'Saving to: {out_dir}')

    # initialize the dataset
    full_dataset = WILDDataset(ROOT, mode=args.mode)
    test_idx = torch.load(SPLIT_DIR / 'test_idx.pt')
    test_dataset = Subset(full_dataset, test_idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    class_names = full_dataset.models
    print(f'Test samples: {len(test_dataset)}')
    print(f'Classes ({len(class_names)}): {class_names}')

    # load the full DRIFT model first, 
    drift = DRIFT().to(device)
    drift_ckpt = torch.load(args.drift_checkpoint, map_location=device)
    drift.load_state_dict(drift_ckpt['model_state'])

    # LinearProbe wraps drift.backbone (same module, shared weights) + a classifier head
    probe = LinearProbe(drift.backbone, NUM_CLASSES).to(device)
    probe_ckpt = torch.load(args.probe_checkpoint, map_location=device)
    probe.classifier.load_state_dict(probe_ckpt['classifier_state'])
    probe.eval()

    # ---- inference ----
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            logits = probe(images)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels)
            if batch_idx % 10 == 0:
                print(f'  batch {batch_idx}/{len(test_loader)}')

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # ---- metrics ----
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    # clip avoids divide-by-zero if some class happens to have 0 test samples
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print(f'\n=== {args.mode.upper()} RESULTS ===')
    print(f'Overall accuracy: {acc:.4f}')
    print('\nPer-class accuracy:')
    for name, a in zip(class_names, per_class_acc):
        print(f'  {name:30s} {a:.4f}')
    print('\nClassification report:')
    print(report)

    # ---- save outputs ----
    np.save(out_dir / 'confusion_matrix.npy', cm)
    np.save(out_dir / 'preds.npy', all_preds)
    np.save(out_dir / 'labels.npy', all_labels)

    with open(out_dir / 'classification_report.txt', 'w') as f:
        f.write(f'Mode: {args.mode}\n')
        f.write(f'DRIFT checkpoint: {args.drift_checkpoint}\n')
        f.write(f'Probe checkpoint: {args.probe_checkpoint}\n')
        f.write(f'Seed: {args.seed}\n')
        f.write(f'Test samples: {len(test_dataset)}\n')
        f.write(f'Overall accuracy: {acc:.4f}\n\n')
        f.write('Per-class accuracy:\n')
        for name, a in zip(class_names, per_class_acc):
            f.write(f'  {name:30s} {a:.4f}\n')
        f.write('\n')
        f.write(report)

    # row-normalized confusion matrix (each row sums to 1) — easier to read
    # for the paper than raw counts since classes may be slightly imbalanced
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True, fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Fraction of true class'},
        ax=ax,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'DRIFT confusion matrix — {args.mode} (acc={acc:.4f})')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png', dpi=200)
    plt.close(fig)

    print(f'\nDone. Results in {out_dir}')


if __name__ == '__main__':
    main()