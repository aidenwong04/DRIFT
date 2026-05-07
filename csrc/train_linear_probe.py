from model import DRIFT
from model import LinearProbe
from losses import SupConLoss
from dataset import WILDDataset
from datetime import datetime

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import argparse
import wandb

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='checkpoint of the model you want to probe')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--run_name', type=str, default=None)
    args = parser.parse_args()

    run_name = args.run_name if args.run_name else datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        drift = DRIFT().to(device)

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Device: {device}")

        # load the model
        model = torch.load(args.model, map_location=device)
        drift.load_state_dict(model['model_state'])

        linear_probe = LinearProbe(drift.backbone, 10, feat_dim=drift.backbone.config.hidden_size).to(device)  # init the linear probe model

        root = Path('/projectnb/cs585/projects/ASUFratLeader/data/Data/Closed_Set')
        full_dataset = WILDDataset(root)

        train_idx = torch.load('/projectnb/cs585/projects/ASUFratLeader/DRIFT/splits/train_idx.pt')
        val_idx = torch.load('/projectnb/cs585/projects/ASUFratLeader/DRIFT/splits/val_idx.pt')

        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True)

        optimizer = torch.optim.Adam(linear_probe.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        epochs = 25
        patience = 5
        epochs_since_improvement = 0
        best_val_loss = float('inf')
        start_epoch = 0

        wandb.init(project='DRIFT', name='linear_probe_degraded', config={
            'epochs': epochs,
            'batch_size': 128,
            'lr': 0.001,
            'seed': 42,
            'drift_checkpoint': args.model,
        })

        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            linear_probe.classifier.load_state_dict(checkpoint['classifier_state'])
            best_val_loss = checkpoint['val_loss']
            start_epoch = checkpoint['epoch'] + 1
            print(f'Resumed probe from {args.resume}, starting at epoch {start_epoch}')

        for epoch in range(start_epoch, epochs):
            # trainng model
            linear_probe.train()
            print('Epoch: ' + str(epoch))
            for batch_idx, (view1, view2, model_idx) in enumerate(train_loader):
                images = view1.to(device)
                labels = model_idx.to(device)
                logits = linear_probe.forward(images)
                loss = criterion.forward(logits, labels)

                wandb.log({'train_loss': loss.item(), 'epoch': epoch, 'batch': batch_idx})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # validation
            linear_probe.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            num_batches = 0
            with torch.no_grad():
                for view1, view2, model_idx in val_loader:
                    images = view1.to(device)
                    labels = model_idx.to(device)
                    logits = linear_probe(images)
                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    num_batches += 1

                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = val_loss / num_batches
            accuracy = correct / total
            wandb.log({'val_loss': avg_val_loss, 'val_accuracy': accuracy, 'epoch': epoch})
            print(f'Epoch {epoch} Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_since_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'classifier_state': linear_probe.classifier.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_accuracy': accuracy,
                    'drift_checkpoint': args.model,  # so we remember which backbone this goes with
                }, f'/projectnb/cs585/projects/ASUFratLeader/DRIFT/checkpoints/probe_best_{run_name}.pth')
                print(f'Saved best probe at epoch {epoch}')
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
    
    else:
        print("Model checkpoint missing.")