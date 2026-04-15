from model import DRIFT
from model import LinearProbe
from losses import SupConLoss
from dataset import WILDDataset
from datetime import datetime

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import argparse
import wandb

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='checkpoint of the model you want to probe')
    args = parser.parse_args()

    if args.model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        drift = DRIFT().to(device)

        # load the model
        model = torch.load(args.model)
        drift.load_state_dict(model['model_state'])

        linear_probe = LinearProbe(drift.backbone, 10) # init the linear probe model

        root = Path('/projectnb/cs585/projects/ASUFratLeader/data/Data/Closed_Set')
        wild_dataset = WILDDataset(root)
        train_size = int(0.8 * len(wild_dataset))
        val_size = len(wild_dataset) - train_size
        train_dataset, val_dataset = random_split(wild_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True)

        epochs = 10

        optimizer = torch.optim.Adam(linear_probe.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        wandb.init(project='DRIFT', name='linear_probe_run', config={
            'epochs': epochs,
            'batch_size': 128,
            'lr': 0.001,
        })

        best_val_loss = float('inf')

        for epoch in range(epochs):
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
        else:
            print("Model checkpoint missing.")