from model import DRIFT
from losses import SupConLoss
from dataset import WILDDataset
from datetime import datetime

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import argparse
import wandb

if __name__ == "__main__":
    # check if u want to resume from a previous training run
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    args = parser.parse_args()

    # instanstiate the dataset, model, and loss.
    root = Path('/projectnb/cs585/projects/ASUFratLeader/data/Data/Closed_Set')
    full_dataset = WILDDataset(root)

    train_idx = torch.load('/projectnb/cs585/projects/ASUFratLeader/DRIFT/splits/train_idx.pt')
    val_idx = torch.load('/projectnb/cs585/projects/ASUFratLeader/DRIFT/splits/val_idx.pt')

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drift = DRIFT().to(device)

    supconloss = SupConLoss() #default temp = 0.1

    # pass in a batch of data from the dataloader to the model

    epochs = 50

    optimizer = torch.optim.Adam(drift.parameters(), lr=0.001)

    wandb.init(project='DRIFT', config={
        'epochs': epochs,
        'batch_size': 128,
        'lr': 0.001,
        'temperature': 0.1,
    })

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint = torch.load(args.resume)
        drift.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch'] + 1
        print('Resumed from checkpoint: ' + args.resume)
        print(f'Resumed from epoch {start_epoch}')

    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # training loop
    for epoch in range(start_epoch, epochs):
        # trainng model
        drift.train()
        print('Epoch: ' + str(epoch))
        for batch_idx, (view1, view2, model_idx) in enumerate(train_loader):
            concatenated_views = torch.cat((view1, view2), dim=0).to(device)
            labels = torch.cat((model_idx, model_idx), dim=0).to(device)

            features, projections = drift.forward(concatenated_views)
            loss = supconloss.forward(projections, labels)

            wandb.log({'train_loss': loss.item(), 'epoch': epoch, 'batch': batch_idx})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        drift.eval()
        val_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for view1, view2, model_idx in val_loader:
                concatenated_views = torch.cat((view1, view2), dim=0).to(device)
                labels = torch.cat((model_idx, model_idx), dim=0).to(device)

                features, projections = drift.forward(concatenated_views)
                loss = supconloss.forward(projections, labels)

                val_loss += loss.item()
                num_batches += 1

        avg_val_loss = val_loss / num_batches
        wandb.log({'val_loss': avg_val_loss, 'epoch': epoch})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state': drift.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, f'/projectnb/cs585/projects/ASUFratLeader/DRIFT/checkpoints/best_model_{run_name}.pth')
            
            print('Saved best model at epoch ' + f'/projectnb/cs585/projects/ASUFratLeader/DRIFT/checkpoints/best_model_{run_name}.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state': drift.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, f'/projectnb/cs585/projects/ASUFratLeader/DRIFT/checkpoints/drift_model_{run_name}.pth')

        



    






