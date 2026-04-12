from model import DRIFT
from losses import SupConLoss
from dataset import WILDDataset

from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # instanstiate the dataset, model, and loss.
    root = Path('/projectnb/cs585/projects/ASUFratLeader/data/Data/Closed_Set')
    wild_dataset = WILDDataset(root)
    data_loader = DataLoader(wild_dataset, batch_size=128, shuffle=True, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drift = DRIFT().to(device)

    supconloss = SupConLoss() #default temp = 0.1

    # pass in a batch of data from the dataloader to the model

    epochs = 50

    optimizer = torch.optim.Adam(drift.parameters(), lr=0.001)

    best_val_loss = float('inf')
    
    # training loop
    for epoch in range(epochs):
        # trainng model
        drift.train()
        print('Epoch: ' + str(epoch))
        for batch_idx, (view1, view2, model_idx) in enumerate(train_loader):
            concatenated_views = torch.cat((view1, view2), dim=0).to(device)
            labels = torch.cat((model_idx, model_idx), dim=0).to(device)

            features, projections = drift.forward(concatenated_views)
            loss = supconloss.forward(projections, labels)

            print('Batch_idx: ' + str(batch_idx) + ', Loss: ' + str(loss.item()))

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
        print('Epoch ' + str(epoch) + ' Val Loss: ' + str(avg_val_loss))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(drift.state_dict(), 'checkpoints/best_model.pth')
            print('Saved best model at epoch ' + str(epoch))
    
    torch.save(drift.state_dict(), 'checkpoints/drift_model.pth')

        



    






