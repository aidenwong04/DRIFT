from pathlib import Path
import torch
from torch.utils.data import random_split
from dataset import WILDDataset

SEED = 42
ROOT = Path('/projectnb/cs585/projects/ASUFratLeader/data/Data/Closed_Set')
SPLIT_DIR = Path('/projectnb/cs585/projects/ASUFratLeader/DRIFT/splits')

def main():
    """ this creates the splits for the data and saves it to disk, so we can ensure that 
        we don't evaluate on images that we trained on. Once we have the splits saved to disk
        best not to run again. (It is okay as long as we don't add more images to the database.)
    """
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = WILDDataset(ROOT)
    n = len(dataset)

    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    torch.save(train_set.indices, SPLIT_DIR/'train_idx.pt')
    torch.save(val_set.indices, SPLIT_DIR/'val_idx.pt')
    torch.save(test_set.indices, SPLIT_DIR/'test_idx.pt')

    print(f'Total samples: {n}')
    print(f'Train: {len(train_set.indices)} -> {SPLIT_DIR}/train_idx.pt')
    print(f'Val: {len(val_set.indices)} -> {SPLIT_DIR}/val_idx.pt')
    print(f'Test: {len(test_set.indices)} -> {SPLIT_DIR}/test_idx.pt')

if __name__ == '__main__':
    main()