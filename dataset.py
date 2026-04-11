import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.io import decode_image
import os

import transformations

# creating a custom dataset from the pytorhc's Dataset class, we need _init_, _len_, _getitem_.
# we load an image using pillow (python image loader), and then we return a pytorch tensor object in _getitem_.

class WILDDataset(Dataset):
    def __init__(self, root):
        #ROOT = Path('/projectnb/cs585/projects/ASUFratLeader/data/Data/Closed_Set')
        self.root = root
        self.models = sorted([sub_dir.name for sub_dir in self.root.iterdir() if sub_dir.is_dir()]) # list of all image generators
        self.models_to_idx = {name: idx for (idx, name) in enumerate(self.models)} # dictionary class to idx
        self.samples = []
        self.transform = transformations.DegradationPipeline()

        for sub_dir in root.iterdir():
            if sub_dir.is_dir(): # these are all the model names
                for img in sub_dir.iterdir(): # iterate through all images in subdir
                    # add all images to the samples as a tuple (path to image, model_name)
                    # img is a path
                    if img.suffix == '.png':
                        self.samples.append((img, self.models_to_idx[sub_dir.name]))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx): 
        # returns two differently degraded copies of the same image 
        path, model_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2, model_idx

if __name__ == "__main__":
    root = Path("/projectnb/cs585/projects/ASUFratLeader/data/Data/Closed_Set")
    dataset = WILDDataset(root)
    print(f"Classes: {dataset.models}")
    print(f"Total samples: {len(dataset)}")
    
    view1, view2, label = dataset[0]
    print(f"view1 shape: {view1.shape}, Label: {label}")
    print(f"view2 shape: {view2.shape}, Label: {label}")


