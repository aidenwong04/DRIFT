import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.io import decode_image

import transformations

# creating a custom dataset from the pytorhc's Dataset class, we need _init_, _len_, _getitem_.
# we load an image using pillow (python image loader), and then we return a pytorch tensor object in _getitem_.

class WILDDataset(Dataset):
    def __init__(self, root, mode="train"):
        """
        mode: "train" -> two degraded views (for SupCon training)
              "clean" -> one clean view (for clean eval)
              "degraded" -> one degraded view (for robustness eval)
        """
        #ROOT = Path('/projectnb/cs585/projects/ASUFratLeader/data/Data/Closed_Set')
        assert mode in ("train", "clean", "degraded")
        self.mode = mode
        self.root = root
        self.models = sorted([sub_dir.name for sub_dir in self.root.iterdir() if sub_dir.is_dir()]) # list of all image generators
        self.models_to_idx = {name: idx for (idx, name) in enumerate(self.models)} # dictionary class to idx
        self.samples = []
        self.transform = transformations.DegradationPipeline()

        self.clean_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]) # just skips the degradation

        for sub_dir in sorted(root.iterdir()):
            if sub_dir.is_dir(): # these are all the model names
                for img in sorted(sub_dir.iterdir()): # iterate through all images in subdir
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

        if self.mode == "train":
            return self.transform(img), self.transform(img), model_idx
        elif self.mode == "clean":
            return self.clean_transform(img), model_idx
        else:  # degraded
            return self.transform(img), model_idx
            
if __name__ == "__main__":
    root = Path("/projectnb/cs585/projects/ASUFratLeader/data/Data/Closed_Set")
    dataset = WILDDataset(root)
    print(f"Classes: {dataset.models}")
    print(f"Total samples: {len(dataset)}")
    
    view1, view2, label = dataset[0]
    print(f"view1 shape: {view1.shape}, Label: {label}")
    print(f"view2 shape: {view2.shape}, Label: {label}")


