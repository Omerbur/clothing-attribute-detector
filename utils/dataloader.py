import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ClothingDataset(Dataset):
    def __init__(self, csv_path, img_dir, attr_list, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.attrs = attr_list
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        label = [int(attr in row['attributes']) for attr in self.attrs]
        return self.transform(image), torch.tensor(label, dtype=torch.float)
