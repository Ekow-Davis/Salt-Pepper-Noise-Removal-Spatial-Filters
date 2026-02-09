import torch
from torch.utils.data import Dataset
import cv2
import os

class NoisyImageDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.files = sorted(os.listdir(clean_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        clean = cv2.imread(os.path.join(self.clean_dir, fname))
        noisy = cv2.imread(os.path.join(self.noisy_dir, fname))

        clean = torch.tensor(clean).permute(2,0,1).float() / 255.0
        noisy = torch.tensor(noisy).permute(2,0,1).float() / 255.0

        return noisy, clean
