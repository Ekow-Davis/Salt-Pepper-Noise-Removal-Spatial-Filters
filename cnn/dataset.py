import torch
from torch.utils.data import Dataset
import cv2
import os

class NoisyImageDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir):
        self.clean_files = sorted(os.listdir(clean_dir))
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean = cv2.imread(os.path.join(self.clean_dir, self.clean_files[idx]))
        noisy = cv2.imread(os.path.join(self.noisy_dir, self.clean_files[idx]))

        clean = torch.tensor(clean).permute(2,0,1).float() / 255
        noisy = torch.tensor(noisy).permute(2,0,1).float() / 255

        return noisy, clean
