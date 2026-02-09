import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import Dataset
import cv2
from utils.add_noise import add_salt_pepper_noise


class NoisyImageDataset(Dataset):
    def __init__(self, clean_dir, noise_level=0.3, img_size=128):
        self.clean_dir = clean_dir
        self.files = sorted(os.listdir(clean_dir))
        self.noise_level = noise_level
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        clean_path = os.path.join(self.clean_dir, fname)

        clean = cv2.imread(clean_path)
        if clean is None:
            raise FileNotFoundError(f"Cannot read image: {clean_path}")

        # Resize to fixed size (IMPORTANT)
        clean = cv2.resize(clean, (self.img_size, self.img_size))

        # Add salt-and-pepper noise
        noisy = add_salt_pepper_noise(clean, self.noise_level)

        # Convert to tensor
        clean = torch.tensor(clean).permute(2, 0, 1).float() / 255.0
        noisy = torch.tensor(noisy).permute(2, 0, 1).float() / 255.0

        return noisy, clean
