import numpy as np
import cv2
import os

def add_salt_pepper_noise(image, amount=0.1):
    noisy = image.copy()
    h, w = image.shape[:2]
    num_pixels = int(amount * h * w)

    # Salt
    coords = (np.random.randint(0, h, num_pixels),
              np.random.randint(0, w, num_pixels))
    noisy[coords] = 255

    # Pepper
    coords = (np.random.randint(0, h, num_pixels),
              np.random.randint(0, w, num_pixels))
    noisy[coords] = 0

    return noisy
