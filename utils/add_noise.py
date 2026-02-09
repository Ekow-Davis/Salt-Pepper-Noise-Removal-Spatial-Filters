import numpy as np

def add_salt_pepper_noise(image, amount=0.3):
    """
    Adds salt-and-pepper noise to an image.

    Args:
        image (ndarray): Input image
        amount (float): Noise density (0–1)

    Returns:
        ndarray: Noisy image
    """
    noisy = image.copy()
    h, w = image.shape[:2]
    num = int(amount * h * w)

    # Salt
    coords = (np.random.randint(0, h, num),
              np.random.randint(0, w, num))
    noisy[coords] = 255

    # Pepper
    coords = (np.random.randint(0, h, num),
              np.random.randint(0, w, num))
    noisy[coords] = 0

    return noisy
