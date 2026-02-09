import numpy as np

def adaptive_filter(image, window=5):
    """
    Adaptive salt-and-pepper noise removal.
    Only replaces pixels with value 0 or 255.

    Args:
        image (ndarray): Grayscale image
        window (int): Neighborhood window size

    Returns:
        ndarray: Filtered image
    """
    pad = window // 2
    padded = np.pad(image, pad, mode="reflect")
    output = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 0 or image[i, j] == 255:
                region = padded[i:i+window, j:j+window]
                valid = region[(region != 0) & (region != 255)]

                if valid.size > 0:
                    output[i, j] = valid.mean()

    return output.astype("uint8")
