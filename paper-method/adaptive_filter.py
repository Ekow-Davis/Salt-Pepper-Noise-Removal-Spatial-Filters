import cv2
import numpy as np

def adaptive_filter(image, window=5):
    padded = cv2.copyMakeBorder(image, window, window, window, window, cv2.BORDER_REFLECT)
    output = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]

            if pixel == 0 or pixel == 255:
                region = padded[i:i+2*window+1, j:j+2*window+1]
                valid = region[(region != 0) & (region != 255)]

                if len(valid) > 0:
                    output[i, j] = np.mean(valid)

    return output
