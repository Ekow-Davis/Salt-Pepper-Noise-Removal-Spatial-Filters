import cv2
import os

def median_denoise(image_path, output_path, kernel_size=3):
    img = cv2.imread(image_path)
    denoised = cv2.medianBlur(img, kernel_size)
    cv2.imwrite(output_path, denoised)
