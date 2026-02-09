import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_metrics(clean_path, denoised_path):
    clean = cv2.imread(clean_path)
    denoised = cv2.imread(denoised_path)

    psnr = peak_signal_noise_ratio(clean, denoised)
    ssim = structural_similarity(clean, denoised, channel_axis=2)

    return psnr, ssim
