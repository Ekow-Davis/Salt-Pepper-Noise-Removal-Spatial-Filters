from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate(clean, restored, is_color):
    """
    Computes PSNR and SSIM.

    Args:
        clean (ndarray): Original image
        restored (ndarray): Denoised image
        is_color (bool): True if RGB

    Returns:
        tuple: (PSNR, SSIM)
    """
    psnr = peak_signal_noise_ratio(clean, restored)
    ssim = structural_similarity(
        clean,
        restored,
        channel_axis=2 if is_color else None
    )
    return psnr, ssim
