import os
import cv2
import pandas as pd
import torch

from utils.add_noise import add_salt_pepper_noise
from utils.metrics import evaluate
from baseline.median_filter import median_denoise
from adaptive.adaptive_filter import adaptive_filter
from cnn.model import SimpleDenoiser

NOISE_LEVEL = 0.3

DATA_ORIG = "data/original"
DATA_NOISY = "data/noisy"
DATA_CLEAN = "data/clean"

MODES = ["grayscale", "colored"]

results = []

# Load CNN (color only)
cnn_model = SimpleDenoiser()
cnn_model.load_state_dict(torch.load("cnn_denoiser.pth", map_location="cpu"))
cnn_model.eval()

for mode in MODES:
    print(f"\n=== Processing {mode.upper()} images ===")

    orig_dir = f"{DATA_ORIG}/{mode}"
    noisy_dir = f"{DATA_NOISY}/{mode}"
    os.makedirs(noisy_dir, exist_ok=True)

    for sub in ["median", "adaptive"]:
        os.makedirs(f"{DATA_CLEAN}/{sub}/{mode}", exist_ok=True)

    if mode == "colored":
        os.makedirs(f"{DATA_CLEAN}/cnn/{mode}", exist_ok=True)

    for img_name in os.listdir(orig_dir):
        path = os.path.join(orig_dir, img_name)

        clean = cv2.imread(path, cv2.IMREAD_GRAYSCALE if mode=="grayscale" else cv2.IMREAD_COLOR)
        noisy = add_salt_pepper_noise(clean, NOISE_LEVEL)
        cv2.imwrite(os.path.join(noisy_dir, img_name), noisy)

        # Median
        med = median_denoise(noisy)
        cv2.imwrite(f"{DATA_CLEAN}/median/{mode}/{img_name}", med)
        psnr, ssim = evaluate(clean, med, mode=="colored")
        results.append([img_name, mode, "Median", psnr, ssim])

        # CNN (color only)
        if mode == "colored":
            tensor = torch.tensor(noisy).permute(2,0,1).float().unsqueeze(0)/255
            with torch.no_grad():
                cnn_out = cnn_model(tensor)[0].permute(1,2,0).numpy()*255
            cnn_out = cnn_out.astype("uint8")
            cv2.imwrite(f"{DATA_CLEAN}/cnn/{mode}/{img_name}", cnn_out)

            psnr, ssim = evaluate(clean, cnn_out, True)
            results.append([img_name, mode, "CNN", psnr, ssim])

        # Adaptive
        if mode == "grayscale":
            adap = adaptive_filter(noisy)
        else:
            chans = cv2.split(noisy)
            adap = cv2.merge([adaptive_filter(c) for c in chans])

        cv2.imwrite(f"{DATA_CLEAN}/adaptive/{mode}/{img_name}", adap)
        psnr, ssim = evaluate(clean, adap, mode=="colored")
        results.append([img_name, mode, "Adaptive", psnr, ssim])

df = pd.DataFrame(results, columns=["Image", "Mode", "Method", "PSNR", "SSIM"])
df.to_csv("results.csv", index=False)

print("\n✔ Experiment completed successfully.")
