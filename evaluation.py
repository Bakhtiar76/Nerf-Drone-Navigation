import os
import numpy as np
import imageio.v2 as imageio
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from scipy.ndimage import zoom


# CONFIG
PRED_DIR = "logs/lego/renderonly_test_199999"
GT_DIR   = "data/lego/test"
SAVE_DIR = "logs/lego/evaluation"
os.makedirs(SAVE_DIR, exist_ok=True)

PAPER_PSNR, PAPER_SSIM, PAPER_LPIPS = 32.54, 0.961, 0.050

# Load Files
pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith('.png')])
gt_files = [f"r_{i*8}.png" for i in range(25)]

print(f"Found {len(pred_files)} renders: {pred_files[0]} to {pred_files[-1]}")
print(f"Matched GT: {gt_files[0]} to {gt_files[-1]}")


# LPIPS Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net='vgg', version='0.1').to(device)


# Helper Functions
def load_img(path):
    img = imageio.imread(path).astype(np.float32) / 255.0
    if img.shape[-1] == 4:
        img = img[..., :3] * img[..., 3:] + (1.0 - img[..., 3:])
    return np.clip(img, 0, 1)

def resize_gt(img):
    if img.shape[0] == 800:
        return zoom(img, (0.5, 0.5, 1), order=1)
    return img

def save_comparison(gt, pred, pred_name, gt_name, psnr_val, ssim_val, lpips_val, save_path):
    """Save GT | NeRF | Error figure with colorbar and image names."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    plt.subplots_adjust(wspace=0.05)

    axs[0].imshow(gt)
    axs[0].set_title(f"GT\n{gt_name}", fontsize=12)
    axs[0].axis('off')

    axs[1].imshow(pred)
    axs[1].set_title(f"NeRF\n{pred_name}", fontsize=12)
    axs[1].axis('off')

    im = axs[2].imshow(np.abs(gt - pred), cmap='hot', vmin=0, vmax=0.3)
    axs[2].set_title("Error", fontsize=12)
    axs[2].axis('off')
    plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04, label='Absolute Error')

    plt.suptitle(f"PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.3f} | LPIPS: {lpips_val:.3f}",
                    fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


# Evaluation Loop
results = []
print("\nEvaluating 25 test views...")
for pred_name, gt_name in tqdm(zip(pred_files, gt_files), total=25):
    pred = load_img(os.path.join(PRED_DIR, pred_name))
    gt   = load_img(os.path.join(GT_DIR,   gt_name))
    gt   = resize_gt(gt)

    # Metrics
    psnr_val = psnr(gt, pred, data_range=1.0)
    ssim_val = ssim(gt, pred, channel_axis=2, data_range=1.0)
    gt_t = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0).to(device) * 2 - 1
    pred_t = torch.from_numpy(pred).permute(2,0,1).unsqueeze(0).to(device) * 2 - 1
    lpips_val = lpips_model(gt_t, pred_t).item()

    results = results + [{'render': pred_name, 'gt': gt_name,
                            'psnr': psnr_val, 'ssim': ssim_val, 'lpips': lpips_val}]

    # Save comparison
    save_comparison(gt, pred, pred_name, gt_name, psnr_val, ssim_val, lpips_val,
                    os.path.join(SAVE_DIR, f"comparison_{pred_name}"))


# Save CSV
df = pd.DataFrame(results)
df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"), index=False)

# Compute Stats
psnr_mean, psnr_std = df['psnr'].mean(), df['psnr'].std()
ssim_mean, ssim_std = df['ssim'].mean(), df['ssim'].std()
lpips_mean, lpips_std = df['lpips'].mean(), df['lpips'].std()

delta_psnr = psnr_mean - PAPER_PSNR
delta_ssim = ssim_mean - PAPER_SSIM
delta_lpips = lpips_mean - PAPER_LPIPS

# Print Summary
print("\n" + "="*60)
print("Test images: 25")
print("Resolution: 400x400")
print("LPIPS network: vgg (version 0.1)")
print("Your Implementation:")
print(f"  PSNR: {psnr_mean:.6f} +/- {psnr_std:.6f} dB")
print(f"  SSIM: {ssim_mean:.6f} +/- {ssim_std:.6f}")
print(f"  LPIPS: {lpips_mean:.6f} +/- {lpips_std:.6f}")
print("Paper Baseline:")
print(f"  PSNR: {PAPER_PSNR:.2f} dB")
print(f"  SSIM: {PAPER_SSIM:.3f}")
print(f"  LPIPS: {PAPER_LPIPS:.3f}")
print("Difference:")
print(f"  Delta PSNR: {delta_psnr:+.6f} dB")
print(f"  Delta SSIM: {delta_ssim:+.6f}")
print(f"  Delta LPIPS: {delta_lpips:+.6f}")
print("="*60)

# Save Summary
with open(os.path.join(SAVE_DIR, "summary.txt"), 'w') as f:
    f.write("Test images: 25\nResolution: 400x400\nLPIPS network: vgg (version 0.1)\n")
    f.write("Your Implementation:\n")
    f.write(f"  PSNR: {psnr_mean:.6f} +/- {psnr_std:.6f} dB\n")
    f.write(f"  SSIM: {ssim_mean:.6f} +/- {ssim_std:.6f}\n")
    f.write(f"  LPIPS: {lpips_mean:.6f} +/- {lpips_std:.6f}\n")
    f.write("Paper Baseline:\n")
    f.write(f"  PSNR: {PAPER_PSNR:.2f} dB\n  SSIM: {PAPER_SSIM:.3f}\n  LPIPS: {PAPER_LPIPS:.3f}\n")
    f.write("Difference:\n")
    f.write(f"  Delta PSNR: {delta_psnr:+.6f} dB\n  Delta SSIM: {delta_ssim:+.6f}\n  Delta LPIPS: {delta_lpips:+.6f}\n")

# Metric Distribution Plot
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(df['psnr'], bins=15, color='#4C72B0', edgecolor='black', alpha=0.8)
ax[0].axvline(PAPER_PSNR, color='red', linestyle='--', linewidth=2, label='Paper')
ax[0].axvline(psnr_mean, color='green', linewidth=2, label='Ours')
ax[0].set_xlabel('PSNR (dB)'); ax[0].set_title('PSNR Distribution'); ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].hist(df['ssim'], bins=15, color='#DD8452', edgecolor='black', alpha=0.8)
ax[1].axvline(PAPER_SSIM, color='red', linestyle='--', linewidth=2)
ax[1].axvline(ssim_mean, color='green', linewidth=2)
ax[1].set_xlabel('SSIM'); ax[1].set_title('SSIM Distribution'); ax[1].grid(alpha=0.3)

ax[2].hist(df['lpips'], bins=15, color='#55A868', edgecolor='black', alpha=0.8)
ax[2].axvline(PAPER_LPIPS, color='red', linestyle='--', linewidth=2)
ax[2].axvline(lpips_mean, color='green', linewidth=2)
ax[2].set_xlabel('LPIPS'); ax[2].set_title('LPIPS Distribution'); ax[2].grid(alpha=0.3)

plt.suptitle("Metric Distributions Across 25 Test Views", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "metric_distributions.png"), dpi=150, bbox_inches='tight')
plt.close()

print(f"\nResults saved in: {SAVE_DIR}")
print("   - 25 comparison figures")
print("   - metrics.csv")
print("   - metric_distributions.png")
print("   - summary.txt")