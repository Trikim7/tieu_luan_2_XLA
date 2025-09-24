
import numpy as np
import math
try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

def mse(img1, img2):
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32))**2)

def psnr(img1, img2, max_pixel: float=255.0) -> float:
    m = mse(img1, img2)
    if m == 0:
        return float('inf')
    return 20 * math.log10(max_pixel) - 10 * math.log10(m)

def ssim(img1, img2, multichannel: bool=True):
    if ssim_fn is None:
        # very simple fallback: return 1 - normalized MSE (not a true SSIM)
        m = mse(img1, img2)
        return max(0.0, 1.0 - m / (255.0**2))
    return ssim_fn(img1, img2, channel_axis=-1 if multichannel else None)
