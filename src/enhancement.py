
import cv2
import numpy as np
from .utils import to_uint8, pad_to_odd

def unsharp_mask(img, ksize: int=5, sigma: float=1.0, amount: float=1.5, threshold: int=0):
    k = pad_to_odd(ksize)
    blurred = cv2.GaussianBlur(img, (k, k), sigma)
    low_contrast_mask = np.absolute(img.astype(np.int16) - blurred.astype(np.int16)) < threshold
    sharpened = img*(1+amount) - blurred*amount
    sharpened = np.where(low_contrast_mask, img, sharpened)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def laplacian_sharpen(img_gray):
    lap = cv2.Laplacian(img_gray, ddepth=cv2.CV_16S, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    sharp = cv2.add(img_gray, lap)
    return sharp

def hist_equalization(img_gray):
    return cv2.equalizeHist(img_gray)

def clahe_equalization(img_gray, clip_limit: float=2.0, tile_grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_gray)
