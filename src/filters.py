
import numpy as np
import cv2
from typing import Tuple
from .utils import to_uint8, pad_to_odd

# ---------- Linear/Nonlinear blurs ----------
def mean_filter(img, ksize: int=3):
    k = pad_to_odd(ksize)
    return cv2.blur(img, (k, k))

def gaussian_filter(img, ksize: int=3, sigma: float=1.0):
    k = pad_to_odd(ksize)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)

def median_filter(img, ksize: int=3):
    k = pad_to_odd(ksize)
    return cv2.medianBlur(img, k)

def bilateral_filter(img, d: int=9, sigmaColor: float=75, sigmaSpace: float=75):
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

# ---------- Convolution from scratch (grayscale) ----------
def convolve2d(img_gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    padded = np.pad(img_gray, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')
    out = np.zeros_like(img_gray, dtype=np.float32)
    kernel_flipped = np.flipud(np.fliplr(kernel))
    for y in range(img_gray.shape[0]):
        for x in range(img_gray.shape[1]):
            region = padded[y:y+kh, x:x+kw]
            out[y, x] = np.sum(region * kernel_flipped)
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)

# ---------- Edge detectors ----------
def sobel_kernels():
    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1,-2,-1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]], dtype=np.float32)
    return gx, gy

def prewitt_kernels():
    gx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1,-1,-1],
                   [ 0, 0, 0],
                   [ 1, 1, 1]], dtype=np.float32)
    return gx, gy

def laplacian_kernel():
    return np.array([[0, 1, 0],
                     [1,-4, 1],
                     [0, 1, 0]], dtype=np.float32)

def sobel(img_gray):
    gx, gy = sobel_kernels()
    ix = convolve2d(img_gray, gx)
    iy = convolve2d(img_gray, gy)
    mag = np.sqrt(ix.astype(np.float32)**2 + iy.astype(np.float32)**2)
    mag = (mag / mag.max() * 255.0).astype(np.uint8)
    return ix, iy, mag

def prewitt(img_gray):
    gx, gy = prewitt_kernels()
    ix = convolve2d(img_gray, gx)
    iy = convolve2d(img_gray, gy)
    mag = np.sqrt(ix.astype(np.float32)**2 + iy.astype(np.float32)**2)
    mag = (mag / mag.max() * 255.0).astype(np.uint8)
    return ix, iy, mag

def laplacian(img_gray):
    k = laplacian_kernel()
    return convolve2d(img_gray, k)

def canny(img_gray, low_thresh: int=100, high_thresh: int=200):
    return cv2.Canny(img_gray, low_thresh, high_thresh)
