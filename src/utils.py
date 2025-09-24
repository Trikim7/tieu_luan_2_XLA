
import os
import cv2
import numpy as np
from typing import Tuple

def read_image(path: str, as_gray: bool=False):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    if as_gray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # return RGB for matplotlib-friendly display
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def to_uint8(img):
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def add_gaussian_noise(img: np.ndarray, mean: float=0.0, std: float=10.0) -> np.ndarray:
    noisy = img.astype(np.float32) + np.random.normal(mean, std, img.shape)
    return to_uint8(noisy)

def add_salt_pepper_noise(img: np.ndarray, amount: float=0.02, s_vs_p: float=0.5) -> np.ndarray:
    noisy = img.copy()
    num_total = img.size
    num_salt = int(num_total * amount * s_vs_p)
    num_pepper = int(num_total * amount * (1.0 - s_vs_p))
    # salt
    coords = tuple([np.random.randint(0, i - 1, num_salt) for i in img.shape])
    noisy[coords] = 255
    # pepper
    coords = tuple([np.random.randint(0, i - 1, num_pepper) for i in img.shape])
    noisy[coords] = 0
    return noisy

def save_image(path: str, img_rgb: np.ndarray) -> None:
    # Save as BGR for OpenCV
    cv2.imwrite(path, rgb2bgr(img_rgb))

def im2float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0

def pad_to_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1
