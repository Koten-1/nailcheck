import cv2
import numpy as np


def white_balance(img: np.ndarray) -> np.ndarray:
    # Gray world assumption — neutralizes color cast from lighting
    result = img.copy().astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3

    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / avg_b), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / avg_g), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / avg_r), 0, 255)

    return result.astype(np.uint8)


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    # Equalize only the V channel (brightness) in HSV — preserves color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = white_balance(img)
    img = histogram_equalization(img)
    return img