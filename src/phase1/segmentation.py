import cv2
import numpy as np
from pathlib import Path
from config import IMAGE_SIZE, MIN_NAIL_AREA


def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img


def segment_nail(img: np.ndarray) -> np.ndarray | None:
    # Convert to HSV for skin tone detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Skin tone range (works across light/dark skin)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick the largest contour
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_NAIL_AREA:
        return None

    # Crop to bounding box
    x, y, w, h = cv2.boundingRect(largest)
    cropped = img[y:y+h, x:x+w]

    return cropped


def resize_image(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def process_single_image(image_path: str) -> np.ndarray | None:
    img = load_image(image_path)
    cropped = segment_nail(img)
    if cropped is None:
        print(f"  [SKIP] No nail detected: {Path(image_path).name}")
        return None
    resized = resize_image(cropped)
    return resized