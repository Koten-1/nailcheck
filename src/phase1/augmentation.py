import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from config import IMAGE_SIZE, AUG_PER_IMAGE


augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.7),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.8
    ),
    A.GaussNoise(p=0.3),
    A.RandomScale(scale_limit=0.1, p=0.5),
    A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
])


def augment_image(img: np.ndarray) -> list[np.ndarray]:
    results = []
    for _ in range(AUG_PER_IMAGE):
        augmented = augment(image=img)["image"]
        results.append(augmented)
    return results


def save_augmented(img: np.ndarray, output_dir: Path, base_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    augmented_images = augment_image(img)
    for i, aug_img in enumerate(augmented_images):
        out_path = output_dir / f"{base_name}_aug{i}.jpg"
        cv2.imwrite(str(out_path), aug_img)
        print(f"  [SAVED] {out_path.name}")