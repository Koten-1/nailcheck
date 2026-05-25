import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import cv2
from tqdm import tqdm
from config import RAW_DIR, PROCESSED_DIR, AUGMENTED_DIR, CLASSES
from segmentation import process_single_image
from normalization import normalize_image
from augmentation import save_augmented


def run_pipeline():
    print("=" * 50)
    print("PHASE 1 PIPELINE STARTING")
    print("=" * 50)

    for class_name in CLASSES:
        class_raw_dir = RAW_DIR / class_name
        class_processed_dir = PROCESSED_DIR / class_name
        class_augmented_dir = AUGMENTED_DIR / class_name
        class_processed_dir.mkdir(parents=True, exist_ok=True)

        # Collect all images including subfolders (batch_0, batch_1 etc)
        all_images = list(class_raw_dir.rglob("*.jpg")) + \
                     list(class_raw_dir.rglob("*.jpeg")) + \
                     list(class_raw_dir.rglob("*.png"))

        print(f"\n[{class_name}] Found {len(all_images)} raw images")

        saved = 0
        skipped = 0

        for img_path in tqdm(all_images, desc=f"Processing {class_name}"):
            try:
                # Step 1: segment + crop nail
                processed = process_single_image(str(img_path))
                if processed is None:
                    skipped += 1
                    continue

                # Step 2: normalize color + lighting
                normalized = normalize_image(processed)

                # Step 3: save to processed/
                out_path = class_processed_dir / img_path.name
                cv2.imwrite(str(out_path), normalized)

                # Step 4: augment + save to augmented/
                save_augmented(normalized, class_augmented_dir, img_path.stem)

                saved += 1

            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")
                skipped += 1

        print(f"  Saved: {saved} | Skipped: {skipped}")

    print("\n" + "=" * 50)
    print("PHASE 1 COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()