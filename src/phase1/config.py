from pathlib import Path

# ── Root ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path("D:/nail_screener")

# ── Data paths ────────────────────────────────────────────────────────────────
RAW_DIR       = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
AUGMENTED_DIR = ROOT_DIR / "data" / "augmented"

# ── Class labels ──────────────────────────────────────────────────────────────
CLASSES = [
    "healthy",
    "iron_deficiency",
    "fungal",
    "nutrient_deficiency",
]

# ── Image settings ────────────────────────────────────────────────────────────
IMAGE_SIZE    = (224, 224)
MIN_NAIL_AREA = 1000

# ── Augmentation settings ─────────────────────────────────────────────────────
AUG_PER_IMAGE = 5

# ── Risk thresholds ───────────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "healthy":             0.5,
    "iron_deficiency":     0.4,
    "fungal":              0.45,
    "nutrient_deficiency": 0.4,
}