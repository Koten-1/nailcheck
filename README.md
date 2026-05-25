# NailCheck 🩺
### AI-Powered Nutritional Deficiency Screener from Fingernail Photography

> "The signs are already there on everyone's fingers. NailCheck makes them readable."

---

## Problem
Over 4 billion people globally are micronutrient deficient — and most never find out. Blood tests require clinical access, cost money, and need a doctor who already suspects the problem. Meanwhile, every nutritional deficiency leaves visible marks on the fingernail (koilonychia, leukonychia, Beau's lines) that go unread because reading them requires clinical training.

**NailCheck closes this gap** — a zero-cost smartphone AI that screens for deficiency signs from a single nail photo and prompts early clinical follow-up.

---

## How It Works
```
📷 Photograph nail  →  🧠 AI analyzes signs  →  📋 Risk report + next step
```

---

## Model Performance
- **Architecture:** EfficientNet-B0 (Transfer Learning)
- **Validation Accuracy:** 92.8%
- **Training Strategy:** Focal Loss + Sensitivity-first (Recall > Precision)
- **Dataset:** 1,174 images → 1,380 after augmentation (4 classes)

---

## Target Deficiencies
| Class | Nail Sign | Recommended Test |
|-------|-----------|-----------------|
| Iron Deficiency | Koilonychia, pale nail bed | Serum Ferritin |
| Fungal | Discoloration, thickening | KOH test |
| Nutrient Deficiency | Leukonychia, Beau's lines | CBC panel |
| Healthy | — | — |

---

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Image Processing | OpenCV, Albumentations |
| Model | PyTorch, EfficientNet-B0 |
| Backend (planned) | FastAPI |
| Language | Python 3.10 |

---

## Roadmap
- [x] Phase 1 — Data pipeline
- [x] Phase 2 — Model training (92.8% val accuracy)
- [ ] Phase 3 — FastAPI backend
- [ ] Phase 4 — Mobile UI (Thai language)

---

*NailCheck · MSIC 2026*
