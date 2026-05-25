import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "phase1"))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from dataset import get_dataloaders
from model import build_model

# ── Settings ──────────────────────────────────────────────────────────────────
EPOCHS     = 20
LR         = 1e-4
BATCH_SIZE = 32
SAVE_PATH  = Path("D:/nail_screener/models")
SAVE_PATH.mkdir(parents=True, exist_ok=True)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}")

    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)
    model = build_model(pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100.0 * correct / total

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        print(f"  Epoch {epoch+1:02d} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Val Acc: {val_acc:.1f}%")

        # ── Save best model ───────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH / "best_model.pth")
            print(f"  [SAVED] New best model — Val Acc: {val_acc:.1f}%")

    print(f"\n[TRAIN] Done. Best Val Acc: {best_val_acc:.1f}%")
    print(f"[TRAIN] Model saved to {SAVE_PATH / 'best_model.pth'}")


if __name__ == "__main__":
    train()