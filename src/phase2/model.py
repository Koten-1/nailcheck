import torch
import torch.nn as nn
from torchvision import models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "phase1"))
from config import CLASSES


def build_model(pretrained=True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Replace the classifier head with our 4-class output
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(CLASSES)),
    )

    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total params: {total:,} | Trainable: {trainable:,}")


if __name__ == "__main__":
    model = build_model(pretrained=True)
    count_parameters(model)
    print("[MODEL] EfficientNet-B0 ready")

    # Quick shape test
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"[MODEL] Output shape: {out.shape} — expected (2, {len(CLASSES)})")