"""CatCNN: the model build from scratch.

    input 3x128x128
    block1  [conv3x3 -> BatchNorm -> ReLU] x2   3 -> 32   then 2x2 max-pool   -> 32 x 64 x 64
    block2  [conv3x3 -> BatchNorm -> ReLU] x2  32 -> 64   then max-pool       -> 64 x 32 x 32
    block3  [conv3x3 -> BatchNorm -> ReLU] x2  64 -> 128  then max-pool       -> 128 x 16 x 16
    block4  [conv3x3 -> BatchNorm -> ReLU] x2 128 -> 256  then max-pool       -> 256 x 8 x 8
    global average pooling -> Dropout -> Linear(256 -> 12)

Run:  python -m src.model
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    """Two (conv3x3 - BatchNorm - ReLU) layers followed by a 2x2 max-pool (halves height and width)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CatCNN(nn.Module):
    def __init__(self, n_classes=12, width=32, dropout=0.3):
        super().__init__()
        w = width
        self.features = nn.Sequential(conv_block(3, w), conv_block(w, 2 * w),
                                      conv_block(2 * w, 4 * w), conv_block(4 * w, 8 * w))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(8 * w, n_classes))

    def forward(self, x):  # x: (batch, 3, H, W) -> class logits (batch, n_classes)
        return self.head(self.pool(self.features(x)).flatten(1))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer="auto"):
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_run(run_dir, device):
    """Rebuild the model saved by train.py. Returns (model, config dict, class names, checkpoint dict)."""
    path = Path(run_dir) / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - train first:  python -m src.train --out {run_dir}")
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg, classes = ck["config"], ck["classes"]
    model = CatCNN(len(classes), cfg["width"], cfg["dropout"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, cfg, classes, ck


if __name__ == "__main__":
    m = CatCNN().eval()
    x = torch.zeros(1, 3, 128, 128)
    print(f"input   {tuple(x.shape)}")
    with torch.no_grad():
        for i, block in enumerate(m.features, 1):
            x = block(x)
            print(f"block{i}  {tuple(x.shape)}   params {count_params(block):>9,}")
    print(f"total trainable parameters: {count_params(m):,}")
