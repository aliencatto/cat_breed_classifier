"""Predict the breed of one photo with a trained model.

Run: python -m src.predict --run outputs/cnn --image cat.jpg
"""
from __future__ import annotations

import argparse

import torch
from PIL import Image

from .data import resize_square
from .loaders import eval_transform
from .model import get_device, load_run


def main():
    ap = argparse.ArgumentParser(description="Predict the breed of a cat photo")
    ap.add_argument("--run", required=True, help="folder that contains best.pt")
    ap.add_argument("--image", required=True, help="path to a photo")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    model, cfg, classes, _ = load_run(args.run, device)
    img = Image.open(args.image).convert("RGB")
    x = eval_transform()(resize_square(img, cfg["img_size"])).unsqueeze(0).to(device)  # same resize as training
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()

    print(f"\n{args.image}")
    top = probs.topk(min(args.topk, len(classes)))
    for rank, (p, i) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
        print(f"  {rank}. {classes[i]:18s} {p * 100:5.1f}%")
    if top.values[0].item() < 0.5:
        print("  note: low confidence - the model is unsure about this photo.")
    print("  (the model only knows these 12 breeds, so any other animal is forced into one of them)")

if __name__ == "__main__":
    main()
