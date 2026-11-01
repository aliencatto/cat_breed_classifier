"""Train CatCNN from scratch.

Model selection uses the VALIDATION set only (best.pt = epoch with the best validation accuracy).
The test set is never loaded here.

    python -m src.train --out outputs/cnn                          # main model (augmentation ON)
    python -m src.train --no_aug --out outputs/cnn_noaug           # experiment: no augmentation
    python -m src.train --dropout 0.5 --out outputs/cnn_dropout05  # experiment: more dropout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .data import set_seed
from .loaders import get_train_val_loaders
from .model import CatCNN, count_params, get_device


def parse_args():
    p = argparse.ArgumentParser(description="Train CatCNN (from scratch) on 12 cat breeds")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3, help="starting learning rate (cosine decay to 0)")
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--width", type=int, default=32, help="channels of the first block (doubles per block)")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--no_aug", action="store_true", help="switch data augmentation off (for experiments)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--data_root", default="data")
    p.add_argument("--no_download", action="store_true")
    p.add_argument("--out", default="outputs/cnn", help="folder for best.pt, history.json, curves.png")
    return p.parse_args()

# evaluate(): one no-grad pass over a loader -> (loss, accuracy)
@torch.no_grad()
def evaluate(model, loader, loss_fn, device, bar=None):
    """One no-grad pass over `loader`; returns (mean loss, accuracy)."""
    model.eval()                                   # dropout off, BatchNorm uses its running statistics
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)          # data on the SAME device as the model
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * len(y)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        n += len(y)
        if bar is not None:
            bar.update(1)
    return total_loss / n, correct / n

# train_epoch(): the five-step loop: forward, loss, zero_grad, backward, step
def train_epoch(model, loader, loss_fn, optimizer, device, bar=None):
    """One pass over the training set. Returns (mean loss, accuracy) measured while training."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                          # 1. forward
        loss = loss_fn(logits, y)                  # 2. loss
        optimizer.zero_grad()                      # 3. clear the previous gradients
        loss.backward()                            # 4. backprop
        optimizer.step()                           # 5. update the weights
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        n += len(y)
        if bar is not None:
            bar.update(1)
    return total_loss / n, correct / n

#plot_curves(): train vs. validation loss / accuracy
def plot_curves(history, path, best_epoch=None):
    """Train vs. validation loss and accuracy - read the gap between the two lines."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3)
        if best_epoch:
            ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=1)  # chosen checkpoint
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)

# train(): epochs of train_epoch + a validation pass each epoch -> history
def train(model, train_loader, val_loader, args, classes, device, out):
    """Train for args.epochs; keep the checkpoint with the best VALIDATION accuracy. Returns history."""
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best = {"val_acc": -1.0, "epoch": 0}

    # ONE live progress bar for the whole training (all epochs, updated after every batch)
    bar = tqdm(total=args.epochs * (len(train_loader) + len(val_loader)), unit="batch")
    for epoch in range(1, args.epochs + 1):
        bar.set_description(f"epoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_epoch(model, train_loader, loss_fn, optimizer, device, bar)
        va_loss, va_acc = evaluate(model, val_loader, loss_fn, device, bar)
        scheduler.step()                           # cosine learning-rate decay, once per epoch

        for key, value in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[key].append(value)
        if va_acc > best["val_acc"]:               # model selection looks at VALIDATION only
            best = {"val_acc": va_acc, "epoch": epoch}
            torch.save({"model_state": model.state_dict(), "config": vars(args), "classes": classes,
                        "epoch": epoch, "val_acc": va_acc}, out / "best.pt")
        tqdm.write(f"epoch {epoch:3d}/{args.epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
                   f"| val loss {va_loss:.3f} acc {va_acc:.3f}")
    bar.close()

    with open(out / "history.json", "w") as f:
        json.dump({"config": vars(args), "best_epoch": best["epoch"], "best_val_acc": best["val_acc"],
                   "history": history}, f, indent=2)
    plot_curves(history, out / "curves.png", best["epoch"])
    return history, best


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, classes = get_train_val_loaders(
        args.data_root, args.batch_size, args.img_size, augment=not args.no_aug,
        seed=args.seed, val_frac=args.val_frac, download=not args.no_download)
    model = CatCNN(len(classes), args.width, args.dropout).to(device)
    print(f"device={device}  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  "
          f"classes={len(classes)}  params={count_params(model):,}  augmentation={'off' if args.no_aug else 'on'}")

    _, best = train(model, train_loader, val_loader, args, classes, device, out)
    print(f"\nbest validation accuracy {best['val_acc']:.3f} at epoch {best['epoch']}  ->  saved {out}/best.pt")
    print(f"next:  python -m src.evaluate --run {out}   (only for your FINAL model - it opens the test set)")


if __name__ == "__main__":
    main()
