"""One-shot evaluation on the held-out TEST set. Run it ONCE, for your final model.

    test_metrics.json     accuracy (+95% interval), macro-F1, top-3, per-breed recall, top confusions
    confusion_matrix.png  row-normalised, with image counts
    failure_cases.png     the 8 MOST CONFIDENT wrong predictions (not cherry-picked)

Run: python -m src.evaluate --run outputs/(best train model)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

from .data import set_seed
from .loaders import get_test_loader
from .model import get_device, load_run

matplotlib.use("Agg")


@torch.no_grad()
def predict_probs(model, loader, device):
    """Softmax probabilities (N, classes) and true labels (N,) for every image in `loader`."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def plot_confusion(cm, classes, path):
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=60, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix on the TEST set (colour = row share, numbers = images)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def plot_failures(paths, y_true, y_pred, conf, classes, idx, path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(Image.open(paths[i]).convert("RGB"))
        ax.set_title(f"true: {classes[y_true[i]]}\npred: {classes[y_pred[i]]} ({conf[i]:.0%})",
                     fontsize=10, color="#c62828")
    fig.suptitle("Most CONFIDENT wrong predictions on the test set", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained run on the held-out test set (once!)")
    ap.add_argument("--run", required=True, help="folder that contains best.pt")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--baselines", default="outputs/baselines.json")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    run, device = Path(args.run), get_device(args.device)
    model, cfg, classes, ck = load_run(run, device)
    set_seed(cfg["seed"])
    loader, paths = get_test_loader(args.data_root, 64, cfg["img_size"], cfg["seed"], cfg["val_frac"])

    probs, y_true = predict_probs(model, loader, device)
    y_pred, conf = probs.argmax(axis=1), probs.max(axis=1)
    n = len(y_true)
    acc = float((y_pred == y_true).mean())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    recall = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    confusions = sorted(((int(cm[i, j]), classes[i], classes[j]) for i in range(len(classes))
                         for j in range(len(classes)) if i != j and cm[i, j] > 0), reverse=True)[:5]
    metrics = {
        "n_test": int(n),
        "test_acc": acc,
        "test_acc_95ci": [acc - 1.96 * float(np.sqrt(acc * (1 - acc) / n)),
                          acc + 1.96 * float(np.sqrt(acc * (1 - acc) / n))],  # normal approximation
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "test_top3_acc": float(np.mean([t in row for t, row in zip(y_true, np.argsort(-probs, axis=1)[:, :3])])),
        "test_loss": float(-np.mean(np.log(probs[np.arange(n), y_true] + 1e-12))),  # plain cross-entropy
        "majority_class_acc": float(np.bincount(y_true).max() / n),
        "per_breed_recall": {c: float(r) for c, r in zip(classes, recall)},
        "top_confusions": [{"count": c, "true": t, "predicted": p} for c, t, p in confusions],
        "checkpoint_epoch": int(ck["epoch"]),
        "val_acc_of_checkpoint": float(ck["val_acc"]),
    }
    (run / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(cm, classes, run / "confusion_matrix.png")
    wrong = np.where(y_pred != y_true)[0]
    worst = wrong[np.argsort(-conf[wrong])][:8]  # most confident mistakes first
    plot_failures(paths, y_true, y_pred, conf, classes, worst, run / "failure_cases.png")

    lo, hi = metrics["test_acc_95ci"]
    print(f"\nTEST accuracy {acc:.3f}  (95% interval {lo:.3f}-{hi:.3f})   n={n}")
    print(f"     macro-F1 {metrics['test_macro_f1']:.3f}   top-3 accuracy {metrics['test_top3_acc']:.3f}   "
          f"loss {metrics['test_loss']:.3f}")
    print(f"validation accuracy of this checkpoint (epoch {ck['epoch']}): {ck['val_acc']:.3f}")
    print(f"majority-class baseline on test: {metrics['majority_class_acc']:.3f}   "
          f"beat baseline: {'YES' if acc > metrics['majority_class_acc'] else 'NO - keep iterating'}")
    if Path(args.baselines).exists():
        b = json.loads(Path(args.baselines).read_text())
        print(f"logistic regression on pixels:   {b['logreg_pixels']['test_acc']:.3f}")
    print("weakest breeds (recall): " + ", ".join(
        f"{classes[i]} {recall[i]:.2f}" for i in np.argsort(recall)[:3]))
    print("most common confusions:  " + "; ".join(f"{t} -> {p} x{c}" for c, t, p in confusions[:3]))
    print(f"saved test_metrics.json, confusion_matrix.png, failure_cases.png to {run}/")


if __name__ == "__main__":
    main()
