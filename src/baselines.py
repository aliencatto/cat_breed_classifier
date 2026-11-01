"""Baselines the CNN has to beat (no torch needed).

1. majority      : always predict the most common TRAIN breed.
2. logreg_pixels : logistic regression on raw 32x32 colour pixels (a simple linear model).

No test leakage: the regularisation strength C is chosen on the VALIDATION set; the test set is
scored once, with the chosen C.

Run: python -m src.baselines  keep in (outputs/baselines.json)
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from .data import CAT_BREEDS, get_splits, load_images, set_seed

SIZE = 32  # images are shrunk to 32x32 -> 3072 features


def scores(y_true, y_pred):
    return {"test_acc": float(accuracy_score(y_true, y_pred)),
            "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}


def main():
    ap = argparse.ArgumentParser(description="Majority-class and logistic-regression baselines")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out", default="outputs/baselines.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--no_download", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    train, val, test = get_splits(args.data_root, args.val_frac, args.seed, not args.no_download)
    y = {name: np.array([label for _, label in items]) for name, items in
         (("train", train), ("val", val), ("test", test))}
    X = {name: load_images(items, SIZE).reshape(len(items), -1).astype(np.float32) / 255.0
         for name, items in (("train", train), ("val", val), ("test", test))}
    print(f"train={len(train)} val={len(val)} test={len(test)}  classes={len(CAT_BREEDS)}")

    # 1) majority class, learned from the TRAIN labels only
    majority = int(np.bincount(y["train"]).argmax())
    maj = {"predicts": CAT_BREEDS[majority], "val_acc": float(np.mean(y["val"] == majority))}
    maj.update(scores(y["test"], np.full_like(y["test"], majority)))

    # 2) logistic regression on pixels: pick C on VALIDATION, then score TEST once
    scaler = StandardScaler().fit(X["train"])  # fitted on train only
    Z = {name: scaler.transform(x) for name, x in X.items()}
    best = None
    for C in (1e-4, 1e-3, 1e-2, 1e-1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf = LogisticRegression(C=C, max_iter=1000).fit(Z["train"], y["train"])
        val_acc = float(accuracy_score(y["val"], clf.predict(Z["val"])))
        print(f"  logistic regression C={C:<7g} val acc {val_acc:.3f}")
        if best is None or val_acc > best[0]:
            best = (val_acc, C, clf)
    val_acc, C, clf = best
    lr = {"C": C, "val_acc": val_acc}
    lr.update(scores(y["test"], clf.predict(Z["test"])))

    results = {"n_train": len(train), "n_val": len(val), "n_test": len(test), "seed": args.seed,
               "chance": 1 / len(CAT_BREEDS), "majority": maj, "logreg_pixels": lr}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n{'baseline':16s} {'val acc':>8s} {'test acc':>9s}")
    print(f"{'majority':16s} {maj['val_acc']:8.3f} {maj['test_acc']:9.3f}   (always '{maj['predicts']}')")
    print(f"{'logreg_pixels':16s} {lr['val_acc']:8.3f} {lr['test_acc']:9.3f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
