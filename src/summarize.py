"""Collect all runs into two ready-to-paste markdown tables (outputs/results.md).

Table 1 - experiments, VALIDATION numbers only (one change per run, compared with the baseline run).
Table 2 - TEST numbers: the baselines and every run that was evaluated with src/evaluate.py.

Run: python -m src.summarize
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KNOBS = ["epochs", "lr", "weight_decay", "dropout", "label_smoothing", "width", "img_size", "batch_size", "no_aug"]


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser(description="Summarise all runs in outputs/")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--baseline_run", default="cnn", help="name of the run the others are compared with")
    args = ap.parse_args()
    root = Path(args.outputs)

    runs = {p.parent.name: read_json(p) for p in sorted(root.glob("*/history.json"))}
    if not runs:
        raise SystemExit(f"No runs found in {root}/ - train first:  python -m src.train --out {root}/cnn")
    base = args.baseline_run if args.baseline_run in runs else next(iter(runs))

    lines = ["| Run | Change vs. baseline | Best val acc | Train acc (same epoch) | Train-val gap | Best epoch |",
             "|---|---|---|---|---|---|"]
    for name, r in runs.items():
        diffs = [f"{k}={r['config'][k]}" for k in KNOBS if r["config"][k] != runs[base]["config"][k]]
        change = "baseline" if name == base else (", ".join(diffs) or "-")
        train_acc = r["history"]["train_acc"][r["best_epoch"] - 1]
        lines.append(f"| `{name}` | {change} | {r['best_val_acc']:.3f} | {train_acc:.3f} | "
                     f"{train_acc - r['best_val_acc']:+.3f} | {r['best_epoch']} |")

    for name in runs:
        if (root / name / "test_metrics.json").exists():
            m = read_json(root / name / "test_metrics.json")
            lo, hi = m["test_acc_95ci"]
            lines.append(f"| CatCNN `{name}` | {m['test_acc']:.3f} (95% {lo:.3f}-{hi:.3f}) | "
                         f"{m['test_macro_f1']:.3f} |")

    text = "\n".join(lines) + "\n"
    (root / "results.md").write_text(text)
    print(text)
    print(f"saved -> {root / 'results.md'}")


if __name__ == "__main__":
    main()
