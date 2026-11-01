"""Data layer (no torch needed): download the cats, build train/val/test splits, decode images.

Dataset: Oxford-IIIT Pet, cats only (12 breeds), downloaded from the Hugging Face mirror
`timm/oxford-iiit-pet` (needs `pip install datasets`)

Splits: stratified 80/20 train/val (fixed seed). Official `test` is only
used by evaluate.py. A check asserts that train, val and test share no image.

Run: python -m src.data
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

CAT_BREEDS = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
    "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx",
]
HF_DATASET = "timm/oxford-iiit-pet"
FOLDER = "oxford-iiit-pet"
RESAMPLE = getattr(Image, "Resampling", Image).BICUBIC


def set_seed(seed: int = 42):
    """Fix every random number generator so runs reproduce (torch is optional here)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# download + annotations
def download_from_hf(base: Path):
    """Write the cat images and the two annotation files from Hugging Face into `base`."""
    try:
        from datasets import Image as HFImage, load_dataset
    except ImportError as e:
        raise RuntimeError("Downloading needs the 'datasets' package:  pip install datasets") from e

    print(f"Downloading cats from Hugging Face ({HF_DATASET}) ...")
    (base / "images").mkdir(parents=True, exist_ok=True)
    (base / "annotations").mkdir(parents=True, exist_ok=True)
    for hf_split, out_split in (("train", "trainval"), ("test", "test")):
        ds = load_dataset(HF_DATASET, split=hf_split)
        ds = ds.cast_column("image", HFImage(decode=False))  # keep the original JPEG bytes
        lines = []
        for row in ds:
            if row["label_cat_dog"] != 0:  # 0 = cat, 1 = dog
                continue
            name = row["image_id"]
            dest = base / "images" / f"{name}.jpg"
            if not dest.exists():
                dest.write_bytes(row["image"]["bytes"])
            lines.append(f"{name} {row['label'] + 1} 1 1")  # <id> <class> <species 1=cat> <breed>
        (base / "annotations" / f"{out_split}.txt").write_text("\n".join(lines) + "\n")
        print(f"  {out_split}: {len(lines)} cat images")


def parse_split(base: Path, split: str):
    """[(image_id, label 0..11)] for the cats listed in annotations/<split>.txt."""
    items = []
    for line in (base / "annotations" / f"{split}.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, _class_id, species, _breed_id = line.split()[:4]
        if int(species) != 1:  # 1 = cat, 2 = dog
            continue
        breed = name.rsplit("_", 1)[0].replace("_", " ")  # "Maine_Coon_12" -> "Maine Coon"
        if breed not in CAT_BREEDS:
            raise ValueError(f"Unexpected breed '{breed}' in '{name}'. Is the dataset corrupted?")
        items.append((name, CAT_BREEDS.index(breed)))
    return items


def ensure_dataset(root="data", download=True) -> Path:
    """Make sure annotations + all cat images exist (download them if not). Returns the dataset folder."""
    base = Path(root) / FOLDER

    def complete():
        for split in ("trainval", "test"):
            if not (base / "annotations" / f"{split}.txt").exists():
                return False
            if any(not (base / "images" / f"{n}.jpg").exists() for n, _ in parse_split(base, split)):
                return False
        return True

    if not complete():
        if not download:
            raise FileNotFoundError(f"Dataset not found in '{base}'. Run `python -m src.data` once "
                                    "(needs internet + `pip install datasets`).")
        download_from_hf(base)
        if not complete():
            raise RuntimeError(f"Dataset in '{base}' is still incomplete after downloading.")
    return base


# splits
def get_splits(root="data", val_frac=0.2, seed=42, download=True):
    """Return (train, val, test): lists of (image_path, label), in a deterministic order."""
    base = ensure_dataset(root, download)

    def read(split):
        return sorted((str(base / "images" / f"{n}.jpg"), y) for n, y in parse_split(base, split))

    trainval, test = read("trainval"), read("test")
    idx_train, idx_val = train_test_split(
        np.arange(len(trainval)), test_size=val_frac, random_state=seed,
        stratify=[y for _, y in trainval])
    train = [trainval[i] for i in sorted(idx_train)]
    val = [trainval[i] for i in sorted(idx_val)]

    # Leakage check: no image may appear in two splits.
    p_train, p_val, p_test = ({p for p, _ in s} for s in (train, val, test))
    assert not (p_train & p_val) and not (p_train & p_test) and not (p_val & p_test), "DATA LEAKAGE!"
    return train, val, test


def class_counts(items):
    return np.bincount([y for _, y in items], minlength=len(CAT_BREEDS))


# decoding
def resize_square(img: Image.Image, size: int) -> Image.Image:
    """The ONE resize used for training, evaluation and prediction (so there is no train/test skew)."""
    return img.convert("RGB").resize((size, size), RESAMPLE)


def load_images(items, size: int) -> np.ndarray:
    """Decode + resize every image once -> uint8 array of shape (N, size, size, 3)."""
    X = np.empty((len(items), size, size, 3), dtype=np.uint8)
    for i, (path, _) in enumerate(items):
        with Image.open(path) as im:
            X[i] = np.asarray(resize_square(im, size))
    return X


def main():
    ap = argparse.ArgumentParser(description="Download (if needed) and sanity-check the cat dataset.")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--no_download", action="store_true", help="never touch the network")
    args = ap.parse_args()

    train, val, test = get_splits(args.data_root, args.val_frac, args.seed, not args.no_download)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}   (no image shared between splits: OK)")
    print(f"{'breed':20s} {'train':>6s} {'val':>5s} {'test':>5s}")
    ct, cv, cs = class_counts(train), class_counts(val), class_counts(test)
    for i, breed in enumerate(CAT_BREEDS):
        print(f"{breed:20s} {ct[i]:6d} {cv[i]:5d} {cs[i]:5d}")
    for name, items in (("train", train), ("val", val), ("test", test)):
        print(f"decode check {name}: {load_images(items, 32).shape} OK")  # catches corrupt files early
    print(f"majority-class accuracy is about {ct.max() / ct.sum():.1%}  (chance = {1 / len(CAT_BREEDS):.1%})")


if __name__ == "__main__":
    main()
