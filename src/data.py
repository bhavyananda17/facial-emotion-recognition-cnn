"""Data handling: Kaggle download, CK+ train/test split, and Keras generators.

Command line (run from the project root):
    python -m src.data --config configs/ck.yaml --download
    python -m src.data --config configs/ck.yaml --split      # CK+ only
    python -m src.data --config configs/ck.yaml --inspect    # counts + class order
"""

import argparse
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from src.utils import ROOT, load_config, save_json

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# --------------------------------------------------------------------------
# 1. Download
# --------------------------------------------------------------------------
def download(cfg):
    """Download and unzip the Kaggle dataset into cfg['paths']['raw_dir'].

    Needs Kaggle credentials (~/.kaggle/kaggle.json or ~/.kaggle/access_token).
    """
    raw_dir = Path(cfg["paths"]["raw_dir"])
    if raw_dir.exists() and any(raw_dir.iterdir()):
        print(f"{raw_dir} already has files, skipping download.")
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Use the kaggle CLI that lives next to the running Python interpreter.
    kaggle = Path(sys.executable).parent / "kaggle"
    cmd = [str(kaggle), "datasets", "download", cfg["kaggle_slug"], "-p", str(raw_dir), "--unzip"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------
# 2. CK+ split (the dataset has no official train/test split)
# --------------------------------------------------------------------------
def _list_images(folder):
    """Image files in a folder, sorted so the result does not depend on the OS."""
    return sorted(f for f in Path(folder).iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


def split_ck(cfg):
    """80/20 split per class, as in FER_CK+.ipynb.

    Original: np.random.shuffle on os.listdir() order with no seed, then the
    first int(0.8 * n) images go to train. Here the file list is sorted first
    and shuffled with a seeded RNG, so the split is identical on every machine.
    A report of how many test images have sibling frames in train is saved.
    """
    raw_dir = Path(cfg["paths"]["raw_dir"])
    train_dir, test_dir = Path(cfg["paths"]["train_dir"]), Path(cfg["paths"]["test_dir"])
    for d in (train_dir, test_dir):  # start clean so a re-run never mixes two splits
        if d.exists():
            shutil.rmtree(d)

    rng = random.Random(cfg["split"]["seed"])
    ratio = cfg["split"]["train_ratio"]
    train_files, test_files = [], []
    for folder in cfg["class_folders"]:
        images = _list_images(raw_dir / folder)
        rng.shuffle(images)
        n_train = int(len(images) * ratio)  # same rounding as the notebook
        for subset, files, out in (("train", images[:n_train], train_dir), ("test", images[n_train:], test_dir)):
            (out / folder).mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, out / folder / f.name)
        train_files += images[:n_train]
        test_files += images[n_train:]

    report = ck_leakage_report(train_files, test_files)
    report.update(seed=cfg["split"]["seed"], train_images=len(train_files), test_images=len(test_files))
    save_json(report, ROOT / "results" / "ck" / "split_report.json")
    print(f"CK+ split done: {len(train_files)} train / {len(test_files)} test images.")
    print(f"{report['test_images_with_sibling_frame_in_train']} of {len(test_files)} test images "
          f"({report['fraction']:.0%}) come from a sequence that also has frames in train.")


def ck_leakage_report(train_files, test_files):
    """File names look like S137_011_00000020.png = subject, sequence, frame.

    Every sequence contributes 3 consecutive frames, which look almost the
    same. If a sequence has frames in both train and test, the test image has
    a near-duplicate in the training data.
    """
    def seq(f):
        return "_".join(Path(f).name.split("_")[:2])

    def subj(f):
        return Path(f).name.split("_")[0]

    train_seq, train_subj = {seq(f) for f in train_files}, {subj(f) for f in train_files}
    leaked_seq = sum(seq(f) in train_seq for f in test_files)
    leaked_subj = sum(subj(f) in train_subj for f in test_files)
    return {
        "test_images_with_sibling_frame_in_train": leaked_seq,
        "fraction": leaked_seq / len(test_files),
        "test_images_with_same_subject_in_train": leaked_subj,
        "fraction_same_subject": leaked_subj / len(test_files),
    }


# --------------------------------------------------------------------------
# 3. Generators
# --------------------------------------------------------------------------
def _datagen(cfg, augment, validation_split=0.0):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if not augment:  # test / validation: only rescale to [0, 1]
        return ImageDataGenerator(rescale=cfg["augmentation"]["rescale"], validation_split=validation_split)
    a = cfg["augmentation"]
    return ImageDataGenerator(
        rescale=a["rescale"],
        rotation_range=a["rotation_range"],
        width_shift_range=a["width_shift_range"],
        height_shift_range=a["height_shift_range"],
        shear_range=a["shear_range"],
        zoom_range=a["zoom_range"],
        horizontal_flip=a["horizontal_flip"],
        validation_split=validation_split,
    )


def _flow(datagen, cfg, directory, shuffle, subset=None):
    return datagen.flow_from_directory(
        directory,
        target_size=tuple(cfg["image"]["size"]),
        batch_size=cfg["training"]["batch_size"],
        color_mode=cfg["image"]["color_mode"],
        class_mode="categorical",
        shuffle=shuffle,
        subset=subset,
        seed=cfg["training"]["seed"],
    )


def make_generators(cfg, protocol):
    """Return (train_gen, monitor_gen, test_gen).

    train_gen   augmented training images. Both protocols use validation_split
                =0.1 with subset='training', exactly like the notebooks, so the
                10% held out per class is NOT trained on in either protocol.
    monitor_gen what early stopping / ReduceLROnPlateau watch:
                  paper -> the test set (what the notebooks do: leakage)
                  clean -> the 10% held-out slice of the training folder
                           (stratified because Keras holds out 10% of every class)
    test_gen    the test set, rescaled only, shuffle=False so labels line up
                with predictions.
    """
    if protocol not in ("paper", "clean"):
        raise ValueError(f"protocol must be 'paper' or 'clean', got {protocol!r}")
    split = cfg["training"]["validation_split"]
    train_dir, test_dir = cfg["paths"]["train_dir"], cfg["paths"]["test_dir"]

    train_gen = _flow(_datagen(cfg, augment=True, validation_split=split), cfg, train_dir, True, "training")
    test_gen = _flow(_datagen(cfg, augment=False), cfg, test_dir, False)
    if protocol == "paper":
        monitor_gen = test_gen
    else:
        # Same directory and same validation_split, so Keras selects exactly the
        # files the training generator left out. No augmentation on validation.
        monitor_gen = _flow(_datagen(cfg, augment=False, validation_split=split), cfg, train_dir, False, "validation")
        overlap = set(train_gen.filepaths) & set(monitor_gen.filepaths)
        assert not overlap, f"validation images leaked into training: {len(overlap)}"

    for g in (train_gen, test_gen, monitor_gen):
        check_class_order(cfg, g)
    return train_gen, monitor_gen, test_gen


def check_class_order(cfg, generator):
    """Fail loudly if Keras' class order differs from the config.

    The official FER2013 / RAF-DB notebooks hard-code a label list that does
    not match Keras' alphabetical folder order, which mislabels their plots.
    """
    found = [name for name, _ in sorted(generator.class_indices.items(), key=lambda kv: kv[1])]
    assert found == cfg["class_folders"], (
        f"Class order mismatch.\n  Keras found: {found}\n  config says: {cfg['class_folders']}"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def inspect(cfg):
    """Print image counts per class for the train and test folders."""
    for name in ("train_dir", "test_dir"):
        folder = Path(cfg["paths"][name])
        if not folder.exists():
            print(f"{name}: {folder} not found")
            continue
        counts = Counter({d.name: len(_list_images(d)) for d in sorted(folder.iterdir()) if d.is_dir()})
        print(f"{name}: {sum(counts.values())} images  {dict(counts)}")
    print("class order (index -> folder -> name):")
    for i, (folder, label) in enumerate(cfg["classes"].items()):
        print(f"  {i} -> {folder} -> {label}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--download", action="store_true", help="download the dataset from Kaggle")
    p.add_argument("--split", action="store_true", help="make the CK+ 80/20 split")
    p.add_argument("--inspect", action="store_true", help="print counts and class order")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.download:
        download(cfg)
    if args.split:
        if cfg["dataset"] != "ck":
            sys.exit("Only CK+ needs a split; FER2013 and RAF-DB ship with train/test folders.")
        split_ck(cfg)
    if args.inspect:
        inspect(cfg)


if __name__ == "__main__":
    main()
