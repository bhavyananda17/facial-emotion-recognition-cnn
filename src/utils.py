"""Shared helpers: config loading, seeding, device detection and JSON output.

Run `python -m src.utils` to print the environment and check the Apple GPU.
"""

import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import yaml

# Project root = the folder that contains src/. All config paths are relative to it.
ROOT = Path(__file__).resolve().parent.parent


def load_config(path):
    """Read a YAML config and add a few derived fields used everywhere."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Folder names in sorted order = the class index order Keras will use.
    cfg["class_folders"] = list(cfg["classes"].keys())
    cfg["class_names"] = list(cfg["classes"].values())
    cfg["num_classes"] = len(cfg["class_folders"])
    # Turn relative data paths into absolute ones.
    cfg["paths"] = {k: str(ROOT / v) for k, v in cfg["paths"].items()}
    return cfg


def set_seed(seed):
    """Seed Python, NumPy and TensorFlow so runs are repeatable.

    ImageDataGenerator draws its augmentations from NumPy's global RNG and
    shuffles with the `seed` passed to flow_from_directory, so seeding NumPy
    here makes the augmentation stream repeatable as well. GPU kernels on
    Apple Metal are not guaranteed bit-for-bit deterministic, so two GPU runs
    can still differ slightly.
    """
    import keras

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)  # also seeds TensorFlow


def device_info():
    """Describe the hardware TensorFlow will use, for logs and run_info.json."""
    import keras
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    return {
        "device": "GPU (Apple Metal)" if gpus else "CPU",
        "gpus": [g.name for g in gpus],
        "machine": platform.machine(),
        "processor": platform.processor(),
        "os": f"macOS {platform.mac_ver()[0]}" if platform.mac_ver()[0] else platform.platform(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
    }


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def check_gpu():
    """Print the environment and time a 4096x4096 matmul on GPU vs CPU."""
    import tensorflow as tf

    info = device_info()
    for k, v in info.items():
        print(f"{k:11s}: {v}")
    if not info["gpus"]:
        print("No GPU found: training will run on the CPU.")
        return
    for dev in ("/GPU:0", "/CPU:0"):
        with tf.device(dev):
            a = tf.random.normal((4096, 4096))
            tf.matmul(a, a).numpy()  # warm-up
            t = time.time()
            tf.matmul(a, a).numpy()
            print(f"matmul 4096x4096 on {dev}: {time.time() - t:.3f} s")


if __name__ == "__main__":
    check_gpu()
