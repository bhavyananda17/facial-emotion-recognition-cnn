"""Predict the emotion in a single face image.

    python -m src.predict path/to/face.jpg
    python -m src.predict path/to/face.jpg --config configs/rafdb.yaml --protocol paper

The image must already be a cropped face (use the webcam demo for full frames).
This file also holds the preprocessing shared with src/webcam_demo.py, so the demo
feeds the model exactly what training did.
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's startup chatter

import cv2
import numpy as np

from src.utils import ROOT, load_config

# Which trained model the demo and predict use unless told otherwise.
# FER2013: its training images are small, in-the-wild grayscale faces, closest to a webcam crop,
# and it has a Neutral class (CK+ does not). See README for the reasoning.
DEFAULT_CONFIG = "configs/fer2013.yaml"
DEFAULT_PROTOCOL = "paper"


def load_model(config_path, protocol):
    """Load the trained model and its config. Fails with a clear message if it was not trained."""
    import keras

    cfg = load_config(config_path)
    path = ROOT / "models" / f"{cfg['dataset']}_{protocol}.keras"
    if not path.exists():
        raise SystemExit(f"No trained model at {path.relative_to(ROOT)}.\n"
                         f"Train it first: python -m src.train --config {config_path} --protocol {protocol}")
    return keras.models.load_model(path), cfg


def preprocess(face, size):
    """Turn a face crop (BGR/RGB/gray uint8 array) into a model input of shape (1, H, W, 1).

    Same steps as training: grayscale, resize to the model's input size, divide by 255.
    cv2.INTER_AREA is the right choice when shrinking a big webcam crop to 48x48.
    """
    if face.ndim == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return (face.astype("float32") / 255.0)[None, :, :, None]


def predict_probs(model, face, size):
    """Class probabilities (sum to 1) for one face crop."""
    return model.predict(preprocess(face, size), verbose=0)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="path to a cropped face image")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--protocol", default=DEFAULT_PROTOCOL, choices=["paper", "clean"])
    p.add_argument("--top", type=int, default=3, help="how many classes to print")
    args = p.parse_args()

    img = cv2.imread(args.image)  # BGR; None if the file is missing or not an image
    if img is None:
        raise SystemExit(f"Could not read an image from {args.image}")
    model, cfg = load_model(args.config, args.protocol)
    probs = predict_probs(model, img, cfg["image"]["size"])

    print(f"model: {cfg['dataset']} / {args.protocol}   input: {cfg['image']['size']}")
    for i in np.argsort(probs)[::-1][: args.top]:
        print(f"  {cfg['class_names'][i]:<10s} {probs[i]:6.1%}")


if __name__ == "__main__":
    main()
