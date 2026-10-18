"""Evaluation: metrics, plots and reports for a trained model on the test set.

train.py calls evaluate_model() once training ends. It can also be run alone
on a saved model:
    python -m src.evaluate --config configs/ck.yaml --protocol paper
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNG files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix, f1_score,
                             roc_curve)
from sklearn.preprocessing import label_binarize

from src.utils import ROOT, load_config, save_json, set_seed

DPI = 300
# Okabe-Ito palette (colourblind-safe). The 7 ROC curves also differ in line style,
# so identity never depends on colour alone.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7A7A00"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-."]

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})


DISPLAY_NAME = {"ck": "CK+", "fer2013": "FER2013", "rafdb": "RAF-DB"}


def results_dir(cfg, protocol):
    return ROOT / "results" / cfg["dataset"] / protocol


def model_path(cfg, protocol):
    return ROOT / "models" / f"{cfg['dataset']}_{protocol}.keras"


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_history(history, out, protocol, title):
    """Accuracy and loss per epoch. In the paper protocol the second curve is the TEST set."""
    second = "Test (used as validation)" if protocol == "paper" else "Validation (held-out 10% of train)"
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key, name in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(epochs, history[key], color=COLORS[0], lw=1.8, label="Training")
        ax.plot(epochs, history["val_" + key], color=COLORS[1], lw=1.8, linestyle="--", label=second)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(f"{name} per epoch")
        ax.legend(loc="best")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_confusion(cm, names, out, title, normalised):
    """Confusion matrix: rows = true class, columns = predicted class."""
    data = cm / cm.sum(axis=1, keepdims=True) if normalised else cm
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    sns.heatmap(data, annot=True, fmt=".2f" if normalised else "d", cmap="Blues", cbar=True,
                vmin=0, vmax=1 if normalised else None, xticklabels=names, yticklabels=names,
                linewidths=0.5, linecolor="white", annot_kws={"size": 11}, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title + (" (row-normalised: recall per class)" if normalised else " (counts)"))
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_roc(y_true, probs, names, out, title):
    """One-vs-rest ROC curve per class with its AUC. Returns {class: auc}."""
    y_onehot = label_binarize(y_true, classes=list(range(len(names))))
    aucs = {}
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for i, name in enumerate(names):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], probs[:, i])
        aucs[name] = float(auc(fpr, tpr))
        ax.plot(fpr, tpr, color=COLORS[i % 7], linestyle=LINESTYLES[i % 7], lw=1.8,
                label=f"{name} (AUC = {aucs[name]:.3f})")
    ax.plot([0, 1], [0, 1], color="#555555", lw=1, linestyle=":", label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return aucs


def plot_prediction_grid(filepaths, y_true, y_pred, conf, names, size, out, title, n=25, seed=42):
    """5x5 random test images (seeded). Green title = correct, red = wrong."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(filepaths), size=min(n, len(filepaths)), replace=False)
    cols = 5
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.9))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, i in zip(np.ravel(axes), idx):
        img = Image.open(filepaths[i]).convert("L").resize((size[1], size[0]))
        ok = y_true[i] == y_pred[i]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"True: {names[y_true[i]]}\nPred: {names[y_pred[i]]} ({conf[i]:.0%})",
                     fontsize=9.5, color="#1a7f37" if ok else "#c62828", fontweight="bold")
    n_ok = int((y_true[idx] == y_pred[idx]).sum())
    fig.suptitle(f"{title}: {n_ok}/{len(idx)} correct in this sample (green = correct, red = wrong)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------
def evaluate_model(model, cfg, protocol, test_gen, history=None, out_dir=None):
    """Run the model over the whole test set once and write every output file.

    out_dir defaults to results/<dataset>/<protocol>/. Returns the headline metrics.
    """
    out = Path(out_dir) if out_dir else results_dir(cfg, protocol)
    out.mkdir(parents=True, exist_ok=True)
    names = cfg["class_names"]
    title = f"{DISPLAY_NAME.get(cfg['dataset'], cfg['dataset'])} / {protocol} protocol"

    test_gen.reset()
    probs = model.predict(test_gen, verbose=0)
    y_true = test_gen.classes  # valid because the test generator has shuffle=False
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    assert len(y_true) == len(y_pred), "prediction / label length mismatch"
    test_loss = float(model.evaluate(test_gen, verbose=0)[0])

    # Accuracy is computed from the predictions with sklearn. (The official notebooks
    # print Keras' evaluate() number, whose progress bar disagrees with it.)
    accuracy = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))

    # Per-class precision / recall / F1
    report = classification_report(y_true, y_pred, labels=list(range(len(names))), target_names=names,
                                   digits=3, zero_division=0, output_dict=True)
    (out / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, labels=list(range(len(names))), target_names=names,
                              digits=3, zero_division=0))
    pd.DataFrame(report).T.to_csv(out / "classification_report.csv")
    save_json(report, out / "classification_report.json")

    # Raw predictions, so every number can be re-derived
    pred_df = pd.DataFrame({
        "file": [str(Path(f).relative_to(ROOT)) for f in test_gen.filepaths],
        "true": [names[i] for i in y_true], "pred": [names[i] for i in y_pred], "confidence": conf,
    })
    for i, n in enumerate(names):
        pred_df[f"p_{n}"] = probs[:, i]
    pred_df.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(cm, index=names, columns=names).to_csv(out / "confusion_matrix.csv")

    # Plots
    if history is not None:
        plot_history(history, out / "training_curves.png", protocol, title)
    plot_confusion(cm, names, out / "confusion_matrix_counts.png", title, normalised=False)
    plot_confusion(cm, names, out / "confusion_matrix_normalised.png", title, normalised=True)
    aucs = plot_roc(y_true, probs, names, out / "roc_curves.png", f"One-vs-rest ROC curves, {title}")
    plot_prediction_grid(test_gen.filepaths, y_true, y_pred, conf, names, tuple(cfg["image"]["size"]),
                         out / "sample_predictions.png", title)

    metrics = {
        "test_accuracy": accuracy,
        "test_loss": test_loss,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_auc": float(np.mean(list(aucs.values()))),
        "auc_per_class": aucs,
        "recall_per_class": {n: float(report[n]["recall"]) for n in names},
        "n_test_images": int(len(y_true)),
        "paper_accuracy": cfg["paper"]["test_accuracy"],
    }
    save_json(metrics, out / "metrics.json")
    return metrics


def main():
    import keras

    from src.data import make_generators

    p = argparse.ArgumentParser(description="Evaluate a saved model on the test set.")
    p.add_argument("--config", required=True)
    p.add_argument("--protocol", required=True, choices=["paper", "clean"])
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    _, _, test_gen = make_generators(cfg, args.protocol)
    model = keras.models.load_model(model_path(cfg, args.protocol))
    hist_file = results_dir(cfg, args.protocol) / "history.csv"
    history = pd.read_csv(hist_file).to_dict("list") if hist_file.exists() else None
    m = evaluate_model(model, cfg, args.protocol, test_gen, history)
    print(json.dumps({k: v for k, v in m.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
