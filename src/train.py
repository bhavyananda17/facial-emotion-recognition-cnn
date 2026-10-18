"""Train one model and write every output.

    python -m src.train --config configs/ck.yaml --protocol paper
    python -m src.train --config configs/ck.yaml --protocol clean

Protocols (see README):
    paper  early stopping / LR reduction watch the TEST set, as in the official notebooks.
    clean  they watch a stratified 10% slice of the training set; the test set is used once, at the end.

Everything else (model, optimizer, augmentation, callbacks, seed) is identical.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ROOT, device_info, load_config, save_json, set_seed


class Tee:
    """Copy everything printed to the screen into a log file as well."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", buffering=1)
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="e.g. configs/ck.yaml")
    p.add_argument("--protocol", required=True, choices=["paper", "clean"])
    p.add_argument("--epochs", type=int, help="override the config's epochs (only for quick tests)")
    p.add_argument("--seed", type=int, help="training seed instead of the config's 42 (weights, augmentation, "
                                            "shuffling). The CK+ split keeps its own fixed seed. Results go to "
                                            "<protocol>_seed<N>/ so the headline run is never overwritten.")
    p.add_argument("--run-name", help="output folder name instead of the protocol (only for quick tests)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    run = args.run_name or (f"{args.protocol}_seed{args.seed}" if args.seed is not None else args.protocol)
    out_dir = ROOT / "results" / cfg["dataset"] / run
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out_dir / "train.log")

    # Imports that load TensorFlow come after the log is open so their messages are captured.
    import keras

    from src.data import make_generators
    from src.evaluate import evaluate_model
    from src.model import build_from_config

    started = datetime.now().isoformat(timespec="seconds")
    t = cfg["training"]
    print(f"=== {cfg['dataset']} | protocol={args.protocol} | seed={t['seed']} | started {started}")
    dev = device_info()
    print("device:", dev["device"], dev["gpus"], "| TF", dev["tensorflow"], "| Keras", dev["keras"])

    # 1. Seed, data, model
    set_seed(t["seed"])
    train_gen, monitor_gen, test_gen = make_generators(cfg, args.protocol)
    model = build_from_config(cfg)
    summary = []
    model.summary(print_fn=summary.append)
    (out_dir / "model_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary[-6:]))
    print(f"train={train_gen.samples}  monitored ({'TEST set' if args.protocol == 'paper' else 'validation slice'})"
          f"={monitor_gen.samples}  test={test_gen.samples}")

    # 2. Callbacks: same arguments as the notebooks
    es_cfg, lr_cfg = cfg["callbacks"]["early_stopping"], cfg["callbacks"]["reduce_lr"]
    early_stop = keras.callbacks.EarlyStopping(monitor=es_cfg["monitor"], patience=es_cfg["patience"], verbose=1,
                                               restore_best_weights=es_cfg["restore_best_weights"])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=lr_cfg["monitor"], factor=lr_cfg["factor"],
                                                  patience=lr_cfg["patience"], verbose=1,
                                                  min_delta=lr_cfg["min_delta"])

    # 3. Train. validation_data is what the callbacks monitor (test set in 'paper', slice of train in 'clean').
    start = time.time()
    history = model.fit(train_gen, epochs=t["epochs"], validation_data=monitor_gen,
                        callbacks=[early_stop, reduce_lr], verbose=2)
    train_seconds = time.time() - start
    hist = history.history
    epochs_run = len(hist["loss"])
    best_epoch = int(np.argmin(hist["val_loss"])) + 1  # epoch whose weights restore_best_weights keeps
    print(f"\ntraining finished: {epochs_run} epochs in {train_seconds / 60:.1f} min, best epoch {best_epoch}")

    # 4. Save model and history
    model_file = ROOT / "models" / f"{cfg['dataset']}_{run}.keras"
    model_file.parent.mkdir(exist_ok=True)
    model.save(model_file)
    hist_df = pd.DataFrame(hist)
    hist_df.insert(0, "epoch", np.arange(1, epochs_run + 1))
    hist_df.to_csv(out_dir / "history.csv", index=False)

    # 5. Evaluate on the test set (one pass) and write plots and reports
    metrics = evaluate_model(model, cfg, args.protocol, test_gen, hist, out_dir=out_dir)

    save_json({
        "dataset": cfg["dataset"], "protocol": args.protocol, "run_name": run, "config": args.config,
        "started": started, "git_commit": git_commit(), "seed": t["seed"],
        "total_params": int(model.count_params()),
        "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "paper_total_params": cfg["paper"]["total_params"],
        "train_images": train_gen.samples, "monitored_images": monitor_gen.samples, "test_images": test_gen.samples,
        "training_time_seconds": round(train_seconds, 1), "epochs_max": t["epochs"], "epochs_run": epochs_run,
        "best_epoch": best_epoch, "stopped_early": bool(early_stop.stopped_epoch > 0),
        "final_learning_rate": float(hist["learning_rate"][-1]) if "learning_rate" in hist else None,
        "device": device_info(), "metrics": {k: v for k, v in metrics.items() if not isinstance(v, dict)},
    }, out_dir / "run_info.json")

    print(f"\nTEST ACCURACY: {metrics['test_accuracy']:.4f}  (paper reports {cfg['paper']['test_accuracy']:.2f})")
    print(f"macro F1 {metrics['macro_f1']:.3f} | macro AUC {metrics['macro_auc']:.3f}")
    print(f"outputs in {out_dir.relative_to(ROOT)}, model in {model_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
