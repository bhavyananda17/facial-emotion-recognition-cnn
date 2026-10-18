# Facial Emotion Recognition CNN: a reproduction

A clean, reproducible re-implementation of

> R. Raj and I. Demirkol, "An improved facial emotion recognition system using convolutional neural network for the optimization of human robot interaction", *Scientific Reports* 15:38940 (2025). [doi:10.1038/s41598-025-22835-0](https://doi.org/10.1038/s41598-025-22835-0)

The three official Keras notebooks ([raj91ravi/Facial-Emotion-Recognition](https://github.com/raj91ravi/Facial-Emotion-Recognition)) are treated as the source of truth for hyperparameters. The paper provides context. The code here is a rewrite, not a copy of the notebooks.

The model is trained on **CK+**, **FER2013** and **RAF-DB** under two protocols:

| Protocol | Validation data used for early stopping / LR schedule | Purpose |
|---|---|---|
| `paper` | the **test set** (exactly as in the notebooks) | numbers comparable to the paper |
| `clean` | a held-out 10% of the training set; test set used once at the end | an unbiased estimate |

> 🚧 Work in progress. Results, runtimes and the full "Differences from the original code" section are added as each milestone lands.

## Setup

Tested on a MacBook Pro (Apple M5 Pro, 24 GB) running macOS 26.6 with Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.utils        # prints the environment and checks the Apple GPU
```

Measured on this machine:

```
device     : GPU (Apple Metal)
tensorflow : 2.18.1
keras      : 3.8.0
matmul 4096x4096 on /GPU:0: 0.029 s
matmul 4096x4096 on /CPU:0: 0.100 s
```

### Environment notes

- **TensorFlow 2.18** is the version used in the paper (Table 3), and `tensorflow-metal 1.2.0` is the Metal plugin release built for it.
- **Keras is pinned to 3.8.0.** TF 2.18 only requires `keras>=3.5`, so pip would otherwise install the newest Keras (3.15 at the time of writing). Pinning keeps the environment close to what TF 2.18 was released with.
- On the Metal GPU, the `SeparableConv2D` layers (FER2013, RAF-DB) run slower than `Conv2D` at 48×48. At 100×100 the GPU is still faster than the CPU (131 vs 181 ms/step in a synthetic benchmark).

## Data

Datasets come from Kaggle (the links in the paper's Data availability section) and are git-ignored. Put your Kaggle credentials in `~/.kaggle/kaggle.json` or `~/.kaggle/access_token`, then:

```bash
python -m src.data --config configs/ck.yaml --download   # fetch into data/ck/raw
python -m src.data --config configs/ck.yaml --split      # CK+ only: seeded 80/20 split
python -m src.data --config configs/ck.yaml --inspect    # image counts and class order
```

FER2013 and RAF-DB already ship with train/test folders, so they only need `--download`.

### CK+ as downloaded

981 images, 48×48 grayscale PNGs, 7 classes (anger, contempt, disgust, fear, happy, sadness, surprise; no neutral). File names are `S<subject>_<sequence>_<frame>.png` and **every sequence contributes exactly 3 consecutive frames**.

| Split (seed 42) | Images |
|---|---|
| train | 783 (708 after Keras holds out 10% for `validation_split`) |
| test | 198 |
| clean-protocol validation | 75 |

These counts equal the ones in the official notebook's output (708 / 198) and the paper's Table 2 (783 / 198).

**Leakage inside the CK+ split.** The notebook splits *images*, not subjects or sequences. With our seeded split, **180 of the 198 test images (91%) have a sibling frame from the same sequence in the training folder**, and 195 of 198 (98%) come from a subject who also appears in training (`results/ck/split_report.json`). The clean protocol fixes the test-set-as-validation problem but not this near-duplicate overlap, so CK+ accuracy stays optimistic under both protocols.

### How the generators are built (`src/data.py`)

- Both protocols train on the same augmented training subset (`validation_split=0.1`, `subset='training'`), exactly like the notebooks, so the 10% held-out slice is never trained on.
- `paper`: that slice is ignored and the **test set** is the monitored validation data.
- `clean`: the held-out slice (`subset='validation'`, rescale only, a stratified 10% of every class) is monitored instead. The test set is only used for the final evaluation.
- `check_class_order` stops the run if Keras' class order differs from the config.

## Training and evaluation

```bash
python -m src.model                                              # check parameter counts vs the notebooks
python -m src.train --config configs/ck.yaml --protocol paper    # train + evaluate, writes results/ck/paper/
python -m src.evaluate --config configs/ck.yaml --protocol paper # re-evaluate a saved model
```

Each run writes to `results/<dataset>/<protocol>/`: `training_curves.png`, `confusion_matrix_counts.png`, `confusion_matrix_normalised.png`, `roc_curves.png` (one-vs-rest, per-class AUC), `sample_predictions.png` (green = correct, red = wrong), `classification_report.{txt,csv,json}`, `predictions.csv`, `metrics.json`, `model_summary.txt`, `history.csv`, `train.log` and `run_info.json` (parameters, training time, epochs, device, git commit). Models are saved to `models/<dataset>_<protocol>.keras`.

`src/model.py` reproduces the notebooks' parameter counts exactly: 7,057,351 (CK+), 2,907,984 (FER2013), 9,985,872 (RAF-DB).

## Results

### CK+, paper protocol (seed 42, Apple M5 Pro GPU)

| | This run | Paper |
|---|---|---|
| Test accuracy | **94.4%** (187/198) | 95% |
| Macro F1 / macro AUC | 0.924 / 0.998 | n/a |
| Training time | 2.2 min (100 epochs, ~1.3 s/epoch) | n/a |

Weakest class is Sadness (recall 0.71; 4 of 17 predicted as Anger). Details are in `results/ck/paper/`.

Things worth knowing when reading the curves:

- **Early stopping never fired.** All 100 epochs ran. The best epoch (87) was chosen by the *test* loss, which is what `restore_best_weights` restores.
- **The learning rate collapses.** `ReduceLROnPlateau(factor=0.001)` multiplies the rate by 0.001, not 0.1: it falls from 1e-3 to 1e-6 at epoch 38, 1e-9 at epoch 82 and 1e-12 at epoch 97, so the model is effectively frozen after epoch ~40. This matches the notebook's own log (its learning rate also reaches 1e-6).
- **Test accuracy (94%) is above training accuracy (88%).** Training accuracy is measured with dropout and augmentation switched on, and the monitored test images are near-duplicates of training frames (see the CK+ leakage note above).

### RAF-DB, paper protocol (seed 42, Apple M5 Pro GPU)

| | This run | Paper |
|---|---|---|
| Test accuracy | **83.0%** (2,546/3,068) | 83% |
| Macro F1 / macro AUC | 0.748 / 0.967 | n/a |
| Training time | 80.9 min (100 epochs, ~49 s/epoch) | n/a |

All 100 epochs ran (early stopping did not fire); the learning rate dropped to 1e-6 at epoch 45 and 1e-9 at epoch 96, and the best epoch by test loss was 99. Happy is easiest (recall 0.90); **Disgust (0.47) and Fear (0.55) are hardest**, and they are also the two smallest classes (717 and 281 training images against 4,772 for Happy). Disgust is most often mistaken for Neutral (21%) and Sad (14%).

**Label coding check.** The RAF-DB folders are named `1`–`7`. Their sizes (1,619 / 355 / 877 / 5,957 / 2,460 / 867 / 3,204 images over train + test) match the published RAF-DB class totals, and sample faces from each folder agree (`results/rafdb/label_check.png`), giving 1 Surprise, 2 Fear, 3 Disgust, 4 Happy, 5 Sad, 6 Angry, 7 Neutral. The official notebook labels its plots Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral, which is wrong for folders 1, 2, 3 and 6. Accuracy is unaffected; the notebook's per-class plots and tables are mislabelled. This repo uses the correct names everywhere.

### FER2013, paper protocol (seed 42, Apple M5 Pro GPU)

| | This run | Paper |
|---|---|---|
| Test accuracy | **65.3%** (4,688/7,178) | 64% |
| Macro F1 / macro AUC | 0.613 / 0.910 | n/a |
| Training time | 98.8 min (100 epochs, ~59 s/epoch) | n/a |

All 100 epochs ran; the learning rate dropped to 1e-6 at epoch 46 and the best epoch by test loss was the last one. Happy (recall 0.87), Surprise (0.80) and Neutral (0.73) are easiest. **Fear (0.36) and Disgust (0.40) are hardest**; Disgust has only 436 training images, and 39% of Disgust test images are predicted as Angry. FER2013's labels are known to be noisy, so a ceiling in the 60s to low 70s is normal and 65% is not a model failure.

**Independent evidence for the label mix-up in the paper's Table 4.** The notebook labels the sorted classes Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral, but the real order is Angry, Disgust, Fear, Happy, **Neutral, Sad, Surprise**. If that is what happened, the paper's rows for Sad, Surprise and Neutral should belong to my Neutral, Sad and Surprise. They do (precision / recall):

| Paper's row label | Paper (P / R) | My true class | Mine (P / R) |
|---|---|---|---|
| Sadness | 0.52 / 0.73 | **Neutral** | 0.521 / 0.732 |
| Surprise | 0.56 / 0.44 | **Sad** | 0.582 / 0.455 |
| Neutral | 0.74 / 0.80 | **Surprise** | 0.740 / 0.800 |

The unaffected rows also agree (Happy 0.86 / 0.87 vs 0.862 / 0.874; Angry 0.53 / 0.60 vs 0.563 / 0.620; Fear 0.55 / 0.34 vs 0.557 / 0.356). The paper's accuracy is correct but its per-class labels are shifted.

### Watching a long run

`python scripts/watch_training.py results/rafdb/paper/train.log` shows a live terminal dashboard (progress, ETA, one accuracy chart). It only reads the log.

## Project layout

```
configs/      one YAML per dataset with every hyperparameter
src/          utils.py, data.py, model.py, train.py, evaluate.py, predict.py, webcam_demo.py
results/      <dataset>/<protocol>/ plots, reports, logs (committed)
models/       trained .keras files (git-ignored, regenerate with src.train)
data/         Kaggle datasets (git-ignored)
```
