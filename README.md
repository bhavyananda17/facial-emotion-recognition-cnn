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

## Project layout

```
configs/      one YAML per dataset with every hyperparameter
src/          utils.py, data.py, model.py, train.py, evaluate.py, predict.py, webcam_demo.py
results/      <dataset>/<protocol>/ plots, reports, logs (committed)
models/       trained .keras files (git-ignored, regenerate with src.train)
data/         Kaggle datasets (git-ignored)
```
