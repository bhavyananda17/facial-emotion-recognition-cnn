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

## Project layout

```
configs/      one YAML per dataset with every hyperparameter
src/          utils.py, data.py, model.py, train.py, evaluate.py, predict.py, webcam_demo.py
results/      <dataset>/<protocol>/ plots, reports, logs (committed)
models/       trained .keras files (git-ignored, regenerate with src.train)
data/         Kaggle datasets (git-ignored)
```
