# Project 2 — Bag-of-Words image classifier

The implementation does not use a library's end-to-end bag-of-words pipeline.
It is split into small, commented modules:

```
project/
├── dataset/                 # Current dataset (can be renamed to data/)
│   ├── train/
│   └── test/
├── src/
│   ├── extract_features.py  # Dataset reading and SIFT extraction/sampling
│   ├── build_vocabulary.py  # k-means and models/vocabulary.pkl
│   ├── compute_histograms.py # Nearest-word assignment and normalized BoW rows
│   └── train_classifier.py  # 1-NN, one-vs-rest SVM, and evaluation
├── models/                  # Created automatically; vocabulary.pkl, svm_model.pkl
├── results/                 # Created automatically; matrices, predictions, metrics
└── main.py                  # Runs the complete experiment
```

## Method

1. SIFT detects keypoints and computes their 128-dimensional descriptors in
   every training image.
2. An approximately balanced sample of 50,000 training descriptors is clustered
   with mini-batch k-means. The resulting `k x 128` centroids are the visual
   vocabulary and are saved for reuse.
3. Every descriptor in each image is assigned to its closest centroid. The
   counts form a `k`-bin image histogram, which is L2-normalized by default.
4. A 1-nearest-neighbor classifier is trained on training histograms.
5. `OneVsRestClassifier(LinearSVC)` trains one linear binary SVM for each of
   the 15 classes. A test image gets the class with the highest SVM margin.

The source contains comments and docstrings for each logical operation.

## Install

Use a virtual environment if possible, then install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run

The included dataset already matches the expected layout (`dataset/train` and
`dataset/test`, each containing the 15 class folders), so run:

```bash
python3 main.py --dataset dataset --clusters 100 \
  --vocab-descriptors 50000 --output results/k100
```

For a small vocabulary comparison, use, for example, `--clusters 50`, `100`,
and `200`, each with a distinct output directory. Use the same descriptor count
and seed to make the comparison fair:

```bash
python3 main.py --dataset dataset --clusters 50  --output results/k50
python3 main.py --dataset dataset --clusters 100 --output results/k100
python3 main.py --dataset dataset --clusters 200 --output results/k200
```

The default 50,000 descriptor sample obeys the required 10K–100K range.
`--reuse-vocabulary` skips descriptor sampling and k-means when re-running an
experiment with the same `--clusters` and `--output` directory.

## Outputs

The selected `models/` directory contains:

- `models/vocabulary.pkl` — saved `k x 128` k-means centroids.
- `models/svm_model.pkl` — trained one-vs-rest linear SVM and label mapping.

Each output directory contains:
- `train_histograms.npz` and `test_histograms.npz` — normalized BoW features
  and labels.
- `nearest_neighbor_*` and `linear_svm_*` — predictions and confusion matrices
  in both CSV and PNG format.
- `metrics.json` — accuracy plus experiment parameters.

In each confusion matrix, **rows are true classes** and **columns are predicted
classes**.
