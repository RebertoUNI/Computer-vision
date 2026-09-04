"""Train/evaluate the nearest-neighbor and one-vs-rest linear SVM classifiers."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def train_nearest_neighbor(histograms: np.ndarray, labels: np.ndarray) -> KNeighborsClassifier:
    """Fit the requested 1-nearest-neighbor classifier on training histograms."""
    classifier = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=-1)
    return classifier.fit(histograms, labels)


def train_linear_svm(
    histograms: np.ndarray, labels: np.ndarray, c_value: float, seed: int
) -> OneVsRestClassifier:
    """Fit one binary linear SVM per class (positive class vs all remaining classes)."""
    classifier = OneVsRestClassifier(
        LinearSVC(C=c_value, class_weight="balanced", random_state=seed, max_iter=20_000),
        n_jobs=-1,
    )
    return classifier.fit(histograms, labels)


def save_svm(classifier: OneVsRestClassifier, class_names: Sequence[str], destination: Path) -> None:
    """Persist the trained multiclass SVM and its label mapping in models/svm_model.pkl."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        pickle.dump({"classifier": classifier, "class_names": list(class_names)}, file)


def predict_svm(classifier: OneVsRestClassifier, histograms: np.ndarray) -> np.ndarray:
    """Choose the class associated with the greatest one-vs-rest SVM margin."""
    scores = classifier.decision_function(histograms)
    return classifier.classes_[np.argmax(scores, axis=1)]


def _save_confusion_matrix(
    matrix: np.ndarray, class_names: Sequence[str], title: str, destination: Path
) -> None:
    """Save a readable confusion-matrix image; rows are true, columns predicted."""
    figure, axis = plt.subplots(figsize=(12, 10))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(ax=axis, cmap="Blues", colorbar=False, xticks_rotation=45, values_format="d")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def evaluate_and_save(
    name: str, true_labels: np.ndarray, predictions: np.ndarray,
    class_names: Sequence[str], test_paths: Sequence[Path], output_directory: Path
) -> float:
    """Evaluate one classifier and save its matrix plus per-image predictions."""
    matrix = confusion_matrix(true_labels, predictions, labels=np.arange(len(class_names)))
    accuracy = float(accuracy_score(true_labels, predictions))

    # CSV stores exact values, suitable for an assignment report or spreadsheet.
    with (output_directory / f"{name}_confusion_matrix.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row.tolist()])
    _save_confusion_matrix(
        matrix, class_names, f"{name.upper()} confusion matrix",
        output_directory / f"{name}_confusion_matrix.png",
    )

    # Prediction CSV makes it easy to inspect the test images classified incorrectly.
    with (output_directory / f"{name}_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "true_class", "predicted_class"])
        for path, truth, prediction in zip(test_paths, true_labels, predictions):
            writer.writerow([str(path), class_names[truth], class_names[prediction]])

    print(f"{name.upper()} accuracy: {accuracy:.4%}")
    return accuracy
