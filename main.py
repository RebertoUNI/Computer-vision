#!/usr/bin/env python3
"""Run the complete SIFT Bag-of-Words scene-classification experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.build_vocabulary import learn_vocabulary, load_vocabulary, save_vocabulary
from src.compute_histograms import build_histograms
from src.extract_features import make_sift, read_split, sample_descriptors
from src.train_classifier import (
    evaluate_and_save,
    predict_svm,
    save_svm,
    train_linear_svm,
    train_nearest_neighbor,
)


def parse_arguments() -> argparse.Namespace:
    """Define the command-line parameters for a reproducible experiment."""
    parser = argparse.ArgumentParser(description="SIFT Bag-of-Words classifier")
    # Use dataset/ for the provided files; pass --dataset data after renaming it.
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--clusters", type=int, default=100)
    # The default sample count is inside the assignment's requested 10K--100K range.
    parser.add_argument("--vocab-descriptors", type=int, default=50_000)
    parser.add_argument("--sift-features", type=int, default=0)
    parser.add_argument("--assignment-batch-size", type=int, default=4_096)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--normalization", choices=("l1", "l2"), default="l2")
    # Learned vocabulary and SVM are retained independently of experiment outputs.
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("results/bow"))
    parser.add_argument("--reuse-vocabulary", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Build BoW features, train the requested classifiers, and save evaluation files."""
    args = parse_arguments()
    args.models.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    # Read both splits and ensure a label index has the same meaning in each split.
    train_paths, train_labels, class_names = read_split(args.dataset / "train")
    test_paths, test_labels, test_class_names = read_split(args.dataset / "test")
    if class_names != test_class_names:
        raise ValueError("train/ and test/ must contain exactly the same class-folder names")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"Training images: {len(train_paths)}; test images: {len(test_paths)}")

    vocabulary_path = args.models / "vocabulary.pkl"
    svm_path = args.models / "svm_model.pkl"
    sift = make_sift(args.sift_features)

    if args.reuse_vocabulary:
        vocabulary = load_vocabulary(vocabulary_path, args.clusters)
        print(f"Reusing visual vocabulary from {vocabulary_path}")
    else:
        rng = np.random.default_rng(args.seed)
        descriptors = sample_descriptors(train_paths, sift, args.vocab_descriptors, rng)
        print(f"Clustering {len(descriptors)} sampled descriptors into {args.clusters} words")
        vocabulary = learn_vocabulary(descriptors, args.clusters, args.seed)
        save_vocabulary(vocabulary, vocabulary_path)
        print(f"Saved visual vocabulary to {vocabulary_path}")

    # Every image becomes one normalized k-dimensional visual-word histogram.
    train_histograms = build_histograms(
        train_paths, sift, vocabulary, args.assignment_batch_size, args.normalization
    )
    test_histograms = build_histograms(
        test_paths, sift, vocabulary, args.assignment_batch_size, args.normalization
    )
    np.savez_compressed(
        args.output / "train_histograms.npz", histograms=train_histograms, labels=train_labels
    )
    np.savez_compressed(
        args.output / "test_histograms.npz", histograms=test_histograms, labels=test_labels
    )

    # Train/evaluate 1-NN using only the normalized training histograms.
    nearest_neighbor = train_nearest_neighbor(train_histograms, train_labels)
    nn_predictions = nearest_neighbor.predict(test_histograms)
    nn_accuracy = evaluate_and_save(
        "nearest_neighbor", test_labels, nn_predictions, class_names, test_paths, args.output
    )

    # Train/evaluate the 15 binary one-vs-rest linear SVMs and save their model.
    svm = train_linear_svm(train_histograms, train_labels, args.svm_c, args.seed)
    save_svm(svm, class_names, svm_path)
    svm_predictions = predict_svm(svm, test_histograms)
    svm_accuracy = evaluate_and_save(
        "linear_svm", test_labels, svm_predictions, class_names, test_paths, args.output
    )

    # This small metadata file records all important settings needed to reproduce results.
    metrics = {
        "nearest_neighbor_accuracy": nn_accuracy,
        "linear_svm_accuracy": svm_accuracy,
        "clusters": args.clusters,
        "vocabulary_descriptors_requested": args.vocab_descriptors,
        "vocabulary_shape": list(vocabulary.shape),
        "histogram_normalization": args.normalization,
        "train_images": len(train_paths),
        "test_images": len(test_paths),
        "class_names": class_names,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"All results were written to {args.output}")


if __name__ == "__main__":
    main()
