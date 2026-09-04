"""Turn local SIFT descriptors into normalized Bag-of-Words histograms."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin

from src.extract_features import extract_sift


def histogram_from_descriptors(
    descriptors: np.ndarray, vocabulary: np.ndarray, batch_size: int, normalization: str
) -> np.ndarray:
    """Assign descriptors to nearest visual words and return one k-bin histogram."""
    histogram = np.zeros(len(vocabulary), dtype=np.float32)
    # Process descriptor blocks so distance computation stays memory bounded.
    for start in range(0, len(descriptors), batch_size):
        block = descriptors[start:start + batch_size]
        # Each returned index identifies the nearest k-means centroid (visual word).
        word_indices = pairwise_distances_argmin(block, vocabulary, metric="euclidean")
        histogram += np.bincount(word_indices, minlength=len(vocabulary)).astype(np.float32)

    # Normalize the frequency vector, retaining an all-zero histogram for blank images.
    denominator = histogram.sum() if normalization == "l1" else np.linalg.norm(histogram)
    if denominator > 0:
        histogram /= denominator
    return histogram


def build_histograms(
    image_paths: Sequence[Path], sift, vocabulary: np.ndarray,
    batch_size: int, normalization: str
) -> np.ndarray:
    """Extract descriptors and create one normalized BoW row for every image."""
    if batch_size < 1:
        raise ValueError("--assignment-batch-size must be positive")
    histograms = np.empty((len(image_paths), len(vocabulary)), dtype=np.float32)
    for index, image_path in enumerate(image_paths, start=1):
        descriptors = extract_sift(image_path, sift)
        histograms[index - 1] = histogram_from_descriptors(
            descriptors, vocabulary, batch_size, normalization
        )
        if index % 100 == 0 or index == len(image_paths):
            print(f"Histograms: processed {index}/{len(image_paths)} images")
    return histograms
