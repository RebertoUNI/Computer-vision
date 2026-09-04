"""Read the dataset and extract/samples SIFT descriptors from its images."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


# Only files with one of these extensions are treated as dataset images.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(class_directory: Path) -> list[Path]:
    """Return the image files of one class folder in a reproducible order."""
    return sorted(
        path
        for path in class_directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_split(split_directory: Path) -> tuple[list[Path], np.ndarray, list[str]]:
    """Read paths, integer labels, and class names from train/ or test/.

    Each immediate subfolder is a class. Alphabetical sorting fixes the same
    label-to-class mapping for all operating systems and subsequent runs.
    """
    if not split_directory.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_directory}")

    class_names = sorted(path.name for path in split_directory.iterdir() if path.is_dir())
    if not class_names:
        raise ValueError(f"No class folders found in {split_directory}")

    image_paths: list[Path] = []
    labels: list[int] = []
    for label, class_name in enumerate(class_names):
        paths = list_images(split_directory / class_name)
        if not paths:
            raise ValueError(f"No readable image files in {split_directory / class_name}")
        image_paths.extend(paths)
        labels.extend([label] * len(paths))

    return image_paths, np.asarray(labels, dtype=np.int32), class_names


def make_sift(max_features: int):
    """Create the SIFT detector/descriptor extractor"""
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "This OpenCV build has no SIFT. Install opencv-contrib-python, then retry."
        )
    # nfeatures=0 means that OpenCV does not impose a maximum descriptor count.
    return cv2.SIFT_create(nfeatures=max_features) # ordered by strenght


def extract_sift(image_path: Path, sift) -> np.ndarray:
    """Return a float32 (N, 128) array of SIFT descriptors for one image."""
    # SIFT expects one channel, hence loading the image directly as grayscale.
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    # detectAndCompute first detects keypoints, then computes a descriptor per point.
    _keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        # Return a correctly shaped array when an image has no detectable keypoints.
        return np.empty((0, 128), dtype=np.float32)
    return descriptors.astype(np.float32, copy=False)


def sample_descriptors(
    image_paths: Sequence[Path], sift, target_count: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample approximately equally from all training images for vocabulary learning.

    A per-image quota prevents textured images from dominating the visual words.
    The returned amount can only be below target_count when the full training set
    contains fewer SIFT descriptors than requested.
    """
    if target_count < 1:
        raise ValueError("--vocab-descriptors must be positive")
    quota = int(np.ceil(target_count / len(image_paths)))
    sampled: list[np.ndarray] = []

    for index, image_path in enumerate(image_paths, start=1):
        descriptors = extract_sift(image_path, sift)
        if len(descriptors) > quota:
            # Randomly retain the per-image quota without duplicate descriptors.
            descriptors = descriptors[rng.choice(len(descriptors), size=quota, replace=False)]
        if len(descriptors):
            sampled.append(descriptors)
        if index % 100 == 0 or index == len(image_paths):
            print(f"Vocabulary descriptors: processed {index}/{len(image_paths)} images")

    if not sampled:
        raise ValueError("SIFT found no descriptors in the training images")
    all_descriptors = np.vstack(sampled).astype(np.float32, copy=False)
    if len(all_descriptors) > target_count:
        # Remove the few excess rows caused by rounding the per-image quota upward.
        all_descriptors = all_descriptors[
            rng.choice(len(all_descriptors), size=target_count, replace=False)
        ]
    return all_descriptors
