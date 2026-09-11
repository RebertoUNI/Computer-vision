"""Read the dataset and extract/samples SIFT descriptors from its images."""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Only files with one of these extensions are treated as dataset images.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(directory: Path) -> list[Path]:
    return [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()]
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
