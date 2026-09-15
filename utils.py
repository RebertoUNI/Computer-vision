"""Read the dataset and extract/samples SIFT descriptors from its images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

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

def plot_image(image, title=""):
    """Visualizza un'immagine in scala di grigi."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def plot_hist(idx, X_train, y_train):
    """Visualizza l'istogramma delle visual words per un campione."""
    hist = np.asarray(X_train[idx])
    label = str(y_train[idx]).strip().capitalize()  # label cleaning

    fig, ax = plt.subplots(figsize=(14, 6))

    # All blue bars
    ax.bar(range(len(hist)), hist, color='#4c9fd6',
           edgecolor='none', width=0.85)

    # Mean line
    media = hist.mean()
    ax.axhline(media, color='gray', linestyle='--', linewidth=1,
               label=f'Mean = {media:.4f}')

    # Readable ticks: 1 every N
    step = max(1, len(hist) // 20)
    ax.set_xticks(range(0, len(hist), step))
    ax.set_xticklabels(range(0, len(hist), step), fontsize=9)

    # Clean title (single string, no weird nesting)
    titolo = f'Visual Word Distribution — Class "{label}"'
    ax.set_title(titolo, fontsize=14, fontweight='bold', loc='left', pad=18)

    # Subtitle with useful info
    picco_bin = int(hist.argmax())
    sottotitolo = (f'Sample n. {idx} · {len(hist)} bins · '
                   f'peak = {hist.max():.4f} (bin {picco_bin})')
    ax.text(0, 1.01, sottotitolo, transform=ax.transAxes,
            fontsize=10, color='#555', va='bottom')

    ax.set_xlabel('Visual Word Index (bin)', fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)

    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    ax.legend(loc='upper right', frameon=False, fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, classes, title, cmap='Blues'):
    """Visualizza una matrice di confusione."""
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap=cmap, xticks_rotation='vertical', values_format='d')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()
    plt.show()
