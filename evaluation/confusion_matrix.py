"""
Confusion matrix visualisation for intent classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import INTENTS, FIGURES_DIR

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: Optional[list[str]] = None,
    title: str = "Intent Classification — Confusion Matrix",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    """
    Generate and save a confusion matrix heatmap.

    Parameters
    ----------
    y_true : list[str]
        Ground-truth intent labels.
    y_pred : list[str]
        Predicted intent labels.
    labels : list[str], optional
        Intent labels to use as axes.  Defaults to INTENTS from config.
    title : str
        Plot title.
    save_path : Path, optional
        Where to save the figure.  Defaults to outputs/figures/confusion_matrix.png.
    show : bool
        Call plt.show() after saving.

    Returns
    -------
    Path  – path to the saved figure.
    """
    labels = labels or INTENTS
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalise to percentages for readability
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, data, fmt, title_suffix in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["(Counts)", "(Row-Normalised)"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(f"{title}\n{title_suffix}", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = save_path or FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Confusion matrix saved to %s", save_path)

    if show:
        plt.show()

    plt.close(fig)
    return save_path


def plot_confidence_distribution(
    confidences: list[float],
    labels: Optional[list[str]] = None,
    correct_mask: Optional[list[bool]] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    """Plot histogram of confidence scores, optionally split by correct/incorrect."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if correct_mask is not None:
        correct = [c for c, m in zip(confidences, correct_mask) if m]
        wrong = [c for c, m in zip(confidences, correct_mask) if not m]
        ax.hist(correct, bins=30, alpha=0.6, label="Correct", color="steelblue")
        ax.hist(wrong, bins=30, alpha=0.6, label="Incorrect", color="salmon")
        ax.legend()
    else:
        ax.hist(confidences, bins=30, color="steelblue", alpha=0.8)

    ax.axvline(x=0.60, color="red", linestyle="--", linewidth=1.5, label="Threshold (0.60)")
    ax.set_xlabel("Confidence Score (Mean Cosine Similarity)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Confidence Score Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = save_path or FIGURES_DIR / "confidence_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Confidence distribution saved to %s", save_path)

    if show:
        plt.show()

    plt.close(fig)
    return save_path
