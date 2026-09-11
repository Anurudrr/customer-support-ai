"""
Evaluation metrics for intent classification and escalation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.config import INTENTS, EVAL_RESULTS_DIR

logger = logging.getLogger(__name__)


def intent_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """
    Compute intent classification metrics.

    Returns
    -------
    dict with keys: accuracy, macro_f1, macro_precision, macro_recall,
                    per_intent (classification_report as dict)
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "per_intent": report,
    }


def escalation_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    """
    Compute binary escalation metrics.
    """
    y_true_int = [int(v) for v in y_true]
    y_pred_int = [int(v) for v in y_pred]

    accuracy = accuracy_score(y_true_int, y_pred_int)
    precision = precision_score(y_true_int, y_pred_int, zero_division=0)
    recall = recall_score(y_true_int, y_pred_int, zero_division=0)
    f1 = f1_score(y_true_int, y_pred_int, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true_int, y_pred_int, labels=[0, 1]).ravel()

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def compute_confidence_stats(confidences: list[float]) -> dict:
    """Basic statistics for the confidence score distribution."""
    arr = np.array(confidences)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
    }


def save_metrics(metrics: dict, filename: str = "metrics.json") -> Path:
    """Save metrics dict to JSON in the evaluation results directory."""
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", path)
    return path


def print_intent_metrics(metrics: dict) -> None:
    """Print a formatted summary of intent metrics."""
    print("\n" + "=" * 50)
    print("  INTENT CLASSIFICATION METRICS")
    print("=" * 50)
    print(f"  Accuracy         : {metrics['accuracy']:.4f}")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision  : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall     : {metrics['macro_recall']:.4f}")
    print("\n  Per-Intent F1:")
    for intent in INTENTS:
        per = metrics.get("per_intent", {}).get(intent, {})
        f1 = per.get("f1-score", "N/A")
        support = per.get("support", "N/A")
        if isinstance(f1, float):
            print(f"    {intent:<20}: F1={f1:.4f}  support={support}")
    print("=" * 50)


def print_escalation_metrics(metrics: dict) -> None:
    """Print a formatted summary of escalation metrics."""
    print("\n" + "=" * 50)
    print("  ESCALATION METRICS")
    print("=" * 50)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  TP={metrics['true_positives']}  TN={metrics['true_negatives']}  "
          f"FP={metrics['false_positives']}  FN={metrics['false_negatives']}")
    print("=" * 50)
