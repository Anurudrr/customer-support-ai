"""
Batch evaluation script.

Usage:
    python -m evaluation.evaluate

Reads data/evaluation_set.csv, runs predictions, saves results.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    EVAL_PATH, EVAL_RESULTS_DIR, LOG_FORMAT, LOG_LEVEL, CLASSIFIER_SAVE_DIR,
)
from src.preprocessing import preprocess
from src.pipeline import CustomerSupportPipeline, build_pipeline, load_pipeline
from evaluation.metrics import (
    intent_metrics, escalation_metrics, compute_confidence_stats,
    save_metrics, print_intent_metrics, print_escalation_metrics,
)
from evaluation.confusion_matrix import plot_confusion_matrix, plot_confidence_distribution
from evaluation.failure_analysis import run_failure_analysis
from evaluation.judge import judge_reply

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def run_evaluation(eval_path: Path = EVAL_PATH) -> dict:
    """
    Run the full evaluation pipeline on *eval_path*.

    Returns a metrics dict.
    """
    # 1. Load data
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Evaluation set not found at {eval_path}. "
            "Run: python data/generate_data.py"
        )

    df = pd.read_csv(eval_path)
    df = df.dropna(subset=["message"])
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"].str.len() > 0]
    logger.info("Loaded %d evaluation examples.", len(df))

    # 2. Load pipeline
    if CLASSIFIER_SAVE_DIR.exists() and any(CLASSIFIER_SAVE_DIR.iterdir()):
        logger.info("Loading saved classifier …")
        try:
            pipeline = load_pipeline()
        except Exception as e:
            logger.warning("Load failed (%s). Rebuilding …", e)
            pipeline = build_pipeline()
    else:
        logger.info("No saved classifier — building from training data …")
        pipeline = build_pipeline()

    # 3. Run predictions
    print(f"\nRunning predictions on {len(df)} examples …")
    results = []
    for _, row in df.iterrows():
        result = pipeline.handle(row["message"])
        results.append(result)

    # 4. Build results DataFrame
    res_df = pd.DataFrame({
        "message": df["message"].values,
        "true_intent": df["intent"].values,
        "predicted_intent": [r["intent"] for r in results],
        "confidence": [r["confidence"] for r in results],
        "should_escalate": [r["should_escalate"] for r in results],
        "expected_escalate": df["should_escalate"].values if "should_escalate" in df.columns else [False] * len(df),
        "reply": [r["reply"] for r in results],
        "escalation_reason": [r["escalation_reason"] for r in results],
    })

    # 5. Compute metrics
    intent_m = intent_metrics(res_df["true_intent"].tolist(), res_df["predicted_intent"].tolist())
    esc_m = escalation_metrics(
        res_df["expected_escalate"].tolist(),
        res_df["should_escalate"].tolist(),
    )
    conf_stats = compute_confidence_stats(res_df["confidence"].tolist())

    # 6. Reply quality via heuristic judge
    judge_scores = [
        judge_reply(
            row["message"], row["predicted_intent"], row["reply"], row["should_escalate"]
        ).score
        for _, row in res_df.iterrows()
    ]
    avg_judge_score = round(sum(judge_scores) / len(judge_scores), 4)

    # 7. Compile full metrics
    all_metrics = {
        "intent": intent_m,
        "escalation": esc_m,
        "confidence": conf_stats,
        "reply_quality_avg": avg_judge_score,
        "n_examples": len(df),
    }

    print_intent_metrics(intent_m)
    print_escalation_metrics(esc_m)
    print(f"\n  Avg Reply Quality (heuristic judge): {avg_judge_score:.4f}")
    print(f"\n  Confidence Stats: mean={conf_stats['mean']:.4f}  std={conf_stats['std']:.4f}  "
          f"min={conf_stats['min']:.4f}  max={conf_stats['max']:.4f}")

    # 8. Save metrics
    save_metrics(all_metrics, "metrics.json")

    # Save full results CSV
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = EVAL_RESULTS_DIR / "evaluation_results.csv"
    res_df.to_csv(results_path, index=False)
    print(f"\n  Results saved to {results_path}")

    # 9. Confusion matrix
    print("\n  Generating confusion matrix...")
    cm_path = plot_confusion_matrix(
        res_df["true_intent"].tolist(),
        res_df["predicted_intent"].tolist(),
    )
    print(f"  Confusion matrix saved to {cm_path}")

    # 10. Confidence distribution
    correct_mask = (res_df["true_intent"] == res_df["predicted_intent"]).tolist()
    cd_path = plot_confidence_distribution(
        res_df["confidence"].tolist(),
        correct_mask=correct_mask,
    )
    print(f"  Confidence distribution saved to {cd_path}")

    # 11. Failure analysis
    print("\n  Running failure analysis...")
    run_failure_analysis(res_df)
    print("  Failure analysis report saved to reports/failure_analysis.md")

    print("\n[OK] Evaluation complete.\n")
    return all_metrics


if __name__ == "__main__":
    run_evaluation()
