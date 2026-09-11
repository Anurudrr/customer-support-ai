"""
Failure analysis — identify and explain misclassified examples.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)

# Categories of failure (used for explanation heuristics)
AMBIGUOUS_PAIRS = {
    ("billing_issue", "complaint_angry"),
    ("complaint_angry", "billing_issue"),
    ("complaint_normal", "complaint_angry"),
    ("complaint_angry", "complaint_normal"),
    ("account_access", "technical_issue"),
    ("technical_issue", "account_access"),
    ("complaint_normal", "feature_request"),
    ("feature_request", "complaint_normal"),
}


def _explain_failure(row: pd.Series) -> str:
    """Generate a heuristic explanation for why a prediction failed."""
    true_i = row.get("true_intent", "")
    pred_i = row.get("predicted_intent", "")
    msg = str(row.get("message", "")).lower()
    confidence = row.get("confidence", 1.0)
    reasons = []

    if confidence < 0.65:
        reasons.append("low confidence — the message is ambiguous or atypical")

    if (true_i, pred_i) in AMBIGUOUS_PAIRS or (pred_i, true_i) in AMBIGUOUS_PAIRS:
        reasons.append(f"overlapping intents — '{true_i}' and '{pred_i}' can look similar")

    anger_words = {"ridiculous", "horrible", "terrible", "worst", "disgusting", "awful"}
    if any(w in msg for w in anger_words) and true_i != "complaint_angry":
        reasons.append("emotional language caused misclassification as complaint_angry")

    if "?" in msg and true_i in ("billing_issue", "account_access"):
        reasons.append("question phrasing may have shifted the prediction")

    if len(msg.split()) < 5:
        reasons.append("very short message — insufficient context for accurate classification")

    sarcasm_hints = ["great job", "well done", "amazing work", "wonderful"]
    if any(h in msg for h in sarcasm_hints) and true_i != "positive":
        reasons.append("possible sarcasm — positive phrasing with negative intent")

    if not reasons:
        reasons.append("insufficient training examples for this message pattern")

    return "; ".join(reasons)


def run_failure_analysis(
    results_df: pd.DataFrame,
    n_examples: int = 10,
    save_report: bool = True,
) -> pd.DataFrame:
    """
    Identify and analyse prediction failures.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns: message, true_intent, predicted_intent,
        confidence, should_escalate, expected_escalate, reply.
    n_examples : int
        Number of failure examples to highlight.
    save_report : bool
        Write the failure analysis report to reports/failure_analysis.md.

    Returns
    -------
    pd.DataFrame of failure cases with an 'explanation' column.
    """
    failures = results_df[
        results_df["true_intent"] != results_df["predicted_intent"]
    ].copy()

    logger.info(
        "Found %d failures out of %d examples (%.1f%%)",
        len(failures),
        len(results_df),
        100 * len(failures) / max(len(results_df), 1),
    )

    if failures.empty:
        logger.info("No failures found — perfect classification!")
        return failures

    failures["explanation"] = failures.apply(_explain_failure, axis=1)
    failures_sorted = failures.sort_values("confidence").reset_index(drop=True)
    top_failures = failures_sorted.head(n_examples)

    if save_report:
        _write_report(top_failures, len(failures), len(results_df))

    return failures_sorted


def _write_report(top_failures: pd.DataFrame, n_failures: int, n_total: int) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "failure_analysis.md"

    lines = [
        "# Failure Analysis Report",
        "",
        f"**Total examples evaluated:** {n_total}",
        f"**Failures:** {n_failures} ({100 * n_failures / max(n_total, 1):.1f}%)",
        "",
        "## Top Failure Cases",
        "",
    ]

    for i, row in top_failures.iterrows():
        lines += [
            f"### Case {int(i) + 1}",
            "",
            f"**Message:** `{row.get('message', '')}`",
            "",
            f"- **True intent:** `{row.get('true_intent', 'N/A')}`",
            f"- **Predicted intent:** `{row.get('predicted_intent', 'N/A')}`",
            f"- **Confidence:** `{row.get('confidence', 0):.4f}`",
            f"- **Escalated:** `{row.get('should_escalate', 'N/A')}`",
            f"- **Expected escalation:** `{row.get('expected_escalate', 'N/A')}`",
            f"- **Generated reply:** _{row.get('reply', 'N/A')}_",
            f"- **Likely reason:** {row.get('explanation', 'N/A')}",
            "",
        ]

    lines += [
        "## Common Failure Patterns",
        "",
        "| Pattern | Description |",
        "|---------|-------------|",
        "| Overlapping intents | complaint_angry vs billing_issue — angry billing messages can fire either intent |",
        "| Short messages | Very short messages (<5 words) lack context for accurate classification |",
        "| Sarcasm | Positive phrasing used sarcastically is hard to detect without deeper NLP |",
        "| Low confidence | Messages unlike any training example trigger the escalation safety net |",
        "| Mixed signals | A message with anger + account issue might be classified as complaint_angry |",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Failure analysis report saved to %s", path)
