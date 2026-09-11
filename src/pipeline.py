"""
Complete customer support pipeline.

This module wires together all components:
  preprocessing → intent classification → escalation → reply generation

Entry point: handle_customer_message()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from src.config import CLASSIFIER_SAVE_DIR
from src.preprocessing import preprocess
from src.intent_classifier import IntentClassifier
from src.escalation_decider import should_escalate
from src.reply_generator import ReplyGenerator

logger = logging.getLogger(__name__)


class CustomerSupportPipeline:
    """
    End-to-end customer support pipeline.

    Parameters
    ----------
    classifier : IntentClassifier, optional
        A pre-fitted classifier.  If None, the pipeline attempts to load a
        saved one from disk.  Build one first with build_pipeline().
    """

    def __init__(self, classifier: Optional[IntentClassifier] = None) -> None:
        self._reply_gen = ReplyGenerator()
        self._classifier = classifier
        self._call_count: int = 0
        self._escalation_count: int = 0

    def _get_classifier(self) -> IntentClassifier:
        if self._classifier is None:
            logger.info("Loading classifier from disk …")
            self._classifier = IntentClassifier.load()
        return self._classifier

    def handle(self, message: str) -> dict:
        """
        Process a single customer message through the full pipeline.

        Parameters
        ----------
        message : str
            Raw customer message (not pre-processed).

        Returns
        -------
        dict with keys:
            message           – original message
            processed_message – cleaned message
            intent            – predicted intent
            confidence        – cosine-similarity-based confidence score
            should_escalate   – bool
            escalation_reason – string reason or empty string
            triggered_rule    – which escalation rule fired
            reply             – generated response
            processing_time_ms – wall-clock time in milliseconds
        """
        t0 = time.perf_counter()
        self._call_count += 1

        # 1. Preprocess
        processed = preprocess(message)

        # 2. Classify
        clf = self._get_classifier()
        classification = clf.predict_with_confidence(processed)
        intent = classification["intent"]
        confidence = classification["confidence"]

        # 3. Escalation decision (uses raw message for anger signals)
        esc_result = should_escalate(message, intent, confidence)

        # 4. Reply generation
        if esc_result.should_escalate:
            self._escalation_count += 1
            reply = self._reply_gen.generate_escalation(
                reason=esc_result.reason,
                message=message,
                triggered_rule=esc_result.triggered_rule,
            )
        else:
            reply = self._reply_gen.generate(intent=intent, message=message)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "message": message,
            "processed_message": processed,
            "intent": intent,
            "confidence": round(confidence, 4),
            "should_escalate": esc_result.should_escalate,
            "escalation_reason": esc_result.reason if esc_result.should_escalate else "",
            "triggered_rule": esc_result.triggered_rule,
            "anger_signals": esc_result.anger_signals,
            "reply": reply,
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def stats(self) -> dict:
        """Return aggregate statistics for this pipeline session."""
        escalation_rate = (
            self._escalation_count / self._call_count
            if self._call_count > 0
            else 0.0
        )
        return {
            "total_messages": self._call_count,
            "escalated": self._escalation_count,
            "bot_handled": self._call_count - self._escalation_count,
            "escalation_rate": round(escalation_rate, 4),
        }


# ── Convenience function ──────────────────────────────────────────────────────

def build_pipeline(train_csv: Optional[Path] = None) -> CustomerSupportPipeline:
    """
    Build and return a ready-to-use pipeline by training the classifier.

    Parameters
    ----------
    train_csv : Path, optional
        Override path to the training CSV.  Defaults to config.TRAIN_PATH.

    Returns
    -------
    CustomerSupportPipeline
    """
    import pandas as pd
    from src.config import TRAIN_PATH
    from src.preprocessing import preprocess_batch

    path = Path(train_csv or TRAIN_PATH)
    logger.info("Building pipeline from training data: %s", path)

    df = pd.read_csv(path).dropna(subset=["message"])
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"].str.len() > 0]

    texts = preprocess_batch(df["message"].tolist())
    labels = df["intent"].tolist()

    clf = IntentClassifier()
    clf.fit(texts, labels, show_progress=True)
    clf.save()

    return CustomerSupportPipeline(classifier=clf)


def load_pipeline() -> CustomerSupportPipeline:
    """Load a previously saved pipeline."""
    clf = IntentClassifier.load()
    return CustomerSupportPipeline(classifier=clf)
