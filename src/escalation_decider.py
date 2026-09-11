"""
Rule-based escalation decision engine.

Rules are applied in priority order and are mutually non-exclusive — the
first matching rule wins.  All rules are documented below so they can be
easily modified by editing this file.

Escalation rules
----------------
1. Angry customer + billing issue          → escalate (billing team)
2. Low classifier confidence               → escalate (uncertain intent)
3. Intent == feature_request               → escalate (product team)
4. Strongly angry customer (any intent)    → escalate (senior agent)
5. Unsupported / unknown intent            → escalate (general support)

Otherwise the bot handles the request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.anger_detector import is_angry, is_strongly_angry, is_urgent
from src.config import CONFIDENCE_THRESHOLD, ANGRY_ESCALATE_INTENTS

logger = logging.getLogger(__name__)


@dataclass
class EscalationResult:
    """Result returned by the escalation decision engine."""
    should_escalate: bool
    reason: str
    triggered_rule: str  # human-readable rule identifier
    anger_signals: list[str] = field(default_factory=list)
    urgency_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "should_escalate": self.should_escalate,
            "reason": self.reason,
            "triggered_rule": self.triggered_rule,
            "anger_signals": self.anger_signals,
            "urgency_signals": self.urgency_signals,
        }


def should_escalate(
    raw_message: str,
    intent: str,
    confidence: float,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> EscalationResult:
    """
    Apply escalation rules and return an :class:`EscalationResult`.

    Parameters
    ----------
    raw_message : str
        The original (un-preprocessed) customer message.
    intent : str
        Predicted intent from the classifier.
    confidence : float
        Confidence score from the classifier (cosine similarity, 0..1).
    confidence_threshold : float
        Predictions below this value are considered uncertain.

    Returns
    -------
    EscalationResult
    """
    angry, anger_reasons = is_angry(raw_message)
    strongly_angry, strong_anger_reasons = is_strongly_angry(raw_message)
    urgent, urgency_reasons = is_urgent(raw_message)

    # ── Rule 1: Angry + billing issue ────────────────────────────────────────
    if angry and intent in ANGRY_ESCALATE_INTENTS:
        return EscalationResult(
            should_escalate=True,
            reason=f"Angry customer with {intent.replace('_', ' ')}",
            triggered_rule="Rule 1: angry + sensitive intent",
            anger_signals=anger_reasons,
            urgency_signals=urgency_reasons,
        )

    # ── Rule 2: Low confidence ────────────────────────────────────────────────
    if confidence < confidence_threshold:
        return EscalationResult(
            should_escalate=True,
            reason=f"Low classifier confidence ({confidence:.2f} < {confidence_threshold})",
            triggered_rule="Rule 2: low confidence",
            anger_signals=anger_reasons,
            urgency_signals=urgency_reasons,
        )

    # ── Rule 3: Feature request → product team ───────────────────────────────
    if intent == "feature_request":
        return EscalationResult(
            should_escalate=True,
            reason="Feature requests are forwarded to the product team",
            triggered_rule="Rule 3: feature request",
            anger_signals=anger_reasons,
            urgency_signals=urgency_reasons,
        )

    # ── Rule 4: Strongly angry customer ──────────────────────────────────────
    if strongly_angry:
        return EscalationResult(
            should_escalate=True,
            reason="Customer is expressing strong anger — senior agent needed",
            triggered_rule="Rule 4: strongly angry",
            anger_signals=strong_anger_reasons,
            urgency_signals=urgency_reasons,
        )

    # ── Rule 5: Unknown intent ────────────────────────────────────────────────
    if intent in ("unknown", "other"):
        return EscalationResult(
            should_escalate=True,
            reason="Message is outside the bot's supported topics",
            triggered_rule="Rule 5: unknown intent",
            anger_signals=anger_reasons,
            urgency_signals=urgency_reasons,
        )

    # ── No rule triggered — bot handles ──────────────────────────────────────
    return EscalationResult(
        should_escalate=False,
        reason="Bot can handle this request",
        triggered_rule="No escalation rule triggered",
        anger_signals=anger_reasons,
        urgency_signals=urgency_reasons,
    )
