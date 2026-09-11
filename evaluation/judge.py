"""
LLM-as-judge placeholder for evaluating reply quality.

In an academic project without an external LLM API, this module provides
a rule-based heuristic judge as a stand-in.  If an OpenAI / Gemini API key
is later configured, it can be extended to use a real LLM judge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result from the reply quality judge."""
    score: float          # 0.0 – 1.0
    label: str            # "good" | "acceptable" | "poor"
    reasoning: str


def judge_reply(
    message: str,
    intent: str,
    reply: str,
    should_escalate: bool,
) -> JudgeResult:
    """
    Heuristic judge for reply quality.

    Rules:
    - Empty reply                 → poor
    - Reply mentions the intent topic → good signal
    - Escalation reply is non-empty   → acceptable
    - Reply very short (<10 words)    → score penalty
    """
    if not reply or not reply.strip():
        return JudgeResult(score=0.0, label="poor", reasoning="Empty reply")

    words = reply.split()
    score = 0.7  # baseline

    # Intent-topic signal
    intent_keywords = {
        "billing_issue": ["billing", "refund", "charge", "payment", "invoice"],
        "technical_issue": ["technical", "app", "error", "crash", "fix", "restart"],
        "account_access": ["account", "password", "reset", "access", "login"],
        "complaint_angry": ["sorry", "apologi", "understand", "frustrat", "escalate"],
        "complaint_normal": ["feedback", "improve", "understand", "sorry"],
        "feature_request": ["suggestion", "product", "feature", "team", "forward"],
        "positive": ["thank", "glad", "happy", "great"],
    }
    keywords = intent_keywords.get(intent, [])
    reply_lower = reply.lower()
    if any(kw in reply_lower for kw in keywords):
        score += 0.15

    # Penalise very short replies
    if len(words) < 10:
        score -= 0.1

    # Escalation replies should acknowledge the issue
    if should_escalate and ("team" in reply_lower or "specialist" in reply_lower or "escalat" in reply_lower):
        score += 0.1

    score = round(min(1.0, max(0.0, score)), 2)

    if score >= 0.8:
        label = "good"
    elif score >= 0.5:
        label = "acceptable"
    else:
        label = "poor"

    reasoning = (
        f"Score based on: length={len(words)} words, "
        f"intent_keyword_match={any(kw in reply_lower for kw in keywords)}, "
        f"escalation_language={'team' in reply_lower or 'specialist' in reply_lower}"
    )

    return JudgeResult(score=score, label=label, reasoning=reasoning)
