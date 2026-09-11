"""
Anger and urgency detection for customer support messages.

Kept deliberately simple and transparent so that the escalation system
remains explainable.  A keyword/phrase list plus structural signals
(ALL CAPS, repeated exclamation marks) are enough for a student-level system.
"""

import re
import logging
from src.preprocessing import (
    detect_caps_ratio,
    detect_all_caps_words,
    detect_exclamation_count,
)

logger = logging.getLogger(__name__)

# ── Anger vocabulary ──────────────────────────────────────────────────────────
ANGER_WORDS: set[str] = {
    "ridiculous", "frustrated", "furious", "terrible", "awful",
    "unacceptable", "disgusted", "angry", "rage", "worst",
    "horrible", "outrageous", "appalling", "atrocious", "disgusting",
    "infuriating", "maddening", "pathetic", "shameful", "scandalous",
    "absurd", "useless", "incompetent", "lazy", "stupid",
    "idiotic", "rubbish", "garbage", "trash", "scam",
    "fraud", "liar", "cheating", "rip-off", "ripoff",
    "never again", "hate", "despise", "fed up", "enough is enough",
    "sick and tired", "last straw", "lost my patience",
}

STRONG_ANGER_WORDS: set[str] = {
    "furious", "rage", "disgusted", "outrageous", "infuriating",
    "appalling", "atrocious", "incompetent", "scam", "fraud",
    "never again", "worst", "ridiculous", "unacceptable",
}

URGENCY_WORDS: set[str] = {
    "urgent", "asap", "immediately", "right now", "emergency",
    "critical", "crucial", "important", "deadline", "escalate",
    "manager", "supervisor", "legal", "lawsuit", "lawyer",
    "refund now", "fix this now", "need help now",
}

_WORD_BOUNDARY_RE = re.compile(r"\b")


def _contains_phrase(text_lower: str, phrases: set[str]) -> set[str]:
    """Return the subset of *phrases* that appear in *text_lower*."""
    found = set()
    for phrase in phrases:
        if phrase in text_lower:
            found.add(phrase)
    return found


def is_angry(text: str, threshold: float = 0.35) -> tuple[bool, list[str]]:
    """
    Determine whether a message expresses anger.

    Parameters
    ----------
    text : str
        Raw (un-preprocessed) customer message.
    threshold : float
        Minimum uppercase-letter ratio that triggers the CAPS signal.

    Returns
    -------
    (is_angry, reasons) : (bool, list[str])
        Boolean flag and a human-readable list of triggered signals.
    """
    reasons: list[str] = []
    text_lower = text.lower()

    # 1. Anger keyword match
    matched_words = _contains_phrase(text_lower, ANGER_WORDS)
    if matched_words:
        reasons.append(f"anger keywords: {', '.join(sorted(matched_words))}")

    # 2. ALL-CAPS words
    caps_words = detect_all_caps_words(text)
    if caps_words:
        reasons.append(f"all-caps words: {', '.join(caps_words)}")

    # 3. High uppercase ratio
    caps_ratio = detect_caps_ratio(text)
    if caps_ratio > threshold:
        reasons.append(f"high caps ratio: {caps_ratio:.0%}")

    # 4. Multiple exclamation marks
    excl_count = detect_exclamation_count(text)
    if excl_count >= 2:
        reasons.append(f"{excl_count} exclamation marks")

    return bool(reasons), reasons


def is_strongly_angry(text: str) -> tuple[bool, list[str]]:
    """
    Return True when the message contains strong anger signals.

    'Strong anger' means at least one of:
    - A word from STRONG_ANGER_WORDS appears.
    - Caps ratio > 50 %.
    - 3+ exclamation marks.
    """
    reasons: list[str] = []
    text_lower = text.lower()

    matched = _contains_phrase(text_lower, STRONG_ANGER_WORDS)
    if matched:
        reasons.append(f"strong anger keywords: {', '.join(sorted(matched))}")

    if detect_caps_ratio(text) > 0.50:
        reasons.append("very high caps ratio (>50%)")

    if detect_exclamation_count(text) >= 3:
        reasons.append("3+ exclamation marks")

    return bool(reasons), reasons


def is_urgent(text: str) -> tuple[bool, list[str]]:
    """
    Return True if the message contains urgency signals.
    """
    reasons: list[str] = []
    text_lower = text.lower()
    matched = _contains_phrase(text_lower, URGENCY_WORDS)
    if matched:
        reasons.append(f"urgency keywords: {', '.join(sorted(matched))}")
    return bool(reasons), reasons
