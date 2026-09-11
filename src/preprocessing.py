"""
Text preprocessing utilities for customer support messages.

Design decisions:
- Lowercase is applied for embedding-based models (transformers are case-aware
  but lowercasing reduces vocabulary variance without hurting semantic meaning).
- We deliberately preserve emotional/sentiment words like 'RIDICULOUS' by
  detecting ALL-CAPS patterns before lowercasing (for anger detection).
- URLs are replaced with a placeholder token, not removed, so the model knows
  a link was present.
- We do NOT aggressively strip stop-words or punctuation because the downstream
  SentenceTransformer handles subword tokenisation internally.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Patterns ─────────────────────────────────────────────────────────────────
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_MULTI_EXCLAIM_RE = re.compile(r"!{2,}")
_MULTI_SPACE_RE = re.compile(r"\s+")
_CAPS_WORD_RE = re.compile(r"\b[A-Z]{3,}\b")  # 3+ uppercase letters


def detect_caps_ratio(text: str) -> float:
    """Return the fraction of alphabetic characters that are uppercase.

    This is used by the anger detector *before* lowercasing.
    """
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    return sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)


def detect_all_caps_words(text: str) -> list[str]:
    """Return a list of ALL-CAPS words (length >= 3) found in *text*."""
    return _CAPS_WORD_RE.findall(text)


def detect_exclamation_count(text: str) -> int:
    """Return the total number of '!' characters in *text*."""
    return text.count("!")


def replace_urls(text: str, placeholder: str = "[URL]") -> str:
    """Replace HTTP/HTTPS and bare www URLs with *placeholder*."""
    return _URL_RE.sub(placeholder, text)


def replace_mentions(text: str, placeholder: str = "[USER]") -> str:
    """Replace Twitter @mentions with *placeholder*."""
    return _MENTION_RE.sub(placeholder, text)


def expand_hashtags(text: str) -> str:
    """Convert #hashtag → 'hashtag' (preserves the word without the symbol)."""
    return _HASHTAG_RE.sub(r"\1", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def normalize_exclamations(text: str) -> str:
    """Reduce repeated '!!!' to a single '!' to avoid noise for embeddings."""
    return _MULTI_EXCLAIM_RE.sub("!", text)


def preprocess(
    text: Optional[str],
    *,
    lowercase: bool = True,
    handle_urls: bool = True,
    handle_mentions: bool = True,
    handle_hashtags: bool = True,
    normalize_excl: bool = True,
) -> str:
    """
    Full preprocessing pipeline for a single customer message.

    Parameters
    ----------
    text : str or None
        Raw customer message.
    lowercase : bool
        Convert to lowercase (default True).
    handle_urls : bool
        Replace URLs with [URL] token (default True).
    handle_mentions : bool
        Replace @mentions with [USER] token (default True).
    handle_hashtags : bool
        Expand #hashtags to plain words (default True).
    normalize_excl : bool
        Collapse repeated '!!!' to a single '!' (default True).

    Returns
    -------
    str
        Cleaned text ready for embedding.

    Notes
    -----
    Call this function on raw text BEFORE passing it to IntentClassifier.
    The anger-detection signals (caps ratio, exclamation count) must be
    extracted from the RAW text before calling this function if needed.
    """
    if text is None or (isinstance(text, float)):
        logger.debug("Received null/nan message; returning empty string.")
        return ""

    text = str(text)

    if handle_urls:
        text = replace_urls(text)
    if handle_mentions:
        text = replace_mentions(text)
    if handle_hashtags:
        text = expand_hashtags(text)
    if normalize_excl:
        text = normalize_exclamations(text)
    if lowercase:
        text = text.lower()

    text = normalize_whitespace(text)
    return text


def preprocess_batch(texts: list[Optional[str]], **kwargs) -> list[str]:
    """Apply :func:`preprocess` to a list of texts."""
    return [preprocess(t, **kwargs) for t in texts]
