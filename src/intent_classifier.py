"""
Embedding-based intent classifier.

Architecture
-----------
1. A SentenceTransformer encodes all training examples once at fit() time.
2. At inference time the query is encoded and cosine similarities are computed
   against every training example.
3. The top-K nearest neighbours vote for an intent.  The winning intent's
   mean similarity score is returned as the confidence.

Note on 'confidence'
--------------------
The score returned is the *mean cosine similarity* of the top-K neighbours
that voted for the winning intent.  Cosine similarity ∈ [-1, 1] and is NOT
a calibrated probability.  It is used as a confidence-*like* signal to trigger
escalation when it falls below a configurable threshold.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    EMBEDDING_MODEL_NAME,
    KNN_K,
    CONFIDENCE_THRESHOLD,
    CLASSIFIER_SAVE_DIR,
    RANDOM_SEED,
    INTENTS,
)

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Nearest-neighbour intent classifier using sentence embeddings.

    Parameters
    ----------
    model_name : str
        HuggingFace / sentence-transformers model identifier.
    k : int
        Number of nearest neighbours used for majority voting.
    confidence_threshold : float
        Similarity score below which the prediction is considered uncertain.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        k: int = KNN_K,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.k = k
        self.confidence_threshold = confidence_threshold

        self._model: Optional[SentenceTransformer] = None
        self._train_embeddings: Optional[np.ndarray] = None
        self._train_labels: Optional[list[str]] = None
        self._train_texts: Optional[list[str]] = None
        self._is_fitted: bool = False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        self._ensure_model_loaded()
        assert self._model is not None
        return self._model.encode(
            texts,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # unit vectors → cosine sim = dot product
            batch_size=64,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, texts: list[str], labels: list[str], show_progress: bool = True) -> "IntentClassifier":
        """
        Encode training texts and store their embeddings.

        Parameters
        ----------
        texts : list[str]
            Pre-processed training sentences.
        labels : list[str]
            Corresponding intent labels.
        show_progress : bool
            Display a tqdm progress bar during encoding.

        Returns
        -------
        self
        """
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length.")

        logger.info("Fitting classifier on %d examples …", len(texts))
        self._train_texts = list(texts)
        self._train_labels = list(labels)
        self._train_embeddings = self._encode(texts, show_progress=show_progress)
        self._is_fitted = True
        logger.info("Classifier fitted.  Embedding shape: %s", self._train_embeddings.shape)
        return self

    def predict_with_confidence(self, text: str) -> dict:
        """
        Predict intent and return a confidence score.

        Parameters
        ----------
        text : str
            Pre-processed customer message.

        Returns
        -------
        dict with keys:
            intent       – predicted intent label
            confidence   – mean cosine similarity of winning neighbours (0..1)
            top_k        – list of (text, label, similarity) for top-K neighbours
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_with_confidence().")

        if not text or not text.strip():
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "top_k": [],
            }

        query_emb = self._encode([text])  # shape (1, dim)
        assert self._train_embeddings is not None
        sims = cosine_similarity(query_emb, self._train_embeddings)[0]  # shape (n,)

        # Top-K neighbours
        k = min(self.k, len(sims))
        top_k_idx = np.argsort(sims)[::-1][:k]

        assert self._train_labels is not None
        assert self._train_texts is not None

        top_k_results = [
            {
                "text": self._train_texts[i],
                "label": self._train_labels[i],
                "similarity": float(sims[i]),
            }
            for i in top_k_idx
        ]

        # Majority vote weighted by similarity
        vote_scores: dict[str, float] = {}
        vote_counts: dict[str, int] = {}
        for item in top_k_results:
            lbl = item["label"]
            vote_scores[lbl] = vote_scores.get(lbl, 0.0) + item["similarity"]
            vote_counts[lbl] = vote_counts.get(lbl, 0) + 1

        predicted_intent = max(vote_scores, key=vote_scores.__getitem__)

        # Confidence = mean similarity of the winning intent's neighbours
        winning_sims = [
            item["similarity"]
            for item in top_k_results
            if item["label"] == predicted_intent
        ]
        confidence = float(np.mean(winning_sims))

        return {
            "intent": predicted_intent,
            "confidence": confidence,
            "top_k": top_k_results,
        }

    def predict(self, text: str) -> str:
        """Return only the predicted intent label."""
        return self.predict_with_confidence(text)["intent"]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Run predict_with_confidence for a list of texts."""
        return [self.predict_with_confidence(t) for t in texts]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, save_dir: Optional[Path] = None) -> Path:
        """
        Save classifier state (embeddings + labels) to *save_dir*.

        The SentenceTransformer model itself is NOT saved here because it
        can be re-downloaded.  Only the training embeddings and metadata are
        persisted as a pickle.
        """
        save_dir = Path(save_dir or CLASSIFIER_SAVE_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "model_name": self.model_name,
            "k": self.k,
            "confidence_threshold": self.confidence_threshold,
            "train_embeddings": self._train_embeddings,
            "train_labels": self._train_labels,
            "train_texts": self._train_texts,
        }
        path = save_dir / "classifier_state.pkl"
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Classifier state saved to %s", path)
        return path

    @classmethod
    def load(cls, save_dir: Optional[Path] = None) -> "IntentClassifier":
        """Load a previously saved classifier state."""
        save_dir = Path(save_dir or CLASSIFIER_SAVE_DIR)
        path = save_dir / "classifier_state.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No saved classifier found at {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        obj = cls(
            model_name=state["model_name"],
            k=state["k"],
            confidence_threshold=state["confidence_threshold"],
        )
        obj._train_embeddings = state["train_embeddings"]
        obj._train_labels = state["train_labels"]
        obj._train_texts = state["train_texts"]
        obj._is_fitted = True
        logger.info("Classifier state loaded from %s", path)
        return obj

    def is_low_confidence(self, confidence: float) -> bool:
        """Return True if *confidence* is below the configured threshold."""
        return confidence < self.confidence_threshold
