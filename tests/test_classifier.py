"""
Tests for the IntentClassifier.
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intent_classifier import IntentClassifier
from src.preprocessing import preprocess

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fitted_classifier():
    """Return a small fitted classifier for testing."""
    texts = [
        # billing
        "i was charged twice for the same order",
        "where is my refund",
        "cancel my subscription and give me a refund",
        "i want my money back",
        "the billing is wrong",
        # technical
        "the app keeps crashing on my phone",
        "error every time i open the website",
        "feature is broken and not working",
        "i get a server error constantly",
        "the app freezes when i try to save",
        # account_access
        "i forgot my password and cant log in",
        "my account is locked please help",
        "i need to reset my password",
        "cant access my account",
        "i cant log in to my account",
        # complaint_angry
        "this is absolutely ridiculous and unacceptable",
        "i am furious about your service",
        "worst customer support ever",
        "i am extremely frustrated",
        "this is outrageous behavior",
        # complaint_normal
        "the app is a bit slow today",
        "the interface is a little confusing",
        "i'm not fully satisfied with the service",
        "could be better overall",
        "i don't like the new design",
        # feature_request
        "can you please add dark mode",
        "it would be great to have offline support",
        "please add a bulk export feature",
        "i would love calendar integration",
        "please add google drive support",
        # positive
        "love your service it is amazing",
        "great app the best tool i have used",
        "thank you so much you are wonderful",
        "perfect experience from start to finish",
        "you are absolutely amazing keep it up",
    ]
    labels = (
        ["billing_issue"] * 5
        + ["technical_issue"] * 5
        + ["account_access"] * 5
        + ["complaint_angry"] * 5
        + ["complaint_normal"] * 5
        + ["feature_request"] * 5
        + ["positive"] * 5
    )
    clf = IntentClassifier(k=3)
    clf.fit([preprocess(t) for t in texts], labels, show_progress=False)
    return clf


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestIntentClassifier:

    def test_predict_returns_string(self, fitted_classifier):
        intent = fitted_classifier.predict(preprocess("my refund hasn't arrived"))
        assert isinstance(intent, str)
        assert intent in [
            "billing_issue", "technical_issue", "account_access",
            "complaint_angry", "complaint_normal", "feature_request", "positive",
        ]

    def test_predict_with_confidence_keys(self, fitted_classifier):
        result = fitted_classifier.predict_with_confidence(preprocess("app keeps crashing"))
        assert "intent" in result
        assert "confidence" in result
        assert "top_k" in result

    def test_confidence_range(self, fitted_classifier):
        result = fitted_classifier.predict_with_confidence(preprocess("i want a refund please"))
        assert 0.0 <= result["confidence"] <= 1.0

    def test_empty_message(self, fitted_classifier):
        result = fitted_classifier.predict_with_confidence("")
        assert result["intent"] == "unknown"
        assert result["confidence"] == 0.0

    def test_billing_message(self, fitted_classifier):
        result = fitted_classifier.predict_with_confidence(
            preprocess("i was charged twice and want a refund")
        )
        assert result["intent"] == "billing_issue"

    def test_technical_message(self, fitted_classifier):
        result = fitted_classifier.predict_with_confidence(
            preprocess("the app keeps crashing every time i open it")
        )
        assert result["intent"] == "technical_issue"

    def test_account_message(self, fitted_classifier):
        result = fitted_classifier.predict_with_confidence(
            preprocess("i forgot my password and cant log in")
        )
        assert result["intent"] == "account_access"

    def test_positive_message(self, fitted_classifier):
        result = fitted_classifier.predict_with_confidence(
            preprocess("love your service it is absolutely amazing")
        )
        assert result["intent"] == "positive"

    def test_unfitted_raises(self):
        clf = IntentClassifier()
        with pytest.raises(RuntimeError):
            clf.predict_with_confidence("hello")

    def test_mismatched_lengths_raises(self):
        clf = IntentClassifier()
        with pytest.raises(ValueError):
            clf.fit(["text1", "text2"], ["label1"])

    def test_save_and_load(self, fitted_classifier, tmp_path):
        save_path = fitted_classifier.save(tmp_path)
        assert save_path.exists()
        loaded = IntentClassifier.load(tmp_path)
        result1 = fitted_classifier.predict(preprocess("i need a refund"))
        result2 = loaded.predict(preprocess("i need a refund"))
        assert result1 == result2
