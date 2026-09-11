"""
End-to-end pipeline integration tests.
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intent_classifier import IntentClassifier
from src.pipeline import CustomerSupportPipeline
from src.preprocessing import preprocess


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    """Build a pipeline with a small in-memory classifier."""
    texts = [
        # billing
        "i was charged twice for the same order",
        "where is my refund",
        "cancel my subscription and give me a refund",
        "i want my money back now",
        "the billing amount is wrong",
        # technical
        "the app keeps crashing on my phone",
        "error every time i open the website",
        "the feature is broken and not working",
        "i get a 500 server error constantly",
        "the app freezes when i try to save",
        # account_access
        "i forgot my password and cannot log in",
        "my account is locked please help me",
        "i need to reset my password urgently",
        "i cannot access my account",
        "i am locked out of my account",
        # complaint_angry
        "this is absolutely ridiculous and unacceptable",
        "i am furious about your terrible service",
        "worst customer support i have ever experienced",
        "i am extremely frustrated with your company",
        "this is outrageous behavior from your team",
        # complaint_normal
        "the app is a bit slow today",
        "the interface is a little confusing to use",
        "i am not fully satisfied with the service",
        "it could be better overall",
        "i do not like the new design",
        # feature_request
        "can you please add a dark mode option",
        "it would be great to have offline support",
        "please add a bulk export feature",
        "i would love to have calendar integration",
        "please add google drive support",
        # positive
        "love your service it is absolutely amazing",
        "great app the best tool i have ever used",
        "thank you so much you are wonderful people",
        "perfect experience from start to finish",
        "you are absolutely amazing keep up the great work",
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
    return CustomerSupportPipeline(classifier=clf)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestPipeline:

    EXPECTED_KEYS = {
        "message", "processed_message", "intent", "confidence",
        "should_escalate", "escalation_reason", "triggered_rule",
        "anger_signals", "reply", "processing_time_ms",
    }

    def test_result_has_all_keys(self, pipeline):
        result = pipeline.handle("my refund hasn't arrived")
        assert self.EXPECTED_KEYS.issubset(result.keys())

    def test_original_message_preserved(self, pipeline):
        msg = "  My Refund Hasn't Arrived!!!  "
        result = pipeline.handle(msg)
        assert result["message"] == msg

    def test_billing_intent(self, pipeline):
        result = pipeline.handle("i was charged twice and want a refund back")
        assert result["intent"] == "billing_issue"

    def test_technical_intent(self, pipeline):
        result = pipeline.handle("the app keeps crashing every time i open it")
        assert result["intent"] == "technical_issue"

    def test_account_intent(self, pipeline):
        result = pipeline.handle("i forgot my password and can't log in")
        assert result["intent"] == "account_access"

    def test_positive_intent(self, pipeline):
        result = pipeline.handle("love your service it is amazing")
        assert result["intent"] == "positive"

    def test_feature_request_escalates(self, pipeline):
        result = pipeline.handle("can you please add a dark mode option")
        assert result["should_escalate"] is True

    def test_angry_billing_escalates(self, pipeline):
        result = pipeline.handle("This is RIDICULOUS! I was charged twice!!!")
        assert result["should_escalate"] is True

    def test_confidence_in_range(self, pipeline):
        result = pipeline.handle("i need help with my account")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_reply_is_non_empty(self, pipeline):
        result = pipeline.handle("the app crashes when i try to save")
        assert isinstance(result["reply"], str)
        assert len(result["reply"].strip()) > 0

    def test_escalation_reason_set_when_escalating(self, pipeline):
        result = pipeline.handle("can you please add dark mode?")
        if result["should_escalate"]:
            assert len(result["escalation_reason"]) > 0

    def test_escalation_reason_empty_when_not_escalating(self, pipeline):
        result = pipeline.handle("love your service, thank you!")
        if not result["should_escalate"]:
            assert result["escalation_reason"] == ""

    def test_stats_tracking(self, pipeline):
        # Reset by creating a fresh pipeline
        clf = pipeline._classifier
        fresh = CustomerSupportPipeline(classifier=clf)
        fresh.handle("test message one")
        fresh.handle("test message two")
        stats = fresh.stats()
        assert stats["total_messages"] == 2
        assert stats["escalated"] + stats["bot_handled"] == 2

    def test_processing_time_is_positive(self, pipeline):
        result = pipeline.handle("the app is slow")
        assert result["processing_time_ms"] >= 0

    def test_empty_message_handled(self, pipeline):
        result = pipeline.handle("")
        assert "intent" in result
        assert "reply" in result

    @pytest.mark.parametrize("message,expected_intent", [
        ("my refund still hasn't arrived!", "billing_issue"),
        ("app keeps crashing every time i open it", "technical_issue"),
        ("i forgot my password and can't login", "account_access"),
        ("love your service thank you!", "positive"),
        ("could you please add dark mode?", "feature_request"),
    ])
    def test_required_demo_messages(self, pipeline, message, expected_intent):
        result = pipeline.handle(message)
        assert result["intent"] == expected_intent, (
            f"Expected {expected_intent}, got {result['intent']} "
            f"(confidence={result['confidence']:.3f})"
        )
