"""
Tests for the escalation decision engine.
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.escalation_decider import should_escalate, EscalationResult
from src.anger_detector import is_angry, is_strongly_angry, is_urgent


class TestAngerDetector:

    def test_angry_message(self):
        flag, reasons = is_angry("This is absolutely ridiculous! I am furious!!!")
        assert flag is True
        assert len(reasons) > 0

    def test_not_angry(self):
        flag, _ = is_angry("I forgot my password")
        assert flag is False

    def test_all_caps(self):
        flag, reasons = is_angry("THIS IS UNACCEPTABLE")
        assert flag is True

    def test_exclamation_marks(self):
        flag, _ = is_angry("fix this now!!")
        assert flag is True

    def test_strongly_angry(self):
        flag, _ = is_strongly_angry("This is absolutely OUTRAGEOUS!!!!")
        assert flag is True

    def test_not_strongly_angry(self):
        flag, _ = is_strongly_angry("the app is a bit slow")
        assert flag is False

    def test_urgency_detected(self):
        flag, reasons = is_urgent("I need this fixed urgently ASAP")
        assert flag is True

    def test_no_urgency(self):
        flag, _ = is_urgent("Love your service, great job!")
        assert flag is False


class TestEscalationDecider:

    def test_angry_billing_escalates(self):
        result = should_escalate(
            raw_message="This is RIDICULOUS! I was charged twice!!!",
            intent="billing_issue",
            confidence=0.90,
        )
        assert result.should_escalate is True
        assert "Rule 1" in result.triggered_rule

    def test_low_confidence_escalates(self):
        result = should_escalate(
            raw_message="some weird query that doesn't match anything",
            intent="billing_issue",
            confidence=0.35,
        )
        assert result.should_escalate is True
        assert "Rule 2" in result.triggered_rule

    def test_feature_request_escalates(self):
        result = should_escalate(
            raw_message="Can you please add a dark mode feature?",
            intent="feature_request",
            confidence=0.85,
        )
        assert result.should_escalate is True
        assert "Rule 3" in result.triggered_rule

    def test_strongly_angry_escalates(self):
        result = should_escalate(
            raw_message="I am FURIOUS and this is completely OUTRAGEOUS!!!",
            intent="complaint_normal",
            confidence=0.75,
        )
        assert result.should_escalate is True

    def test_normal_technical_does_not_escalate(self):
        result = should_escalate(
            raw_message="the app is a bit slow today",
            intent="technical_issue",
            confidence=0.80,
        )
        assert result.should_escalate is False

    def test_positive_message_does_not_escalate(self):
        result = should_escalate(
            raw_message="Love your service, keep up the great work!",
            intent="positive",
            confidence=0.90,
        )
        assert result.should_escalate is False

    def test_result_has_reason(self):
        result = should_escalate(
            raw_message="app crashes",
            intent="technical_issue",
            confidence=0.82,
        )
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_escalation_result_to_dict(self):
        result = should_escalate(
            raw_message="cant log in",
            intent="account_access",
            confidence=0.78,
        )
        d = result.to_dict()
        assert "should_escalate" in d
        assert "reason" in d
        assert "triggered_rule" in d

    def test_unknown_intent_escalates(self):
        result = should_escalate(
            raw_message="I need help with something unusual",
            intent="unknown",
            confidence=0.80,
        )
        assert result.should_escalate is True
