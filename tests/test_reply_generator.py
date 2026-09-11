"""
Tests for the ReplyGenerator.
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reply_generator import ReplyGenerator
from src.config import INTENTS


class TestReplyGenerator:

    @pytest.fixture
    def gen(self):
        return ReplyGenerator()

    def test_all_intents_have_templates(self, gen):
        assert gen.all_intents_have_templates()

    @pytest.mark.parametrize("intent", INTENTS)
    def test_reply_is_non_empty(self, gen, intent):
        reply = gen.generate(intent=intent, message="test message")
        assert isinstance(reply, str)
        assert len(reply.strip()) > 0

    def test_billing_reply(self, gen):
        reply = gen.generate(intent="billing_issue", message="my refund hasn't arrived")
        assert len(reply) > 10

    def test_technical_reply(self, gen):
        reply = gen.generate(intent="technical_issue", message="app keeps crashing")
        assert len(reply) > 10

    def test_account_reply(self, gen):
        reply = gen.generate(intent="account_access", message="forgot my password")
        assert len(reply) > 10

    def test_positive_reply(self, gen):
        reply = gen.generate(intent="positive", message="love your service")
        assert "thank" in reply.lower() or "glad" in reply.lower() or "great" in reply.lower()

    def test_unknown_intent_fallback(self, gen):
        reply = gen.generate(intent="completely_unknown_intent", message="test")
        assert len(reply) > 0

    def test_escalation_reply_non_empty(self, gen):
        reply = gen.generate_escalation(
            reason="Low classifier confidence",
            message="some weird message",
            triggered_rule="Rule 2: low confidence",
        )
        assert isinstance(reply, str)
        assert len(reply.strip()) > 0

    def test_escalation_billing_reply(self, gen):
        reply = gen.generate_escalation(
            reason="Angry customer with billing issue",
            message="I was charged twice!!!",
            triggered_rule="Rule 1: angry + sensitive intent",
        )
        assert "billing" in reply.lower() or "team" in reply.lower()

    def test_deterministic_reply(self, gen):
        """Same message always produces the same reply."""
        msg = "my refund hasn't arrived yet"
        reply1 = gen.generate(intent="billing_issue", message=msg)
        reply2 = gen.generate(intent="billing_issue", message=msg)
        assert reply1 == reply2

    def test_escalation_reply_categories(self, gen):
        """All escalation categories return non-empty strings."""
        categories = [
            ("billing issue escalated", "Rule 1: angry + sensitive intent"),
            ("Strongly angry customer", "Rule 4: strongly angry"),
            ("Feature requests are forwarded", "Rule 3: feature request"),
            ("Low confidence", "Rule 2: low confidence"),
            ("Outside supported topics", "Rule 5: unknown intent"),
        ]
        for reason, rule in categories:
            reply = gen.generate_escalation(reason=reason, message="test", triggered_rule=rule)
            assert len(reply.strip()) > 0
