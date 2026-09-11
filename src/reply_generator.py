"""
Template-based reply generator.

Multiple templates per intent are stored so responses feel less repetitive.
A deterministic selection strategy picks a template based on a hash of the
input message (so the same message always gets the same reply, ensuring
reproducibility).

Escalation replies are generated separately and can vary based on the
escalation reason.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Normal intent templates ───────────────────────────────────────────────────
INTENT_TEMPLATES: dict[str, list[str]] = {
    "billing_issue": [
        "I understand your concern about the billing. Let me look into this right away and connect you with our billing team if needed.",
        "I'm sorry to hear about the billing problem. Could you confirm your account email so I can pull up your details?",
        "I can see why that's frustrating. Our billing team will review your account and get back to you within 24 hours.",
        "Thank you for flagging this. I'll make sure the billing issue gets resolved as quickly as possible.",
        "I apologise for the inconvenience. Please allow me to investigate your billing concern immediately.",
    ],
    "technical_issue": [
        "Sorry you're having trouble! Please try restarting the app, clearing the cache, and reinstalling if necessary. Let me know if that helps.",
        "I understand how frustrating technical issues can be. Could you tell me which device and operating system you're using?",
        "Thanks for reporting this. Our tech team is aware of some intermittent issues — please try again in a few minutes.",
        "Let's get this sorted! First, try force-closing the app and reopening it. If the issue persists, I can escalate to our engineering team.",
        "I apologise for the disruption. Please try clearing your browser cache or app data and let me know if the problem continues.",
    ],
    "account_access": [
        "No problem! You can reset your password using the 'Forgot Password' option on the login page. Check your email inbox for the reset link.",
        "I can help you regain access. Please use the password reset link at our login page, and check your spam folder if you don't see the email.",
        "Account access issues are usually resolved quickly. Try the password reset flow — if your account is locked, reply and I'll escalate to our security team.",
        "Happy to help! Visit our login page and click 'Forgot Password'. If your account is locked, our team can unlock it for you within a few hours.",
        "Let's get you back in! Use the password reset option on the login screen. If you still can't access your account, I'll pass this to our account team.",
    ],
    "complaint_angry": [
        "I'm sincerely sorry about your experience. I'll make sure this gets the attention it deserves right away.",
        "I completely understand your frustration, and I apologise. Let me escalate this immediately so a senior team member can assist you.",
        "Your experience should never have happened, and I'm truly sorry. I'm prioritising your case right now.",
        "I hear you, and I'm very sorry. This is absolutely not the experience we want for our customers. I'll act on this immediately.",
        "I sincerely apologise for putting you through this. I'm taking your feedback very seriously and will escalate this to our management team.",
    ],
    "complaint_normal": [
        "Thank you for sharing your feedback — it's really valuable to us. We'll use it to improve.",
        "I'm sorry to hear you're not fully satisfied. Could you tell me more so we can make things right?",
        "Thanks for letting us know. Your feedback helps us improve. Is there anything specific you'd like us to address?",
        "We appreciate your honest feedback. Our team reviews every suggestion to make the experience better.",
        "I understand, and I'm sorry for the inconvenience. We're constantly working to improve — thank you for taking the time to share this.",
    ],
    "feature_request": [
        "That's a great suggestion! I'll pass your feedback directly to our product team for review.",
        "Thanks for the idea — I love hearing feature suggestions! I'll log this for our product team.",
        "I'll make sure your suggestion reaches the right people on our product team. Thank you!",
        "We really appreciate feature suggestions from our users. I've noted yours and will forward it to the team.",
        "Great input! Your suggestion will be shared with our product team as part of our regular feedback review.",
    ],
    "positive": [
        "Thank you so much! We're really glad you're enjoying the service. 😊",
        "That's wonderful to hear! We work hard to make a great experience, and your kind words mean a lot.",
        "Thank you! We're thrilled to have you as a customer. Don't hesitate to reach out if you ever need anything.",
        "Wow, thank you! Feedback like yours keeps our team motivated. We appreciate you!",
        "You've made our day! Thank you for the kind words. We're here for you whenever you need us.",
    ],
}

# ── Escalation reply templates ────────────────────────────────────────────────
ESCALATION_TEMPLATES: dict[str, list[str]] = {
    "default": [
        "Thank you for reaching out. I've forwarded your case to our support team. A specialist will review it and follow up as soon as possible.",
        "I've escalated your request to our team, and someone will be in touch shortly to assist you further.",
        "Your case has been passed on to a specialist who will reach out to you as soon as possible. Thank you for your patience.",
    ],
    "billing": [
        "I've escalated your billing concern to our dedicated billing team. They will review your account and contact you within 24 hours.",
        "Your billing issue has been prioritised and forwarded to our billing specialists. Expect a follow-up within 24 hours.",
    ],
    "angry": [
        "I'm very sorry for the poor experience. I've escalated your case to a senior support agent who will follow up with you as a priority.",
        "Your frustration is completely understandable. I've immediately escalated this to our senior team to ensure you receive the attention you deserve.",
    ],
    "feature_request": [
        "Thank you for your suggestion! I've passed it on to our product team. While we can't guarantee every feature will be implemented, all suggestions are reviewed.",
        "Great idea! Your feature request has been logged and forwarded to our product team for review.",
    ],
    "low_confidence": [
        "I want to make sure I give you the best help possible. I've forwarded your message to our support team who will be better placed to assist you.",
        "I'd like to make sure your question gets the most accurate answer. I've passed this to our specialist team who will follow up shortly.",
    ],
}

# ── Fallback ───────────────────────────────────────────────────────────────────
FALLBACK_REPLY = (
    "Thank you for contacting support. I've received your message and will "
    "get back to you as soon as possible."
)


def _pick_template(templates: list[str], seed_text: str) -> str:
    """
    Deterministically pick a template from *templates* using a hash of *seed_text*.

    This ensures the same message always receives the same reply, which is
    important for reproducible evaluation.
    """
    idx = int(hashlib.md5(seed_text.encode()).hexdigest(), 16) % len(templates)
    return templates[idx]


class ReplyGenerator:
    """
    Template-based reply generator.

    Usage
    -----
    >>> gen = ReplyGenerator()
    >>> reply = gen.generate(intent="billing_issue", message="I want a refund")
    >>> escalation_reply = gen.generate_escalation(reason="billing", message="...")
    """

    def generate(self, intent: str, message: str) -> str:
        """
        Generate a bot reply for *intent*.

        Parameters
        ----------
        intent : str
            Predicted intent label.
        message : str
            Original customer message (used as seed for deterministic selection).

        Returns
        -------
        str
            A non-empty reply string.
        """
        templates = INTENT_TEMPLATES.get(intent)
        if not templates:
            logger.warning("No templates found for intent '%s'; using fallback.", intent)
            return FALLBACK_REPLY
        return _pick_template(templates, message)

    def generate_escalation(
        self, reason: str, message: str, triggered_rule: str = ""
    ) -> str:
        """
        Generate an escalation reply appropriate for the escalation *reason*.

        Parameters
        ----------
        reason : str
            Human-readable escalation reason string.
        message : str
            Original customer message (used as seed for deterministic selection).
        triggered_rule : str
            The triggered rule identifier, used to pick the template category.

        Returns
        -------
        str
            Escalation reply string.
        """
        # Map rule to template category
        category = "default"
        reason_lower = reason.lower()
        rule_lower = triggered_rule.lower()

        if "billing" in reason_lower or "billing" in rule_lower:
            category = "billing"
        elif "angry" in rule_lower or "anger" in reason_lower:
            category = "angry"
        elif "feature" in reason_lower or "feature" in rule_lower:
            category = "feature_request"
        elif "confidence" in reason_lower or "confidence" in rule_lower:
            category = "low_confidence"

        templates = ESCALATION_TEMPLATES.get(category, ESCALATION_TEMPLATES["default"])
        return _pick_template(templates, message)

    def all_intents_have_templates(self) -> bool:
        """Return True if every known intent has at least one template."""
        from src.config import INTENTS
        return all(intent in INTENT_TEMPLATES for intent in INTENTS)
