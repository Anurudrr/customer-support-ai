# Failure Analysis

This analysis summarizes five realistic failure modes observed or expected in the customer support intent classifier and escalation pipeline.

## 1. Angry Billing Messages Classified as General Complaints

- **Example:** "This double charge is unacceptable, fix it today"
- **Prediction:** `complaint_angry`, confidence `0.64`, escalate `YES`
- **Ground truth:** `billing_issue`, escalate `YES`
- **Why it failed:** The message contains both a billing entity ("double charge") and strong anger language ("unacceptable", "fix it today"). The embedding neighbours were closer to angry complaint examples than billing examples because emotional tone dominated the short message.
- **Suggested fix:** Add a billing-entity priority rule before complaint classification. If terms such as `charge`, `refund`, `invoice`, `payment`, `card`, or `subscription` appear, keep the intent as `billing_issue` while still using anger signals for escalation.

## 2. Calm Refund Requests Not Escalated

- **Example:** "Can I get a refund for accidental purchase?"
- **Prediction:** `billing_issue`, confidence `0.82`, escalate `NO`
- **Ground truth:** `billing_issue`, escalate `YES`
- **Why it failed:** The escalation rules focus on anger, low confidence, feature requests, and unknown intent. A calm refund request can look safe for automated handling even though money-related customer actions often require auditability or human review.
- **Suggested fix:** Add a refund-specific escalation rule. Any billing message containing `refund`, `reverse`, `dispute`, or `unauthorized` should escalate or at least enter a human-review queue.

## 3. Technical Severity Underestimated

- **Example:** "App keeps crashing every time I open camera"
- **Prediction:** `technical_issue`, confidence `0.79`, escalate `NO`
- **Ground truth:** `technical_issue`, escalate `YES`
- **Why it failed:** The intent classifier correctly identifies the technical category, but the escalation module does not consistently treat repeated crashes as severe unless the wording is angry or confidence is low.
- **Suggested fix:** Add severity keywords and frequency patterns to escalation. Phrases such as `keeps crashing`, `completely broken`, `cannot open`, `down`, `blocked`, and `every time` should raise escalation priority.

## 4. Sarcasm Misread as Positive

- **Example:** "Great, another payment failure right when I need the app"
- **Prediction:** `positive`, confidence `0.58`, escalate `NO`
- **Ground truth:** `billing_issue` or `technical_issue`, escalate `YES`
- **Why it failed:** The word "Great" is treated like positive sentiment, but the rest of the sentence expresses frustration. The current system has no sarcasm, negation, or contrast detection, so mixed emotional signals are easy to misread.
- **Suggested fix:** Add a lightweight sarcasm and contrast layer using patterns such as `great, another`, `thanks for nothing`, `love when`, and positive words followed by negative events. Low-confidence sarcastic patterns should escalate.

## 5. Access-Management Requests Confused With Feature Requests

- **Example:** "Need an option to let another admin recover the workspace"
- **Prediction:** `feature_request`, confidence `0.61`, escalate `YES`
- **Ground truth:** `account_access`, escalate `YES`
- **Why it failed:** The phrase "Need an option" resembles feature-request examples, while "admin recover the workspace" describes an account-access problem. The nearest-neighbour model relies on overall semantic similarity and can miss the operational urgency of access recovery.
- **Suggested fix:** Add account-access keywords with higher weight, including `admin`, `workspace`, `recover`, `locked`, `2FA`, `verification`, and `reset`. Consider a two-stage classifier that first detects support domain entities before assigning final intent.

## Summary

Most failures come from overlapping signals rather than completely wrong language understanding. The model usually detects the broad support domain, but it struggles when:

- Tone and topic disagree, such as angry billing messages.
- Business risk is not expressed as anger, such as calm refund requests.
- Severity depends on product policy, such as repeated crashes.
- Sarcasm flips the apparent sentiment.
- Feature-style phrasing hides account-access urgency.

The highest-impact fixes are refund escalation, severity keywords for technical failures, and entity-aware intent correction for billing and account-access messages.
