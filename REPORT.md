# Customer Support AI Bot - Evaluation Report

## 1. Problem Framing

This project builds an AI customer support bot that classifies customer intent, decides whether a case should be escalated, and generates a reply from a controlled template library. The goal is not to replace human support agents. The goal is to triage common messages quickly, handle low-risk cases consistently, and surface high-risk messages that need human attention.

**Brand:** Multi-brand support system

The system is designed around short customer-support messages similar to Twitter/X replies, app-store complaints, chat snippets, and helpdesk tickets. These messages are usually brief, informal, and emotionally mixed. A user may say "refund me now", "app keeps crashing", or "thanks for fixing that bug" with very little context. The bot therefore needs to infer intent from limited text while avoiding unsafe automation for sensitive cases.

**Core tasks:**

- Classify the customer message into an intent category.
- Estimate model confidence using embedding-neighbour similarity.
- Detect anger or urgency signals.
- Decide whether the case should be escalated.
- Generate a consistent template-based response.

**Intent labels:**

- `billing_issue`: refunds, charges, invoices, subscriptions, payment failures
- `technical_issue`: crashes, bugs, errors, failed uploads, broken features
- `account_access`: login, password reset, verification, locked accounts
- `complaint_angry`: intense frustration, threats to leave, hostile language
- `complaint_normal`: mild dissatisfaction, product criticism, unclear frustration
- `feature_request`: product suggestions and new-feature requests
- `positive`: compliments, thanks, satisfaction

**What is good about the approach:**

- Embedding-based classification handles paraphrases better than plain keyword matching.
- Rule-based escalation is transparent and easy to audit.
- Template replies maintain consistency and reduce hallucination risk.
- The pipeline is modular, so each part can be improved independently.

**What we did not build:**

- Sarcasm detection
- Multi-turn conversation tracking
- Real customer satisfaction measurement
- Multilingual support
- Image or attachment understanding
- Production deployment monitoring

## 2. Evaluation Methodology

The evaluation uses a manually curated synthetic dataset saved at `data/evaluation_set.csv`. It contains 100 realistic Twitter-style customer support messages. The examples are synthetic, but they are written to resemble common customer language: short phrasing, direct complaints, account-access issues, billing problems, and feature requests.

The dataset has three columns:

- `message`: the customer message
- `intent`: the ground-truth intent label
- `should_escalate`: `YES` or `NO` depending on whether a human should handle the case

### 2.1 Dataset Size and Class Distribution

The evaluation set contains exactly 100 labelled rows. The class distribution is intentionally imbalanced:

```text
Intent              Examples   Share
billing_issue          40       40%
technical_issue        13       13%
account_access         12       12%
complaint_angry        10       10%
complaint_normal       10       10%
feature_request         8        8%
positive                7        7%
```

This imbalance matters. Billing examples make up 40% of the evaluation set, so a model that over-predicts billing can achieve a deceptively high accuracy. This mirrors real support environments where billing and account issues may be more frequent than compliments or feature suggestions, but it also means macro F1 is more informative than accuracy alone.

Escalation label distribution:

```text
Label   Examples   Share
YES        54       54%
NO         46       46%
```

### 2.2 Model Under Test

The main model uses `SentenceTransformer('all-MiniLM-L6-v2')` to convert messages into dense semantic embeddings. Intent classification is performed with K-nearest-neighbour matching over the embedding space using cosine similarity.

At inference time:

1. The message is normalized by the preprocessing module.
2. The normalized message is encoded into an embedding.
3. Cosine similarity is computed against stored training embeddings.
4. The nearest neighbours vote for the predicted intent.
5. The mean similarity of the winning neighbours is reported as confidence.
6. Anger and urgency signals are checked on the raw message.
7. The escalation module applies transparent rules.
8. The reply generator selects a deterministic template.

### 2.3 Baselines

Two baselines were used for comparison:

**Dummy majority baseline:** Always predicts `billing_issue` for intent. For escalation, it uses a simple majority-style prior that marks obvious billing, angry complaint, and feature-request cases as escalated. This baseline is intentionally weak but useful for showing the effect of class imbalance.

**Keyword matching baseline:** Uses direct keywords such as `refund`, `charge`, `login`, `password`, `crash`, `bug`, `feature`, `add`, `thanks`, and `love`. This baseline is stronger than the dummy baseline, but it fails on paraphrases, sarcasm, and mixed-intent messages.

### 2.4 Metrics

The evaluation reports:

- Intent accuracy
- Macro precision, recall, and F1
- Weighted F1
- Per-class precision, recall, and F1
- ASCII confusion matrix
- Escalation precision, recall, F1, and accuracy
- Confidence distribution by score range

Macro F1 is important because it gives each class equal weight. Weighted F1 is also included because it reflects the real distribution of the evaluation set. Escalation recall is especially important because missed escalations can create a poor support experience.

## 3. Results

### 3.1 Baseline Comparison

```text
Model                                  Intent Accuracy    Escalation F1
Dummy majority baseline                 40.0%              0.70
Keyword matching baseline               66.0%              0.74
Embedding KNN model                     78.0%              0.85
```

The embedding KNN model outperforms both baselines on intent accuracy and escalation F1. The gap over keyword matching is mainly due to semantic matching. For example, messages about "card declined", "payment screen timed out", and "subscription still renewed" can all map to billing even when they do not share the exact same words.

The dummy baseline still reaches 40% intent accuracy because billing is overrepresented. This is why the headline accuracy must be read carefully.

### 3.2 Intent Classification Metrics

```text
Overall accuracy: 78 / 100 = 78.0%
Macro precision: 75.1%
Macro recall:    75.7%
Macro F1:        75.1%
Weighted F1:     78.6%
```

Per-intent metrics:

```text
Intent              Precision   Recall   F1      Support
billing_issue       0.85        0.85     0.85    40
technical_issue     0.82        0.69     0.75    13
account_access      0.90        0.75     0.82    12
complaint_angry     0.64        0.70     0.67    10
complaint_normal    0.64        0.70     0.67    10
feature_request     0.67        0.75     0.71     8
positive            0.75        0.86     0.80     7
```

Billing performs best because it has the most examples and clear repeated patterns such as `refund`, `invoice`, `charge`, `card`, `payment`, and `subscription`. Account access also performs well because terms like `login`, `locked`, `verification`, and `password reset` are distinctive.

The weakest classes are `complaint_angry` and `complaint_normal`. This is expected because both classes contain negative sentiment, and the difference between mild dissatisfaction and true escalation-worthy anger is often subtle.

### 3.3 Confusion Matrix

Rows are true labels. Columns are predicted labels.

```text
                  Predicted
True Label        BILL  TECH  ACCT  ANGR  NORM  FEAT  POS   Row Total
billing_issue       34     1     0     3     1     0    1      40
technical_issue      2     9     1     0     1     0    0      13
account_access       1     1     9     0     0     1    0      12
complaint_angry      2     0     0     7     1     0    0      10
complaint_normal     1     0     0     1     7     1    0      10
feature_request      0     0     0     0     1     6    1       8
positive             0     0     0     0     0     1    6       7
Column Total        40    11    10    11    11     9    8     100
```

Correct predictions: 78

Incorrect predictions: 22

Most common confusion patterns:

- Angry billing messages sometimes become `complaint_angry`.
- Payment-screen technical failures sometimes become `billing_issue`.
- Mild complaints sometimes become `complaint_angry`.
- Account administration requests sometimes become `feature_request`.
- Polite feature requests can resemble `positive` messages.

### 3.4 Escalation Metrics

```text
True Positive  (TP): 47
False Positive (FP):  9
True Negative  (TN): 37
False Negative (FN):  7
Total:             100
```

Derived metrics:

```text
Positive labels:      TP + FN = 54
Negative labels:      TN + FP = 46
Predicted positive:   TP + FP = 56
Predicted negative:   TN + FN = 44

Precision: 47 / 56 = 0.839
Recall:    47 / 54 = 0.870
F1:        0.854
Accuracy:  84 / 100 = 84.0%
```

The escalation system performs better than pure intent classification because it uses additional business rules. Even if the model predicts the exact intent incorrectly, escalation can still be correct when anger, feature-request, refund, or low-confidence signals are present.

### 3.5 Confidence Distribution Plot Description

Confidence is based on cosine similarity between the input message and its nearest training examples. It is useful as a ranking signal, but it is not a calibrated probability.

```text
Score Range      Examples   Correct   Accuracy
0.30 - 0.49          12         5       41.7%
0.50 - 0.59          18        11       61.1%
0.60 - 0.74          31        25       80.6%
0.75 - 0.89          29        27       93.1%
0.90 - 1.00          10        10      100.0%
```

If plotted as a histogram, the distribution would be right-skewed. Most correct predictions sit between 0.65 and 0.90. Incorrect predictions cluster near the 0.45 to 0.65 range. A threshold line at 0.60 is useful for flagging uncertain cases, but the score should not be interpreted as "60% likely to be correct."

## 4. Detailed Failure Analysis

### Failure Mode 1: Angry Billing Messages Classified as General Complaints

- **Example:** "This double charge is unacceptable, fix it today"
- **Prediction:** `complaint_angry`, confidence `0.64`, escalate `YES`
- **Ground truth:** `billing_issue`, escalate `YES`
- **Root cause:** The message contains both a billing entity and strong anger language. The emotional tone pulls the embedding toward angry complaint examples.
- **Suggested fix:** Add a billing-entity priority rule. If terms like `charge`, `refund`, `invoice`, `payment`, or `subscription` are present, preserve the billing intent while still using anger to escalate.

### Failure Mode 2: Calm Refund Requests Not Escalated

- **Example:** "Can I get a refund for accidental purchase?"
- **Prediction:** `billing_issue`, confidence `0.82`, escalate `NO`
- **Ground truth:** `billing_issue`, escalate `YES`
- **Root cause:** Current escalation rules treat anger and low confidence as high-risk signals, but a calm refund request can still require human review.
- **Suggested fix:** Add a refund-specific escalation rule for messages containing `refund`, `reverse`, `dispute`, or `unauthorized`.

### Failure Mode 3: Technical Severity Underestimated

- **Example:** "App keeps crashing every time I open camera"
- **Prediction:** `technical_issue`, confidence `0.79`, escalate `NO`
- **Ground truth:** `technical_issue`, escalate `YES`
- **Root cause:** The model detects the technical intent correctly, but the escalation engine does not consistently identify repeated crashes as severe.
- **Suggested fix:** Add technical severity patterns such as `keeps crashing`, `cannot open`, `completely broken`, `down`, `blocked`, and `every time`.

### Failure Mode 4: Sarcasm Misread as Positive

- **Example:** "Great, another payment failure right when I need the app"
- **Prediction:** `positive`, confidence `0.58`, escalate `NO`
- **Ground truth:** `billing_issue` or `technical_issue`, escalate `YES`
- **Root cause:** Positive words like "great" can hide frustration when used sarcastically. The current pipeline has no sarcasm, negation, or contrast detector.
- **Suggested fix:** Add a sarcasm pattern layer for phrases like `great, another`, `thanks for nothing`, and positive words followed by negative events.

### Failure Mode 5: Access-Management Requests Confused With Feature Requests

- **Example:** "Need an option to let another admin recover the workspace"
- **Prediction:** `feature_request`, confidence `0.61`, escalate `YES`
- **Ground truth:** `account_access`, escalate `YES`
- **Root cause:** The phrase "Need an option" resembles feature-request examples, while the actual problem is account recovery and workspace access.
- **Suggested fix:** Add account-access entity weighting for `admin`, `workspace`, `recover`, `locked`, `2FA`, `verification`, and `reset`.

## 5. Top 5 Failure Modes

The detailed examples above can be summarized into five recurring failure categories:

### 1. Sarcasm Not Detected

Sarcasm is difficult because the surface sentiment can be positive while the actual meaning is negative.

- Example: "Oh great, another 2-hour wait!"
- Predicted: `positive`
- Should be: `complaint_angry`
- Fix: Add sentiment, negation, and sarcasm pattern detection.

### 2. Out-of-Domain Queries

The model is trained on support categories, so unrelated questions can be forced into the nearest known intent.

- Example: "Does your CEO donate to climate groups?"
- Predicted: `feature_request`
- Should be: escalate as out-of-domain
- Fix: Add an OOD detector using low confidence and semantic distance.

### 3. Missing Urgency Signals

Some urgent messages do not use angry language.

- Example: "App KEEPS CRASHING before my meeting"
- Predicted: `technical_issue`, no escalation
- Should be: `technical_issue`, escalation
- Fix: Add urgency and business-impact keywords.

### 4. Long Conversations Not Detected

The current system only sees one message at a time.

- Example: A fifth message in the same thread saying "still not fixed"
- Predicted: no escalation
- Should be: escalation because repeated unresolved contact needs human attention
- Fix: Track conversation count, elapsed time, and unresolved status.

### 5. Refund Without Anger Missed

Money-related requests can be important even when written politely.

- Example: "I need a refund for this accidental purchase"
- Predicted: `billing_issue`, no escalation
- Should be: escalation or human review
- Fix: Escalate refund and dispute keywords regardless of sentiment.

## 6. What's Misleading About 78%?

**Headline claim:** 78% intent accuracy

**More honest interpretation:** 78% on this synthetic 100-row evaluation set does not mean 78% production accuracy.

### 6.1 Training Data Was Synthetic

The training and evaluation data were generated or curated for demonstration. Real customer messages are messier. They include misspellings, slang, screenshots, missing context, mixed languages, abbreviations, repeated follow-ups, and emotional ambiguity.

Impact:

- Real-world accuracy would likely drop.
- Sarcasm and indirect complaints would be more common.
- Class boundaries would be less clean.
- Some messages would not belong to any supported intent.

### 6.2 Class Imbalance Bias

Billing issues are 40% of the evaluation set. This means the model can get many examples right by doing well on billing. A dummy model that always predicts `billing_issue` already gets 40% accuracy.

Impact:

- Overall accuracy may overstate performance.
- Rare classes such as `positive` and `feature_request` have less stable metrics.
- Macro F1 should be reported alongside accuracy.

### 6.3 No Multi-Turn Support

The system treats each message as independent. Real customer support is often multi-turn. A message like "still broken" cannot be understood without previous messages.

Impact:

- Repeated frustration is missed.
- The model cannot tell whether the customer has already tried troubleshooting.
- Escalation may be delayed for long unresolved threads.

### 6.4 No Conversation History

Conversation history is related to multi-turn support but deserves separate mention. The bot does not store customer state, past predictions, prior escalation decisions, previous replies, or open ticket status.

Impact:

- Responses can feel repetitive.
- The bot cannot recognize that a customer is contacting support for the same unresolved case.
- There is no memory of prior promises or agent actions.

### 6.5 Single-Policy Evaluation

The project is described as multi-brand, but the evaluation effectively uses one generic support policy. Different brands escalate different things. A bank, streaming service, marketplace, and SaaS product would not use identical rules.

Impact:

- The evaluation does not prove transfer across brands.
- Escalation rules may need brand-specific tuning.
- Templates may not match every brand voice.

### 6.6 No Production Deployment Evidence

The project is a prototype. It has not been tested under production traffic, monitoring, latency requirements, privacy constraints, or agent-feedback loops.

Impact:

- Runtime behavior under load is unknown.
- There is no live customer satisfaction measurement.
- There is no human-agent acceptance or rejection data.

### 6.7 Confidence Scores Are Not Calibrated

The confidence score is cosine similarity, not a probability. A confidence of 0.80 does not mean the model has an 80% chance of being correct.

Impact:

- Thresholds are heuristic.
- Calibration would be required before production use.
- Confidence should be used as a risk signal, not a guarantee.

**Bottom line:** 78% on synthetic data is a useful prototype signal, but it should not be presented as production readiness.

## 7. Limitations & Assumptions

### What Will Not Work Well

- Sarcasm, irony, and indirect complaints
- Multilingual or code-switched messages
- Images, screenshots, receipts, and attachments
- Messages requiring account-specific lookup
- Legal, safety, or compliance-sensitive cases
- Long multi-turn support threads
- Measuring customer satisfaction from classification accuracy alone

Correct classification does not guarantee customer satisfaction. A reply can be technically relevant but still unhelpful, too slow, too generic, or inappropriate for the customer's emotional state.

### What Works Reasonably Well

- English text messages
- Clear intent classification
- Direct customer language
- Common billing, login, technical, feature, complaint, and positive messages
- Template-based replies where consistency is more important than creativity
- Transparent escalation rules that can be inspected and modified

### Assumptions

- The customer message is text-only.
- The message is in English.
- The message belongs to one primary intent.
- The available intent labels are sufficient for the support domain.
- The escalation policy is generic and not brand-specific.
- Synthetic data is acceptable for MVP evaluation.

## 8. Next Steps

### Next Steps With One More Week

1. Collect a real Twitter or helpdesk sample of 100-200 examples.
2. Re-label the examples with at least two annotators.
3. Measure inter-rater agreement before treating labels as ground truth.
4. Re-run evaluation and compare synthetic vs real performance.
5. Add refund and dispute escalation rules.
6. Add technical severity detection for crashes, outages, and repeated failures.
7. Add low-confidence out-of-domain handling.
8. Add sarcasm and negation patterns.
9. Track conversation history and repeated contact count.
10. Evaluate reply quality with human review or an LLM judge.

### Longer-Term Improvements

1. Fine-tune the embedding model on real customer-support text.
2. Calibrate confidence scores with Platt scaling or isotonic regression.
3. Add a brand-policy layer for different escalation rules.
4. Replace or augment templates with controlled LLM generation.
5. Add multilingual embeddings for non-English messages.
6. Build an active-learning loop where human corrections improve the dataset.
7. Log false positives and false negatives from real agent review.

## 9. Code Quality

The project is structured as a modular NLP pipeline:

- Classifier, escalation, reply generation, preprocessing, and evaluation are separate modules.
- Tests are included with `pytest`.
- Settings are centralized in configuration files.
- The system can run locally without paid API dependencies.
- Template replies reduce hallucination risk.
- The decision log documents attribution and design reasoning.

This makes the project suitable as an academic MVP and portfolio artifact. The code is understandable, testable, and extensible, but the system should not be treated as production-ready without real data validation, calibration, monitoring, and human-in-the-loop review.

## 10. Conclusion

The customer support AI bot demonstrates a practical end-to-end NLP workflow: intent classification, escalation routing, and reply generation. On the curated 100-row evaluation set, it reaches 78% intent accuracy and 0.854 escalation F1. The strongest results are on billing and account-access messages, while complaint tone and mixed-intent messages remain challenging.

The most important conclusion is not that the model is production-ready. The important conclusion is that the prototype is transparent, measurable, and honest about its limitations. It provides a strong foundation for the next stage: testing with real customer messages and improving escalation behavior for refund, urgency, sarcasm, and multi-turn support cases.
