# AI-Powered Customer Support Chatbot

> An academic NLP project demonstrating intent classification, escalation decision-making, and template-based reply generation for customer support automation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Vercel Deployment](https://img.shields.io/badge/Vercel-Live%20Demo-brightgreen.svg)](https://customer-support-ai-olive.vercel.app)

**🌐 Live Production Demo:** [https://customer-support-ai-olive.vercel.app](https://customer-support-ai-olive.vercel.app)

---

## Features

- **7-class intent classification** using `SentenceTransformer` (`all-MiniLM-L6-v2`) + K-Nearest Neighbours cosine similarity
- **Transparent escalation engine** with 5 explicit, modifiable rules
- **Anger detection** using keyword lists + structural signals (ALL-CAPS, exclamation marks)
- **Template-based reply generation** with multiple templates per intent and deterministic selection
- **Interactive CLI** with session stats
- **Batch evaluation** with confusion matrix, per-intent F1, failure analysis
- **Pytest test suite** (35+ tests)
- **Optional Streamlit web UI**
- **Synthetic demo dataset** (1,400 realistic examples) — replaceable with real data

---

## Architecture

```
Customer Message
      ↓
Text Preprocessing (lowercase, URL/mention handling, normalization)
      ↓
Intent Classification (SentenceTransformer + KNN cosine similarity)
      ↓
Anger Detection (on raw text — keywords, ALL-CAPS, exclamation marks)
      ↓
Escalation Decision (5 transparent rules)
      ↓
Reply Generation (templates — escalation or intent-specific)
      ↓
Result: {intent, confidence, should_escalate, reason, reply}
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `sentence-transformers` | Semantic text embeddings |
| `scikit-learn` | Cosine similarity, metrics |
| `numpy / pandas` | Data handling |
| `matplotlib / seaborn` | Visualisations |
| `pytest` | Testing |
| `streamlit` | Optional web UI |

---

## Project Structure

```
customer-support-ai/
├── data/
│   ├── raw/twitter_conversations.csv   ← Drop real dataset here
│   ├── processed/{train,val,test}.csv
│   ├── evaluation_set.csv
│   └── generate_data.py                ← Synthetic data generator
│
├── src/
│   ├── config.py                       ← All configuration constants
│   ├── preprocessing.py                ← Text cleaning
│   ├── anger_detector.py               ← Keyword + structural anger signals
│   ├── intent_classifier.py            ← Embedding KNN classifier
│   ├── escalation_decider.py           ← Rule-based escalation
│   ├── reply_generator.py              ← Template-based replies
│   ├── pipeline.py                     ← Orchestrates all components
│   ├── data_loader.py                  ← CSV loading utilities
│   └── main.py                         ← CLI logic
│
├── evaluation/
│   ├── metrics.py                      ← Classification + escalation metrics
│   ├── confusion_matrix.py             ← Confusion matrix plots
│   ├── failure_analysis.py             ← Failure case identification
│   ├── judge.py                        ← Heuristic reply quality judge
│   └── evaluate.py                     ← Main evaluation script
│
├── tests/
│   ├── test_classifier.py
│   ├── test_escalation.py
│   ├── test_reply_generator.py
│   └── test_pipeline.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_build_classifier.ipynb
│   └── 03_failure_analysis.ipynb
│
├── reports/
│   ├── REPORT.md
│   ├── DECISION_LOG.md
│   └── failure_analysis.md             ← Generated after evaluation
│
├── outputs/
│   ├── figures/                        ← Saved plots
│   └── evaluation_results/             ← Saved metrics JSON/CSV
│
├── run.py                              ← CLI entry point
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Installation

### 1. Clone / navigate to the project

```bash
cd customer-support-ai
```

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
# For development (testing, notebooks):
pip install -r requirements-dev.txt
```

---

## Generating the Dataset

The project ships with a synthetic data generator. Run it once before training:

```bash
python data/generate_data.py
```

This creates:
- `data/processed/train.csv` (1,000 examples)
- `data/processed/val.csv` (200 examples)
- `data/processed/test.csv` (200 examples)
- `data/evaluation_set.csv` (200 labelled examples)

> **Note:** This is synthetic demo data. See [Using Real Data](#using-real-data) to replace it.

---

## How to Run

### Interactive CLI

```bash
python run.py
```

Example session:
```
============================================================
  AI CUSTOMER SUPPORT BOT
============================================================

You > My refund still hasn't arrived!

Processing ...
------------------------------------------------------------
  Intent      : billing_issue
  Confidence  : 0.8234
  Escalate    : NO ✓

  Bot Reply:
  I understand your concern about the billing. Let me look into this right away.
------------------------------------------------------------

You > This is RIDICULOUS! I was charged twice!!!

Processing ...
------------------------------------------------------------
  Intent      : billing_issue
  Confidence  : 0.8891
  Escalate    : YES ⚠
  Reason      : Angry customer with billing issue
  Anger Signals: anger keywords: ridiculous; all-caps words: RIDICULOUS; 2 exclamation marks

  Bot Reply:
  I've escalated your billing concern to our dedicated billing team...
------------------------------------------------------------

You > /stats
  Total messages : 2
  Escalated      : 1
  Bot handled    : 1
  Escalation rate: 50.0%

You > /quit
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/stats` | Session statistics |
| `/quit` | Exit |

---

## How to Train / Build the Classifier

The classifier is built automatically when you run `run.py` for the first time. To rebuild manually:

```bash
python -c "from src.pipeline import build_pipeline; build_pipeline()"
```

The trained embeddings are saved to `models/intent_classifier/classifier_state.pkl` and reloaded on subsequent runs.

---

## How to Evaluate

```bash
python -m evaluation.evaluate
```

This will:
1. Load `data/evaluation_set.csv`
2. Run predictions with the full pipeline
3. Print intent classification metrics (accuracy, macro F1, per-intent F1)
4. Print escalation metrics (precision, recall, F1)
5. Save confusion matrix to `outputs/figures/confusion_matrix.png`
6. Save confidence distribution to `outputs/figures/confidence_distribution.png`
7. Save metrics JSON to `outputs/evaluation_results/metrics.json`
8. Save full results CSV to `outputs/evaluation_results/evaluation_results.csv`
9. Generate `reports/failure_analysis.md`

---

## How to Run Tests

```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Optional Streamlit Web UI

```bash
streamlit run app.py
```

*(Requires `streamlit` to be installed — included in `requirements.txt`.)*

---

## Example Input/Output

### Example 1: Billing Issue (No Escalation)
```
Input:  "My refund still hasn't arrived"
Output: intent=billing_issue, confidence=0.82, escalate=NO
Reply:  "I understand your concern about the billing. Let me look into this right away..."
```

### Example 2: Angry Billing (Escalation)
```
Input:  "This is RIDICULOUS! I was charged twice!!!"
Output: intent=billing_issue, confidence=0.89, escalate=YES
Reason: Angry customer with billing issue
Reply:  "I've escalated your billing concern to our dedicated billing team..."
```

### Example 3: Technical Issue
```
Input:  "App keeps crashing every time I open it"
Output: intent=technical_issue, confidence=0.91, escalate=NO
Reply:  "Sorry you're having trouble! Please try restarting the app..."
```

### Example 4: Feature Request (Always Escalates)
```
Input:  "Could you please add dark mode?"
Output: intent=feature_request, confidence=0.88, escalate=YES
Reason: Feature requests are forwarded to the product team
Reply:  "Thank you for your suggestion! I've passed it on to our product team..."
```

### Example 5: Positive (No Escalation)
```
Input:  "Love your service, thank you!"
Output: intent=positive, confidence=0.93, escalate=NO
Reply:  "Thank you so much! We're really glad you're enjoying the service."
```

---

## Dataset Instructions

### Using Real Data

1. Download the Customer Support on Twitter dataset from Kaggle:
   https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

2. Process it to have `message` and `intent` columns.

3. Place it at:
   ```
   data/raw/twitter_conversations.csv
   ```

4. Run:
   ```bash
   python -c "from src.pipeline import build_pipeline; build_pipeline('data/raw/twitter_conversations.csv')"
   ```

---

## Evaluation Metrics

| Category | Metrics |
|----------|---------|
| Intent Classification | Accuracy, Macro F1, per-intent Precision/Recall/F1 |
| Escalation | Binary Precision, Recall, F1, TP/TN/FP/FN |
| Confidence | Mean, Std, Min, Max, Percentiles |
| Reply Quality | Heuristic judge score (0–1) |

---

## Intents

| Intent | Description |
|--------|-------------|
| `billing_issue` | Refunds, charges, billing errors |
| `technical_issue` | App crashes, bugs, errors |
| `account_access` | Login problems, password reset |
| `complaint_angry` | Frustrated, angry customers |
| `complaint_normal` | Mild dissatisfaction, feedback |
| `feature_request` | Suggestions, feature asks |
| `positive` | Compliments, satisfaction |

---

## Limitations

- Uses synthetic demo data (real dataset must be downloaded separately)
- Confidence score is cosine similarity, NOT a calibrated probability
- Template-based replies lack nuance for complex situations
- No multi-turn conversation support
- English only
- No sarcasm detection

---

## Future Improvements

- Fine-tune embedding model on domain data
- Replace templates with local LLM (Llama, Mistral)
- Add conversation history support
- Calibrate confidence scores
- Add multilingual support
- Active learning loop for mislabelled examples

---

## Attribution & Citations

This project adapts and extends:
- **GitHub:** https://github.com/Anurudrr/customer-support-ai
- **What we kept:** Embedding + KNN architecture, rule-based escalation, template replies
- **What we added:** Full evaluation report, failure analysis, honest limitations section

See `reports/DECISION_LOG.md` for detailed attribution and adaptations.

---

## License

MIT — see [LICENSE](./LICENSE)
