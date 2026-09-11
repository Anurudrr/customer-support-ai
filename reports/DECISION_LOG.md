# Decision Log — AI Customer Support NLP System

This document records approximately 12 key engineering decisions made during the design and implementation of this system.

---

## Decision 1: Embedding Model Selection

### Problem
The system needs to classify customer intent from short, noisy text messages. Traditional bag-of-words models struggle with paraphrases, typos, and informal language.

### Options
- TF-IDF + Logistic Regression
- Word2Vec averaging
- SentenceTransformers (`all-MiniLM-L6-v2`)
- Fine-tuned BERT

### Decision
Use `SentenceTransformer('all-MiniLM-L6-v2')`.

### Reason
Sentence-transformers produce dense semantic embeddings that capture meaning rather than just vocabulary. `all-MiniLM-L6-v2` is fast (< 90 MB), runs locally without a GPU, handles paraphrases well, and is widely used in academic NLP projects. Fine-tuning BERT would require more labelled data and significantly higher compute resources — overkill for a student-level system.

---

## Decision 2: Similarity-Based Classification (KNN) Over Trained Classifier

### Problem
Should we train a parametric classifier (logistic regression / SVM on top of embeddings) or use a non-parametric similarity-based approach?

### Options
- Logistic Regression on frozen embeddings
- SVM on frozen embeddings
- K-Nearest Neighbours via cosine similarity

### Decision
Use K-Nearest Neighbours with cosine similarity on the embedding space.

### Reason
KNN requires no training step beyond embedding, naturally handles new classes by adding examples, and is transparent and debuggable. For a portfolio project with ~1000 training examples, KNN and logistic regression perform comparably. KNN also makes it easy to inspect which training examples drove a prediction (the top-K neighbours).

---

## Decision 3: Confidence Threshold Set to 0.60

### Problem
The classifier should trigger escalation when its prediction is uncertain. What threshold to use?

### Options
- 0.50 (very permissive — too many uncertain predictions handled by bot)
- 0.60 (moderate — recommended baseline for semantic similarity classifiers)
- 0.70 (strict — too many unnecessary escalations)
- Dynamic / learned threshold

### Decision
Use `0.60` as the default, stored in `config.py` so it can be easily changed.

### Reason
For cosine similarity in the range [0, 1], a score below 0.60 typically means the query is not clearly close to any training cluster. This is an assumption based on common practice rather than a trained calibration curve. The threshold is documented clearly, and the code notes that cosine similarity is NOT a calibrated probability.

---

## Decision 4: Rule-Based Escalation Over ML-Based Escalation

### Problem
Should escalation be decided by a learned ML model or by explicit rules?

### Options
- Train a binary classifier for escalation (requires labelled escalation data)
- Rule-based decision engine
- Hybrid (rules + ML fallback)

### Decision
Use a rule-based decision engine with clear, documented rules.

### Reason
Escalation in production support systems must be explainable and auditable. Rule-based systems are transparent, easy to modify, and don't require a separate labelled dataset. For an academic project, rules also make the system easier to understand and present. A learning-based escalation system would be a natural future improvement.

---

## Decision 5: Separate Anger Detection Module

### Problem
Anger/frustration affects escalation independently of intent. Should anger detection be embedded in the classifier or separate?

### Options
- Incorporate anger as a separate intent label
- Detect anger signals in the escalation module
- Separate `anger_detector.py` module

### Decision
Create a dedicated `anger_detector.py` module.

### Reason
Separating concerns keeps each module focused. Anger detection needs to run on the *raw* text (before lowercasing removes caps signals), so it cannot be part of the preprocessing step. A separate module makes the logic reusable, testable, and easy to improve independently.

---

## Decision 6: Template-Based Reply Generation

### Problem
How should the bot generate responses?

### Options
- Large Language Model (GPT-4, Gemini) via API
- Fine-tuned seq2seq (T5, BART)
- Template-based generation

### Decision
Use template-based generation with multiple templates per intent, selected deterministically by message hash.

### Reason
An academic project should work locally without an API key or GPU. Template-based generation is predictable, reproducible, and easy to audit. Having 5 templates per intent ensures variety without randomness. This is explicitly identified as a baseline — LLM-based generation is listed as a future improvement.

---

## Decision 7: Preserve Sentiment/Emotional Words in Preprocessing

### Problem
Aggressive text normalisation (stemming, stop-word removal, punctuation stripping) might remove anger signals needed for escalation.

### Options
- Aggressive normalisation (remove stop words, stem, strip punctuation)
- Minimal normalisation (just lowercase + whitespace cleanup)
- Keep emotional words explicitly

### Decision
Use minimal preprocessing: lowercase, URL/mention handling, whitespace normalisation, and exclamation normalisation. Do NOT remove stop words or apply stemming.

### Reason
The SentenceTransformer handles subword tokenisation internally — it doesn't need stop-word removal. Emotional words like "furious", "ridiculous", and "disgusting" are critical for anger detection and should not be stripped. All-caps patterns are detected *before* lowercasing.

---

## Decision 8: Precompute Training Embeddings at Fit Time

### Problem
Encoding the full training set at every prediction would be extremely slow.

### Options
- Recompute all training embeddings per prediction
- Cache embeddings at fit() time
- Cache with disk-backed storage

### Decision
Compute all training embeddings once in `fit()` and store as a NumPy array in memory. Persist to disk with `save()` so re-encoding is not needed across sessions.

### Reason
Encoding ~1000 training examples takes several seconds — doing this per-prediction would make the system unusable. Caching in memory is the standard approach and keeps the prediction path fast (< 100ms).

---

## Decision 9: Use F1 in Addition to Accuracy

### Problem
Which metrics should the evaluation report?

### Options
- Accuracy only
- Accuracy + F1
- Full set: accuracy, precision, recall, F1, per-class metrics, confusion matrix

### Decision
Report accuracy, macro F1, precision, recall, per-class F1, confusion matrix, and escalation binary metrics.

### Reason
Accuracy is misleading for imbalanced datasets. Macro F1 treats all classes equally regardless of size. Per-class metrics reveal which intents are hard to classify. A confusion matrix shows which intent pairs are most commonly confused. For escalation (binary), precision/recall trade-off matters — false negatives (missed escalations) are more costly than false positives.

---

## Decision 10: Manually Labelled Evaluation Set

### Problem
How should evaluation examples be generated?

### Options
- Use the test split (already seen during design)
- Create a completely separate evaluation set with rule-based labels
- Use real-world held-out data (not available)

### Decision
Create a synthetic evaluation set with deterministic `should_escalate` labels derived from the same rules as the escalation engine.

### Reason
Without a real labelled dataset, we cannot have a fully independent evaluation. The generated evaluation set is honest about this limitation. The labels are derived by applying the same rules used at inference time, which tests the end-to-end pipeline consistency rather than generalisation. This is clearly documented as a limitation.

---

## Decision 11: Synthetic Demo Data Strategy

### Problem
The project requires a large real Twitter customer support dataset (not available).

### Options
- Download from Kaggle (requires account + manual step)
- Use a tiny sample (insufficient for demonstration)
- Generate realistic synthetic data

### Decision
Generate 1400 synthetic examples (1000 train / 200 val / 200 test / 200 eval) with intentional diversity: typos, informal language, Twitter-style short messages, emotional phrasing, and varied templates.

### Reason
The project should be runnable out of the box. The data is clearly labelled as synthetic/demo, and the code is structured so a real dataset can be dropped in at `data/raw/twitter_conversations.csv`. The synthetic data is diverse enough to demonstrate the system meaningfully.

---

## Decision 12: No External API Dependency

### Problem
Should the system use a cloud LLM for classification or reply generation?

### Options
- OpenAI GPT-4 for classification + reply generation
- Gemini / Claude API
- Fully local models only

### Decision
Everything runs locally. No API keys required.

### Reason
An academic/portfolio project should be reproducible by anyone with a computer. External APIs add cost, require accounts, and introduce failure modes outside the developer's control. The `all-MiniLM-L6-v2` model downloads automatically from Hugging Face on first run.

---

## Decision 13: Streamlit UI as Optional Component

### Problem
Should a web UI be built as part of the core project?

### Options
- No UI (CLI only)
- Streamlit UI (optional, post-core)
- FastAPI + React frontend

### Decision
Include a Streamlit UI as an optional extra, clearly separated from the core architecture.

### Reason
The CLI is sufficient for demonstration and evaluation. A Streamlit UI adds visual appeal for presentations without adding complexity to the core pipeline. FastAPI + React would be disproportionately complex for an academic project.
