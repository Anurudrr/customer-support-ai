# Decision Log

This document records the main engineering and evaluation decisions for the AI-powered customer support chatbot.

## 1. Embedding-Based Intent Classification

- **Citation:** https://github.com/Anurudrr/customer-support-ai
- **Why:** The referenced project demonstrates a proven, efficient, and understandable approach to customer-support intent classification using embeddings.
- **Adaptation:** The classifier structure was retained, while the intent labels, examples, and response templates were modified for the project brand voice and support scenarios.

## 2. Rule-Based Escalation

- **Citation:** https://github.com/Anurudrr/customer-support-ai
- **Why:** Rule-based escalation is transparent, maintainable, and easy to explain in an academic or portfolio project.
- **Adaptation:** The escalation conditions were customized around this project's data patterns, including anger signals, confidence thresholding, billing complaints, feature requests, and unknown intents.

## 3. Template-Based Reply Generation

- **Citation:** https://github.com/Anurudrr/customer-support-ai
- **Why:** Template replies reduce hallucination risk, keep tone consistent, and make the system predictable.
- **Adaptation:** The reply templates were customized for Spotify/Amazon-style customer-support scenarios and brand-safe responses.

## 4. Synthetic Data for Evaluation

- **Why:** Synthetic examples made it possible to build and evaluate a quick prototype without depending on restricted or manually downloaded customer data.
- **Limitation:** Real customer-support data would be noisier, more ambiguous, and less balanced than the synthetic examples.

## 5. SentenceTransformer Model Selection

- **Decision:** Use `all-MiniLM-L6-v2` from SentenceTransformers.
- **Why:** The model is fast, lightweight, runs locally, and produces useful semantic embeddings for short intent-classification messages.

## 6. KNN Similarity Matching

- **Decision:** Use a KNN classifier with cosine similarity over sentence embeddings.
- **Why:** KNN is simple, interpretable, and fast at inference for the project scale. It also makes predictions easier to inspect because similar training examples can be reviewed.

## 7. Five Escalation Rules

- **Decision:** Use five escalation rules covering angry billing/complaint cases, low confidence, feature requests, strong anger, and unknown intent.
- **Why:** A small fixed rule set keeps the decision logic clear while covering the most important high-risk support cases.

## 8. No Fine-Tuning

- **Decision:** Use the embedding model off the shelf.
- **Why:** Fine-tuning would require more labelled data, more compute, and more time. The synthetic data is sufficient for an MVP demonstration.

## 9. No Sarcasm Detection

- **Decision:** Do not add sarcasm detection in the MVP.
- **Why:** Sarcasm detection would require more complex NLP and a specific labelled dataset. It is listed as a limitation and possible future improvement.

## 10. Template Replies Instead of Generative Replies

- **Decision:** Use deterministic template replies rather than LLM-generated responses.
- **Why:** Templates maintain brand consistency, avoid hallucination, and make responses easier to audit.

## 11. Synthetic Evaluation Set

- **Decision:** Evaluate the system on a synthetic evaluation set.
- **Limitation:** The evaluation does not fully measure real-world generalisation because the examples are generated and cleaner than real customer messages.

## 12. No Inter-Rater Agreement Study

- **Decision:** Do not perform inter-rater agreement analysis.
- **Why:** The labels are synthetic and were created by a single labelling process, so there were no multiple human annotators to compare.

## 13. Basic Metrics Only

- **Decision:** Report standard metrics such as accuracy and F1.
- **Why:** Accuracy and F1 are sufficient for an MVP. More advanced evaluation, including human reply-quality judgement and production calibration, is left for future work.
