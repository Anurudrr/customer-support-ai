"""Script to generate the Jupyter notebooks programmatically."""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def make_notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells,
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
        "id": f"cell_{hash(source) % 10**8:08x}",
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
        "id": f"md_{hash(source) % 10**8:08x}",
    }


# ── Notebook 1: EDA ───────────────────────────────────────────────────────────
nb1_cells = [
    md_cell("# 01 — Exploratory Data Analysis\n\nThis notebook explores the synthetic customer support dataset."),
    code_cell("""import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('.')), 'customer-support-ai'))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
BASE = Path('..')
train = pd.read_csv(BASE / 'data/processed/train.csv')
val   = pd.read_csv(BASE / 'data/processed/val.csv')
test  = pd.read_csv(BASE / 'data/processed/test.csv')
print('Train shape:', train.shape)
print('Val shape:  ', val.shape)
print('Test shape: ', test.shape)
"""),
    md_cell("## Dataset Overview"),
    code_cell("""train.head(10)"""),
    code_cell("""print("Missing values in train:")
print(train.isnull().sum())
print(f"\\nDuplicate rows: {train.duplicated().sum()}")
"""),
    md_cell("## Intent Distribution"),
    code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (df, title) in zip(axes, [(train, 'Train'), (val, 'Validation'), (test, 'Test')]):
    counts = df['intent'].value_counts()
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette='Blues_r')
    ax.set_title(f'{title} — Intent Distribution')
    ax.set_xlabel('Count')
plt.tight_layout()
plt.savefig('../outputs/figures/intent_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
"""),
    md_cell("## Message Length Analysis"),
    code_cell("""train['length'] = train['message'].str.split().str.len()
print(train.groupby('intent')['length'].describe().round(2))
"""),
    code_cell("""fig, ax = plt.subplots(figsize=(12, 5))
for intent in sorted(train['intent'].unique()):
    subset = train[train['intent'] == intent]['length']
    ax.hist(subset, bins=30, alpha=0.5, label=intent)
ax.set_xlabel('Message Length (words)')
ax.set_ylabel('Count')
ax.set_title('Message Length Distribution by Intent')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""),
    md_cell("## Sample Messages Per Intent"),
    code_cell("""for intent in sorted(train['intent'].unique()):
    samples = train[train['intent'] == intent]['message'].sample(3, random_state=42).tolist()
    print(f"\\n{'='*60}")
    print(f"Intent: {intent}")
    print('='*60)
    for s in samples:
        print(f"  • {s}")
"""),
]

nb1 = make_notebook(nb1_cells)
with open(NOTEBOOKS_DIR / "01_eda.ipynb", "w") as f:
    json.dump(nb1, f, indent=2)
print("Created 01_eda.ipynb")


# ── Notebook 2: Build Classifier ─────────────────────────────────────────────
nb2_cells = [
    md_cell("# 02 — Build Classifier\n\nThis notebook demonstrates loading data, generating embeddings, and running the classifier."),
    code_cell("""import sys, os
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_batch
from src.intent_classifier import IntentClassifier
from src.config import INTENTS
"""),
    md_cell("## Load Training Data"),
    code_cell("""train = pd.read_csv('../data/processed/train.csv')
val   = pd.read_csv('../data/processed/val.csv')
print(f"Training examples: {len(train)}")
print(f"Validation examples: {len(val)}")
train.head()
"""),
    md_cell("## Preprocess Text"),
    code_cell("""train['processed'] = preprocess_batch(train['message'].tolist())
val['processed']   = preprocess_batch(val['message'].tolist())
print("Sample preprocessed messages:")
for i in range(3):
    print(f"  Raw:  {train['message'].iloc[i]}")
    print(f"  Proc: {train['processed'].iloc[i]}")
    print()
"""),
    md_cell("## Train the Classifier\n\nThis encodes all training examples using `all-MiniLM-L6-v2`."),
    code_cell("""clf = IntentClassifier(k=5)
clf.fit(train['processed'].tolist(), train['intent'].tolist(), show_progress=True)
print("Classifier fitted!")
print(f"Training embeddings shape: {clf._train_embeddings.shape}")
"""),
    md_cell("## Predictions on Validation Set"),
    code_cell("""results = clf.predict_batch(val['processed'].tolist())
val['predicted_intent'] = [r['intent'] for r in results]
val['confidence'] = [r['confidence'] for r in results]

from sklearn.metrics import accuracy_score, classification_report
acc = accuracy_score(val['intent'], val['predicted_intent'])
print(f"Validation Accuracy: {acc:.4f}")
print()
print(classification_report(val['intent'], val['predicted_intent'], zero_division=0))
"""),
    md_cell("## Sample Predictions with Confidence"),
    code_cell("""sample_msgs = [
    "my refund still hasn't arrived",
    "app keeps crashing every time I open it",
    "I forgot my password and can't login",
    "Love your service, thank you!",
    "Could you please add dark mode?",
    "This is RIDICULOUS! I was charged twice!!!",
]

print(f"{'Message':<50} {'Intent':<20} {'Confidence'}")
print("-" * 80)
for msg in sample_msgs:
    from src.preprocessing import preprocess
    r = clf.predict_with_confidence(preprocess(msg))
    print(f"{msg[:48]:<50} {r['intent']:<20} {r['confidence']:.4f}")
"""),
    md_cell("## Confidence Distribution"),
    code_cell("""import matplotlib.pyplot as plt
correct = val['intent'] == val['predicted_intent']
plt.figure(figsize=(10, 5))
plt.hist(val[correct]['confidence'], bins=30, alpha=0.6, label='Correct', color='steelblue')
plt.hist(val[~correct]['confidence'], bins=30, alpha=0.6, label='Incorrect', color='salmon')
plt.axvline(x=0.60, color='red', linestyle='--', label='Threshold (0.60)')
plt.xlabel('Confidence (Mean Cosine Similarity)')
plt.ylabel('Count')
plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
plt.legend()
plt.tight_layout()
plt.show()
"""),
]

nb2 = make_notebook(nb2_cells)
with open(NOTEBOOKS_DIR / "02_build_classifier.ipynb", "w") as f:
    json.dump(nb2, f, indent=2)
print("Created 02_build_classifier.ipynb")


# ── Notebook 3: Failure Analysis ─────────────────────────────────────────────
nb3_cells = [
    md_cell("# 03 — Failure Analysis\n\nThis notebook explores misclassified examples after running evaluation."),
    code_cell("""import sys, os
sys.path.insert(0, '..')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

results_path = Path('../outputs/evaluation_results/evaluation_results.csv')
if not results_path.exists():
    print("Run: python -m evaluation.evaluate  first to generate results.")
else:
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} evaluation results")
    df.head()
"""),
    md_cell("## Overall Accuracy"),
    code_cell("""if results_path.exists():
    acc = (df['true_intent'] == df['predicted_intent']).mean()
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    failures = df[df['true_intent'] != df['predicted_intent']]
    print(f"Failures: {len(failures)} / {len(df)} ({len(failures)/len(df)*100:.1f}%)")
"""),
    md_cell("## Failure Cases — Low Confidence First"),
    code_cell("""if results_path.exists():
    failures_sorted = failures.sort_values('confidence')
    display_cols = ['message', 'true_intent', 'predicted_intent', 'confidence', 'should_escalate']
    print(failures_sorted[display_cols].head(10).to_string(index=False))
"""),
    md_cell("## Confusion Matrix"),
    code_cell("""if results_path.exists():
    from sklearn.metrics import confusion_matrix
    from src.config import INTENTS
    cm = confusion_matrix(df['true_intent'], df['predicted_intent'], labels=INTENTS)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=INTENTS, yticklabels=INTENTS, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Counts)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
"""),
    md_cell("## Confidence vs Correctness"),
    code_cell("""if results_path.exists():
    df['correct'] = df['true_intent'] == df['predicted_intent']
    print("Mean confidence by correctness:")
    print(df.groupby('correct')['confidence'].describe().round(4))
"""),
    md_cell("## Common Failure Patterns"),
    code_cell("""if results_path.exists():
    failure_pairs = failures.groupby(['true_intent', 'predicted_intent']).size().reset_index(name='count')
    failure_pairs = failure_pairs.sort_values('count', ascending=False)
    print("Most common confusion pairs:")
    print(failure_pairs.head(10).to_string(index=False))
"""),
    md_cell("## Conclusions\n\nCommon failure patterns:\n- Overlapping intents (complaint_angry vs billing_issue)\n- Short messages with insufficient context\n- Ambiguous phrasing that could belong to multiple intents\n\nThe escalation safety net (confidence < 0.60) catches many of these low-confidence predictions."),
]

nb3 = make_notebook(nb3_cells)
with open(NOTEBOOKS_DIR / "03_failure_analysis.ipynb", "w") as f:
    json.dump(nb3, f, indent=2)
print("Created 03_failure_analysis.ipynb")

print("\n[OK] All notebooks created.")
