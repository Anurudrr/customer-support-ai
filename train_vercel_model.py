"""
Train and export a lightweight classifier for Vercel serverless deployment.
Uses TF-IDF + LogisticRegression / Cosine Similarity so it fits in <5MB (under Vercel's 50MB limit).
"""
import pickle
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
train_df = pd.read_csv(ROOT / "data" / "processed" / "train.csv").dropna()

# Create lightweight TF-IDF + Logistic Regression pipeline
clf_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
    ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42))
])

clf_pipeline.fit(train_df["message"], train_df["intent"])

out_dir = ROOT / "models"
out_dir.mkdir(exist_ok=True)
out_file = out_dir / "vercel_classifier.pkl"

with open(out_file, "wb") as f:
    pickle.dump(clf_pipeline, f)

print(f"[OK] Exported lightweight Vercel model ({out_file.stat().st_size / 1024:.1f} KB) -> {out_file}")
