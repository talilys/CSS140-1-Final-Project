"""
train_model.py  v3
Trains on dataset that includes slangs and emojis.
Normalizer runs on each row so the model learns from enriched text.
"""

import os
import re
import pickle
import pandas as pd
import sys

# Limit OpenBLAS threading to prevent memory allocation errors
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(__file__))
from normalizer import normalize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def full_pipeline(raw_text):
    """Normalize (emojis/slangs) then clean for TF-IDF."""
    enriched, _, _ = normalize(raw_text)
    return preprocess_text(enriched)


print("Loading dataset...")
df = pd.read_csv("data/dataset.csv")
print(f"Dataset loaded: {len(df)} rows")
print(df["label"].value_counts())

df["clean_text"] = df["text"].apply(full_pipeline)

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_features=5000,
        sublinear_tf=True,
        min_df=1,
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=1.5,
        solver="lbfgs",
    ))
])

print("\nTraining model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n=== Model Evaluation ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel saved to model/sentiment_model.pkl")

# Test cases — slangs, emojis, sarcasm, emoticons
samples = [
    "grabe sobrang stress ko ngayon 😭",
    "slay talaga yung presentation namin 🔥",
    "vibing lang no biggie",
    "oh great another deadline love that for me 🙃",
    "totally fine not crying at all haha 💀",
    "what a wonderful day to fail my exam",
    "T_T di ko na kaya",
    "yesss finally passed!! 🎉",
    "burnout na ako fr fr ngl",
    "edi wow ang galing mo naman :)",
    "down bad talaga sa thesis namin huhu",
    "kaya natin laban! 💪",
]

print("\n=== Sample Predictions ===")
for t in samples:
    enriched, is_sarcasm, sarcasm_type = normalize(t)
    cleaned = preprocess_text(enriched)
    pred = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    conf = round(max(proba) * 100, 1)
    sarcasm_flag = f" [SARCASM:{sarcasm_type}]" if is_sarcasm else ""
    print(f"Input:    {t}")
    print(f"Enriched: {enriched[:80]}...")
    print(f"Result:   {pred} ({conf}%){sarcasm_flag}\n")