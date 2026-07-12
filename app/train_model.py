import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
from utils.preprocessing import clean_resume


# ==============================
# Paths
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
# ==============================
# Load Dataset
# ==============================

DATA_PATH = BASE_DIR / "data" / "raw" / "resumes_dataset.jsonl"

print("=" * 50)
print("Loading Dataset...")
print("=" * 50)

df = pd.read_json(DATA_PATH, lines=True)

print("Dataset Loaded Successfully!")
print(f"Dataset Shape : {df.shape}")
print(df.head())

# ==============================
# Data Preprocessing
# ==============================
print("=" * 50)
print("Cleaning Resume Text...")
print("=" * 50)

TEXT_COLUMN = "Text"
TARGET_COLUMN = "Category"

df["Cleaned_Text"] = df[TEXT_COLUMN].apply(clean_resume)

X = df["Cleaned_Text"]
y = df[TARGET_COLUMN]
# ==============================
# Encode Labels
# ==============================
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)


# ==============================
# Train Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# TF-IDF Vectorization
# ==============================
print("=" * 50)
print("Vectorizing Resume Text...")
print("=" * 50)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)


# ==============================
# Train Model
# ==============================
print("=" * 50)
print("Training Logistic Regression Model...")
print("=" * 50)

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(X_train, y_train)


# ==============================
# Evaluation
# ==============================
print("=" * 50)
print("Evaluating Model...")
print("=" * 50)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy : {accuracy:.4f}\n")
print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))


# ==============================
# Save Models
# ==============================
print("=" * 50)
print("Saving Models...")
print("=" * 50)

joblib.dump(model, MODEL_DIR / "resume_classifier.pkl")
joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")

print(" resume_classifier.pkl saved")
print(" tfidf_vectorizer.pkl saved")
print(" label_encoder.pkl saved")

print("\n Training Completed Successfully!")