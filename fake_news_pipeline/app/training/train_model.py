import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json
from datetime import datetime


SHARED_MODELS_DIR = "../../shared_models"
LOCAL_MODELS_DIR = "models"


os.makedirs(SHARED_MODELS_DIR, exist_ok=True)
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)


df = pd.read_csv("data/final_training_data.csv")

print(f"Ukupan broj vesti za trening: {len(df)}")
print("Raspodela labela:\n", df['label'].value_counts())

X = df['text'].fillna("")
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words='english'
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))


metadata = {
    "version": "2.0",
    "trained_at": datetime.now().isoformat(),
    "accuracy": float(accuracy),
    "num_samples": len(df),
    "num_fake": int((df['label'] == 0).sum()),
    "num_true": int((df['label'] == 1).sum()),
    "test_accuracy": float(accuracy),
    "model_type": "TF-IDF + Logistic Regression",
    "features": {
        "max_features": 10000,
        "ngram_range": "(1, 2)",
        "stop_words": "english"
    }
}


print("\n" + "="*80)
print("Čuvam modele...")
print("="*80)

joblib.dump(model, f"{LOCAL_MODELS_DIR}/fake_news_model_updated.pkl")
joblib.dump(vectorizer, f"{LOCAL_MODELS_DIR}/tfidf_vectorizer.pkl")
print(f"✓ Local models sačuvani u: {LOCAL_MODELS_DIR}/")


joblib.dump(model, f"{SHARED_MODELS_DIR}/lr_model.pkl")
joblib.dump(vectorizer, f"{SHARED_MODELS_DIR}/tfidf_vectorizer.pkl")

with open(f"{SHARED_MODELS_DIR}/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Shared models sačuvani u: {SHARED_MODELS_DIR}/")
print(f"✓ Metadata sačuvan")

print("\n" + "="*80)
print("🎉 TRENIRANJE ZAVRŠENO!")
print("="*80)
print(f"Model verzija: {metadata['version']}")
print(f"Accuracy: {metadata['accuracy']:.4f} ({metadata['accuracy']*100:.2f}%)")
print(f"Dataset: {metadata['num_samples']} vesti")
print(f"  - FAKE: {metadata['num_fake']}")
print(f"  - TRUE: {metadata['num_true']}")
print(f"\n✅ Streamlit app će automatski koristiti novi model pri sledećem pokretanju!")
print("="*80)