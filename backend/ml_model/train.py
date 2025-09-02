import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

# 📂 Path setup
HERE = Path(__file__).parent
DATASET = HERE / "train_data.csv"   # ✅ Corrected file name

try:
    print("📂 Script folder :", HERE)
    print("📂 Running from  :", Path.cwd())
    print("📄 Files here    :", [p.name for p in HERE.iterdir() if p.is_file()])
    print("📄 Using dataset :", DATASET)

    # Load dataset
    df = pd.read_csv(DATASET)
    print("✅ Dataset loaded:", DATASET)
    print("➡ Columns:", df.columns.tolist())

    # ✅ Use correct columns
    X_text = df["SENTENCES"].astype(str)
    y = df["EMOTION"].astype(str)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

    # Vectorization
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_tfidf, y_train)

    # Evaluation
    y_pred = model.predict(X_test_tfidf)
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save model + vectorizer
    joblib.dump(model, HERE / "empathy_model.pkl")
    joblib.dump(vectorizer, HERE / "vectorizer.pkl")
    print("✅ Model and vectorizer saved!")

except Exception as e:
    print("❌ Error:", e)