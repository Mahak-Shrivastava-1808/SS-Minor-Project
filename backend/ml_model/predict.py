
import joblib
from pathlib import Path

# 📂 Path setup
HERE = Path(__file__).parent

# Load trained model and vectorizer
try:
    model = joblib.load(HERE / "empathy_model.pkl")
    vectorizer = joblib.load(HERE / "vectorizer.pkl")   # ✅ same name as saved in train.py
    print("✅ Model & Vectorizer loaded successfully!")
except Exception as e:
    print("❌ Error loading model/vectorizer:", e)
    exit()


def predict_emotion(text: str):
    """Predict emotion from given text"""
    X_tfidf = vectorizer.transform([text])
    prediction = model.predict(X_tfidf)[0]
    return prediction


# 🔽 Example usage
if __name__ == "__main__":
    while True:
        user_input = input("\n📝 Enter a sentence (or 'quit' to exit): ")
        if user_input.lower() in ["quit", "exit"]:
            break
        emotion = predict_emotion(user_input)
        print(f"💡 Predicted Emotion: {emotion}")