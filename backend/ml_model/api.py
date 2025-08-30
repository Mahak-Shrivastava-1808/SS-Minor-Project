from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# ✅ FastAPI app
app = FastAPI(title="Empathy Emotion Prediction API")

# ✅ Model & vectorizer load
MODEL_PATH = "empathy_model.pkl"
VECTORIZER_PATH = "vectorizer copy.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Model and Vectorizer loaded successfully.")
except Exception as e:
    print("❌ Error loading model/vectorizer:", e)
    model = None
    vectorizer = None


# ✅ Input schema
class InputText(BaseModel):
    text: str


# ✅ Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to Empathy Emotion Prediction API 🚀"}


# ✅ Prediction endpoint
@app.post("/predict")
def predict(data: InputText):
    if not model or not vectorizer:
        return {"error": "Model not loaded. Please check server logs."}

    try:
        # Transform input
        text_vector = vectorizer.transform([data.text])

        # Predict
        prediction = model.predict(text_vector)[0]

        return {"input_text": data.text, "predicted_emotion": prediction}
    except Exception as e:
        return {"error": str(e)}
