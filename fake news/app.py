# app.py
# ======================================
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load ML Models
cnn_model = load_model("cnn_fake_news_model.keras")
tokenizer = joblib.load("tokenizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Home route
@app.route("/")
def index():
    return render_template("index.html")

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    # Get text input from form
    news_text = request.form.get("news_text")

    if not news_text:
        return render_template("index.html", prediction="Please enter news text!")

    # Preprocess input (example for CNN model)
    # Convert text to sequence using tokenizer
    seq = tokenizer.texts_to_sequences([news_text])
    # Pad sequences to fixed length (adjust if needed)
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq_padded = pad_sequences(seq, maxlen=200)

    # Predict using CNN model
    prediction_prob = cnn_model.predict(seq_padded)[0][0]
    # Assuming binary classification: 0 = Real, 1 = Fake
    prediction_label = "Fake" if prediction_prob > 0.5 else "Real"

    return render_template("index.html", prediction=prediction_label)

# Run the app
if __name__ == "__main__":
    app.run(debug=False)
