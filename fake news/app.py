from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px
import plotly.io as pio

# ======================================
# Flask Setup
# ======================================
app = Flask(__name__)

# ======================================
# Load Saved Models and Objects
# ======================================
cnn_model = load_model("cnn_fake_news_model.keras")
tokenizer = joblib.load("tokenizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

MAXLEN = 300


# ======================================
# Route 1 — Home Page (News Detector)
# ======================================
@app.route("/")
def home():
    return render_template("index.html", title="Fake News Detector")


# ======================================
# Route 2 — Prediction Logic
# ======================================
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        news_text = request.form["news"]

        # Convert text to padded sequence
        sequence = tokenizer.texts_to_sequences([news_text])
        padded = pad_sequences(sequence, maxlen=MAXLEN, padding="post", truncating="post")

        # Predict
        prediction = cnn_model.predict(padded)
        label = 1 if prediction[0][0] > 0.5 else 0
        predicted_class = label_encoder.inverse_transform([label])[0]
        confidence = float(prediction[0][0]) if label == 1 else float(1 - prediction[0][0])

        return render_template(
            "index.html",
            title="Fake News Detector",
            prediction_text=f"🔎 This news is: {predicted_class.upper()} (Confidence: {confidence:.2f})"
        )

    return render_template("index.html", title="Fake News Detector")


# ======================================
# Route 3 — Interactive Dashboard
# ======================================
@app.route("/dashboard")
def dashboard():
    # Load data
    real = pd.read_csv("True.csv")
    fake = pd.read_csv("Fake.csv")

    real["label"] = "Real"
    fake["label"] = "Fake"
    df = pd.concat([real, fake], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)

    # ------------------------
    # Chart 1: Real vs Fake
    # ------------------------
    fig1 = px.pie(
        df,
        names="label",
        title="Distribution of Real vs Fake News",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    chart1 = pio.to_html(fig1, full_html=False)

    # ------------------------
    # Chart 2: Subject Distribution
    # ------------------------
    subject_counts = df["subject"].value_counts().reset_index()
    subject_counts.columns = ["subject", "count"]
    fig2 = px.bar(
        subject_counts,
        x="subject",
        y="count",
        title="News Topics Distribution",
        color="subject",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig2.update_layout(xaxis_tickangle=-45)
    chart2 = pio.to_html(fig2, full_html=False)

    # ------------------------
    # Chart 3: Word Count Comparison
    # ------------------------
    df["title_word_count"] = df["title"].apply(lambda x: len(str(x).split()))
    fig3 = px.box(
        df,
        x="label",
        y="title_word_count",
        color="label",
        title="Title Word Count Comparison (Fake vs Real)"
    )
    chart3 = pio.to_html(fig3, full_html=False)

    # ------------------------
    # Chart 4: Timeline Trend
    # ------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    monthly = df.groupby([df["date"].dt.to_period("M"), "label"]).size().unstack()
    monthly.index = monthly.index.astype(str)
    monthly = monthly.reset_index().melt(id_vars="date", var_name="label", value_name="count")

    fig4 = px.line(
        monthly,
        x="date",
        y="count",
        color="label",
        title="Monthly News Volume (Fake vs Real)",
        markers=True
    )
    fig4.update_xaxes(type="category")
    chart4 = pio.to_html(fig4, full_html=False)

    # Render all charts
    return render_template(
        "dashboard.html",
        title="Interactive Dashboard",
        chart1=chart1,
        chart2=chart2,
        chart3=chart3,
        chart4=chart4
    )


if __name__ == "__main__":
    app.run(debug=True)
