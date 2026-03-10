import streamlit as st
import torch
import os
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Load model path
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "finbert_model")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

labels = ["Bearish", "Neutral", "Bullish"]


# Streamlit UI
st.title("📈 Financial News Sentiment Analyzer")

st.write(
    "Enter a financial news headline or tweet and the model will predict whether the sentiment is **Bearish, Neutral, or Bullish**."
)


text = st.text_area("Enter Financial News or Tweet")


if st.button("Predict Sentiment"):

    if text.strip() == "":
        st.warning("Please enter some financial text.")
    else:

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)

        prediction = torch.argmax(probs).item()

        sentiment = labels[prediction]

        confidence = probs[0][prediction].item() * 100


        # Colored sentiment display
        if sentiment == "Bullish":
            st.success(f"🟢 Sentiment: {sentiment} ({confidence:.2f}% confidence)")

        elif sentiment == "Bearish":
            st.error(f"🔴 Sentiment: {sentiment} ({confidence:.2f}% confidence)")

        else:
            st.info(f"🔵 Sentiment: {sentiment} ({confidence:.2f}% confidence)")


        # Probability chart
        probabilities = probs.detach().numpy()[0]

        df = pd.DataFrame({
            "Sentiment": ["Bearish", "Neutral", "Bullish"],
            "Probability": probabilities
        })

        st.subheader("Prediction Probabilities")

        st.bar_chart(df.set_index("Sentiment"))