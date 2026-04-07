import streamlit as st
import torch
import os
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "finbert_model")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

labels = ["Bearish", "Neutral", "Bullish"]

# -------------------- UI --------------------

st.title("📈 Financial News Sentiment Analyzer")

# 🔹 Introduction Section
st.markdown("""
### 📌 About This App
This application uses a **FinBERT deep learning model** to analyze financial news and predict sentiment.

👉 It classifies text into:
- **📉 Negative (Bearish)** → Market may go down  
- **⚖️ Neutral** → No major impact  
- **📈 Positive (Bullish)** → Market may go up  

💡 This helps investors and analysts understand market mood quickly.
""")

# Input box
text = st.text_area("📝 Enter Financial News or Tweet")

# -------------------- Prediction --------------------

if st.button("Predict Sentiment"):

    if text.strip() == "":
        st.warning("⚠️ Please enter some financial text.")
    else:

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)

        prediction = torch.argmax(probs).item()

        sentiment = labels[prediction]

        confidence = probs[0][prediction].item() * 100

        # 🔹 Convert to user-friendly label
        if sentiment == "Bullish":
            display_text = "📈 Positive Sentiment (Market may rise)"
            st.success(f"{display_text} \n\nConfidence: {confidence:.2f}%")

        elif sentiment == "Bearish":
            display_text = "📉 Negative Sentiment (Market may fall)"
            st.error(f"{display_text} \n\nConfidence: {confidence:.2f}%")

        else:
            display_text = "⚖️ Neutral Sentiment (No strong impact)"
            st.info(f"{display_text} \n\nConfidence: {confidence:.2f}%")

        # 🔹 Show explanation
        st.markdown("### 📊 What this means:")
        if sentiment == "Bullish":
            st.write("The news indicates **positive market movement**, which may increase stock prices.")
        elif sentiment == "Bearish":
            st.write("The news indicates **negative market movement**, which may decrease stock prices.")
        else:
            st.write("The news is **balanced or unclear**, with no strong impact on the market.")

        # 🔹 Probability Chart
        probabilities = probs.detach().numpy()[0]

        df = pd.DataFrame({
            "Sentiment": ["Bearish", "Neutral", "Bullish"],
            "Probability": probabilities
        })

        st.subheader("📊 Prediction Probabilities")
        st.bar_chart(df.set_index("Sentiment"))