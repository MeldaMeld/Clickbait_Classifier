import streamlit as st
import joblib
from collections import deque
import random

st.set_page_config(page_title="Clickbait Headline Classifier")

# Load model and vectorizer
model = joblib.load("models/model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

st.title("Clickbait Headline Classifier")
st.write("This app predicts whether a given headline is clickbait or not.")

example_headlines = [
    "The Top Beaches In The World, According To National Geographic",
    "10 Tricks That Will Make You Smarter Overnight",
    "President Donald Trump to Host Patriots at White House",
    "He Cheated. Now His Ex-Girlfriend Has Some Heartbreaking Questions",
    "China and Economic Reform: Xi Jinping’s Track Record"
]

if st.button("Show Random Example Headline"):
    st.info(random.choice(example_headlines))

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=5)

headline = st.text_input("Enter a headline:")

if st.button("Predict"):
    if headline.strip() == "":
        st.warning("Please enter a headline!")
    else:
        headline_tfidf = vectorizer.transform([headline])
        proba = model.predict_proba(headline_tfidf)[0][1]
        prediction = 1 if proba >= 0.5 else 0

        result_text = "Clickbait ✅" if prediction == 1 else "Normal Headline ❌"
        st.write(f"**Prediction:** {result_text}")
        st.caption(f"Clickbait probability: {proba:.2f}")

        st.session_state.history.appendleft(f"\"{headline}\" → {result_text} (p={proba:.2f})")

if st.session_state.history:
    st.write("### Recent Predictions")
    for item in st.session_state.history:
        st.write(item)
