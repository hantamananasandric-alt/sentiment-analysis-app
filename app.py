# app.py
import streamlit as st
import pandas as pd
from src.preprocessing import TweetPreprocessor
from src.models import SentimentAnalyzer

st.title("🐦 Analyse de sentiment — Demo")

tweet_input = st.text_area("Collez un tweet ici :", height=120)

analyzer = SentimentAnalyzer(method='transformers')
pre = TweetPreprocessor()

if st.button("Analyser"):
    if not tweet_input.strip():
        st.warning("Veuillez entrer un tweet.")
    else:
        cleaned = pre.clean_tweet(tweet_input)
        sentiment = analyzer.predict(cleaned)
        st.success(f"Sentiment détecté : **{sentiment}**")
