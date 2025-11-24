# src/models.py

from transformers import pipeline
import pandas as pd
import numpy as np


class SentimentAnalyzer:
    """Charge et exécute le pipeline Transformers pour l'analyse de sentiment."""

    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        self.model_name = model_name

        # Initialise le pipeline.
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            tokenizer=self.model_name
        )

    def analyze(self, text: str):
        """Retourne le label et le score du sentiment pour un seul texte."""
        if not text:
            # Retourne une valeur neutre si le texte est vide
            return "3 stars", 0.5

        result = self.pipeline(text)[0]
        return result['label'], result['score']

    def analyze_batch(self, texts: pd.Series) -> pd.DataFrame:
        """
        Analyse une série de textes et retourne un DataFrame avec les résultats.
        """
        # Exécute l'analyse sur la liste complète des textes
        results = self.pipeline(texts.tolist(), batch_size=32)

        labels = [res['label'] for res in results]
        scores = [res['score'] for res in results]

        # Le label est au format "X stars" (ex: "5 stars"), on veut le chiffre X (5)
        notes = [int(label.split()[0]) for label in labels]

        return pd.DataFrame({
            'sentiment_label': labels,
            'sentiment_score': scores,
            'sentiment_note': notes
        })