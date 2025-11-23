# src/models.py

from transformers import pipeline


class SentimentAnalyzer:
    """Charge et exécute le pipeline Transformers pour l'analyse de sentiment."""

    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        self.model_name = model_name

        # Initialise le pipeline. Le modèle sera téléchargé si ce n'est pas déjà fait.
        # Le chargement d'un gros modèle doit être géré par st.cache_resource dans app.py
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            tokenizer=self.model_name
        )

    def analyze(self, text: str):
        """Retourne le label et le score du sentiment."""
        if not text:
            return "Neutre", 0.0

        # Le pipeline retourne une liste d'un dictionnaire
        result = self.pipeline(text)[0]

        # Exemple de sortie : ('5 stars', 0.999)
        return result['label'], result['score']