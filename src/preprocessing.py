# src/preprocessing.py

import re
import nltk
from nltk.corpus import stopwords
import spacy

# --- CHARGEMENT DU MODÈLE SPACY ---
# S'assurer que le modèle est chargé une seule fois
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    print("Modèle spaCy 'fr_core_news_sm' non trouvé. Veuillez l'installer.")
    exit()


class TweetPreprocessor:
    """Nettoie et normalise le texte pour l'analyse de sentiment."""

    def __init__(self):
        # Stop words en français et en anglais
        self.stop_words = set(stopwords.words('french')) | set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        # 1. Mise en minuscules
        text = text.lower()

        # 2. Suppression des liens web
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # 3. Suppression des mentions (@user) et hashtags (#tag)
        text = re.sub(r'@\w+|#', '', text)

        # 4. Suppression de la ponctuation et des caractères spéciaux (sauf espaces)
        text = re.sub(r'[^a-z\s]', '', text)

        # 5. Tokenisation, lemmatisation (via spaCy) et suppression des stop words
        doc = nlp(text)
        tokens = [
            token.lemma_ for token in doc
            if token.text not in self.stop_words and token.is_alpha
        ]

        return " ".join(tokens)