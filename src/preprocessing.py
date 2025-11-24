# src/preprocessing.py

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
import spacy
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CHARGEMENT DU MODÈLE SPACY ---
# Le chargement de ce modèle est lourd, il doit être fait une seule fois
try:
    # Utilisez le modèle français pour l'efficacité sur les tweets francophones
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    logging.error("Modèle spaCy 'fr_core_news_sm' non trouvé. Veuillez l'installer.")
    exit()


class TweetPreprocessor:
    """Nettoie et normalise le texte pour l'analyse de sentiment, incluant lemmatisation."""

    def __init__(self):
        # Stop words en français et en anglais
        self.french_stopwords = set(stopwords.words('french'))
        self.english_stopwords = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Applique les étapes de nettoyage à une seule chaîne de caractères."""
        if not isinstance(text, str):
            return ""  # Gérer les valeurs non-chaînes

        # 1. Mise en minuscules
        text = text.lower()

        # 2. Suppression des liens web, des mentions et des retweets
        text = re.sub(r'http\S+|www\S+|https\S+|@\w+|rt', '', text, flags=re.MULTILINE)

        # 3. Suppression des chiffres et de la ponctuation résiduelle
        text = re.sub(r'[^a-z\s]', '', text)

        # 4. Lemmatisation (via spaCy) et suppression des stop words
        doc = nlp(text)

        tokens = [
            token.lemma_ for token in doc
            if token.text not in self.french_stopwords and
               token.text not in self.english_stopwords and
               token.is_alpha and
               len(token.text) > 2  # Supprime les mots très courts
        ]

        return " ".join(tokens)

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Applique la fonction de nettoyage à une colonne d'un DataFrame et retourne le DF."""
        if df.empty:
            return df

        logging.info(f"Démarrage du pré-traitement de la colonne '{text_column}'...")

        # Appliquer la méthode clean_text à la colonne 'text' et créer une nouvelle colonne
        df['cleaned_text'] = df[text_column].apply(self.clean_text)

        logging.info("Pré-traitement terminé. Nouvelle colonne 'cleaned_text' ajoutée.")
        return df


if __name__ == "__main__":
    # Test de la classe de prétraitement
    preprocessor = TweetPreprocessor()

    test_data = {
        'text': [
            "C'est un super produit ! J'adore ça ! Regardez le lien : https://exemple.com @user123",
            "Je déteste le mauvais service et les bugs. #problème"
        ],
        'lang': ['fr', 'fr']
    }

    df_test = pd.DataFrame(test_data)
    df_processed = preprocessor.preprocess_dataframe(df_test)

    print("\n--- Résultat du Pré-traitement ---")
    print(df_processed[['text', 'cleaned_text']])