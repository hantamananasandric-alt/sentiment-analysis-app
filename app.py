# app.py

import streamlit as st
from src.preprocessing import TweetPreprocessor
from src.models import SentimentAnalyzer


# --- CACHE DES RESSOURCES LOURDES ---
# @st.cache_resource s'assure que le modèle n'est chargé qu'une seule fois.
@st.cache_resource
def load_analyzer():
    """Charge et met en cache l'analyseur de sentiments."""
    return SentimentAnalyzer()


@st.cache_resource
def load_preprocessor():
    """Charge et met en cache le préprocesseur."""
    return TweetPreprocessor()


# --- INITIALISATION ---
analyzer = load_analyzer()
preprocessor = load_preprocessor()

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Analyse de Sentiment Avancée", layout="centered")

st.title("🤖 Analyse de Sentiment avec BERT (Multilingue)")
st.markdown("Saisissez un texte pour évaluer le sentiment (Note de 1 à 5 étoiles).")

# Zone de saisie utilisateur
user_input = st.text_area("Texte à analyser :",
                          "J'aime vraiment ce nouveau produit, le support client est fantastique !",
                          height=150)

if st.button("Analyser"):
    if user_input:
        with st.spinner("Analyse en cours..."):

            # 1. Prétraitement (facultatif si le modèle BERT gère déjà bien le bruit)
            cleaned_text = preprocessor.clean_text(user_input)

            # 2. Analyse
            label, score = analyzer.analyze(user_input)  # On analyse le texte brut pour BERT

            # 3. Affichage des résultats

            note = int(label.split()[0])
            etoiles = "⭐" * note

            st.success(f"**Sentiment Détecté :** {etoiles} ({note}/5)")
            st.info(f"**Confiance (Score) :** {score:.4f}")

            with st.expander("Détails"):
                st.write(f"**Label brut du modèle :** {label}")
                st.write(f"**Texte après pré-traitement :** {cleaned_text}")
    else:
        st.warning("Veuillez saisir du texte.")