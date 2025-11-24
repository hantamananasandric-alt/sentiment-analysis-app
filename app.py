import streamlit as st
import pandas as pd
import os
import sys
import time
from src.models import SentimentAnalyzer

# --- CONFIGURATION DU CHEMIN D'ACCÈS ---
# Ajoute le répertoire parent (la racine du projet) au chemin Python
# CECI EST CRUCIAL pour que les imports 'from src...' fonctionnent lors de l'exécution Streamlit.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- CHEMIN D'ACCÈS AUX DONNÉES ---
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'tweets_processed.csv')


# --- CACHE DES RESSOURCES LOURDES ---
# @st.cache_resource s'assure que le modèle BERT n'est chargé qu'une seule fois.
@st.cache_resource
def load_analyzer():
    """Charge et met en cache l'analyseur de sentiments."""
    # Le modèle peut prendre un moment à charger la première fois
    st.info("Chargement du modèle d'analyse de sentiment (BERT)... Veuillez patienter.")
    analyzer = SentimentAnalyzer()
    st.success("Modèle chargé!")
    return analyzer


@st.cache_data
def load_and_analyze_data():
    """Charge les données prétraitées et y applique l'analyse de sentiment."""

    if not os.path.exists(PROCESSED_DATA_PATH):
        st.error(f"Fichier de données non trouvé : {PROCESSED_DATA_PATH}")
        st.info("Veuillez d'abord exécuter python run_pipeline.py pour générer le fichier CSV.")
        return pd.DataFrame()

    df = pd.read_csv(PROCESSED_DATA_PATH)

    if df.empty:
        st.warning("Le DataFrame est vide.")
        return df

    st.info(f"Analyse de sentiment en cours sur {len(df)} tweets...")

    analyzer = load_analyzer()

    # Utilisation de la méthode batch
    results_df = analyzer.analyze_batch(df['cleaned_text'])

    df = pd.concat([df, results_df], axis=1)

    return df


# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Dashboard d'Analyse de Sentiment (Tweets)", layout="wide")

st.title("📊 Tableau de bord d'Analyse de Sentiment Avancée")

# --- SECTION 1 : ANALYSE EN TEMPS RÉEL (Interactive) ---
st.header("Analyse de Sentiment en Temps Réel 💬")

input_text = st.text_area("Entrez le texte à analyser ici :", height=100)

if st.button("Analyser le Texte"):
    if input_text:
        analyzer = load_analyzer()

        with st.spinner('Analyse du sentiment en cours...'):
            label, score = analyzer.analyze(input_text)

        st.success("Analyse Terminée!")

        col_res1, col_res2 = st.columns(2)
        note = label.split()[0]

        col_res1.metric("Note de Sentiment (1-5)", f"{note} ⭐")
        col_res2.metric("Niveau de Confiance", f"{score * 100:.2f}%")

        if int(note) > 3:
            st.write(f"Ce texte est considéré comme : **{label} (Positif)**")
        elif int(note) < 3:
            st.write(f"Ce texte est considéré comme : **{label} (Négatif)**")
        else:
            st.write(f"Ce texte est considéré comme : **{label} (Neutre)**")
    else:
        st.warning("Veuillez entrer du texte pour l'analyse.")

st.markdown("---")

# --- SECTION 2 : TABLEAU DE BORD (Mode Batch) ---
st.header("Résultats de la Dernière Collecte de Données")

# Bouton de relance (ATTENTION à l'indentation ici)
if st.button("Recharger et Analyser les Données"):
    # Ceci est le bloc indenté attendu par Python!
    st.cache_data.clear()

# Chargement et analyse des données batch
df_final = load_and_analyze_data()

if not df_final.empty:
    # --- Métriques Clés (KPIs) ---
    col1, col2, col3 = st.columns(3)

    avg_sentiment = df_final['sentiment_note'].mean()

    col1.metric("Nombre Total de Tweets", len(df_final))
    col2.metric("Note de Sentiment Moyenne (1-5)", f"{avg_sentiment:.2f} ⭐")
    col3.metric("Tweets Positifs (>3 étoiles)",
                f"{len(df_final[df_final['sentiment_note'] > 3]) / len(df_final) * 100:.1f}%")

    st.markdown("---")

    # --- Visualisation 1 : Distribution des Sentiments ---
    st.subheader("Distribution des Notes de Sentiment")
    sentiment_counts = df_final['sentiment_note'].value_counts().sort_index()
    st.bar_chart(sentiment_counts)

    # --- Affichage du Tableau ---
    st.subheader("Détails des Tweets Analysés")
    st.dataframe(df_final[['text', 'cleaned_text', 'sentiment_note', 'sentiment_score', 'like_count', 'created_at']])