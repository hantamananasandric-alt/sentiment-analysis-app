# run_pipeline.py

import pandas as pd
import os
from src.data_collection import collect_tweets_to_dataframe
from src.preprocessing import TweetPreprocessor

# --- PARAMÈTRES DE BASE ---
# Définissez ici les paramètres pour la collecte (facile à modifier)
SEARCH_QUERY = "\"analyse de sentiment\" #IA"
MAX_TWEETS = 100
OUTPUT_FILE_PATH = os.path.join('data', 'processed', 'tweets_processed.csv')


def run_data_pipeline():
    """
    Orchestre la collecte et le pré-traitement des données.
    """
    print(f"--- 1. Démarrage de la collecte pour la requête : '{SEARCH_QUERY}' ---")

    # 1. Collecte des données
    df_raw = collect_tweets_to_dataframe(query=SEARCH_QUERY, max_results=MAX_TWEETS)

    if df_raw.empty:
        print("❌ Échec de la collecte ou aucun tweet trouvé. Arrêt du pipeline.")
        return

    print(f"✅ Collecte réussie : {len(df_raw)} tweets bruts récupérés.")

    # 2. Prétraitement des données
    print("--- 2. Prétraitement du texte ---")

    preprocessor = TweetPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df_raw, text_column='text')

    # 3. Sauvegarde des résultats
    print(f"--- 3. Sauvegarde du fichier ---")

    # S'assurer que le dossier 'data/processed' existe
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

    # Sauvegarde du DataFrame dans le dossier 'data/processed'
    df_processed.to_csv(OUTPUT_FILE_PATH, index=False)

    print(f"✅ Pipeline terminé ! Données sauvegardées dans : {OUTPUT_FILE_PATH}")
    print(df_processed[['text', 'cleaned_text']].head())


if __name__ == "__main__":
    run_data_pipeline()