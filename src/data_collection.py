# src/data_collection.py

import os
from dotenv import load_dotenv
import tweepy
import pandas as pd
from src.utils import safe_twitter_call  # Import de la fonction de gestion de limite

# Charge le fichier .env à l'exécution de ce script
load_dotenv()


def get_twitter_client() -> tweepy.Client:
    """Initialise et retourne le client Tweepy en utilisant le Bearer Token."""
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")

    if not bearer_token:
        raise ValueError("Le jeton TWITTER_BEARER_TOKEN est manquant dans le fichier .env.")

    return tweepy.Client(bearer_token)


def collect_tweets_to_dataframe(query: str, max_results: int = 100) -> pd.DataFrame:
    """
    Collecte des tweets récents basés sur une requête et les retourne dans un DataFrame.
    """
    client = get_twitter_client()

    try:
        # Utilisation de la fonction sécurisée pour gérer les limites de l'API
        response = safe_twitter_call(
            client.search_recent_tweets,
            query,
            max_results=max_results,
            # Champs de données à récupérer (essentiels pour l'analyse)
            tweet_fields=["created_at", "lang", "public_metrics"]
        )

    except Exception as e:
        print(f"Erreur fatale lors de la collecte des données: {e}")
        return pd.DataFrame()  # Retourne un DataFrame vide en cas d'échec

    if response.data:
        # Convertir les données Tweepy en une liste de dictionnaires pour Pandas
        tweets_list = []
        for tweet in response.data:
            tweets_list.append({
                'id': tweet.id,
                'text': tweet.text,
                'created_at': tweet.created_at,
                'lang': tweet.lang,
                'retweet_count': tweet.public_metrics['retweet_count'],
                'like_count': tweet.public_metrics['like_count']
            })

        return pd.DataFrame(tweets_list)
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    # Test de la fonction de collecte
    test_query = "analyse de sentiment"
    df = collect_tweets_to_dataframe(test_query, max_results=50)

    if not df.empty:
        print(f"\nCollecte réussie de {len(df)} tweets pour '{test_query}'.")
        print("Aperçu des données :")
        print(df.head())
    else:
        print("La collecte n'a retourné aucun tweet ou a échoué.")