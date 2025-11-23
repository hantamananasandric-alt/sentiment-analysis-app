import tweepy
import os
from dotenv import load_dotenv

# Charger les variables du fichier .env à la racine
# Cette ligne est cruciale pour que os.environ.get fonctionne
load_dotenv()


def test_twitter_connection():
    """Test de connexion à l'API Twitter (X) en utilisant le Bearer Token."""

    # 1. Récupération des secrets depuis l'environnement
    BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")

    if not BEARER_TOKEN:
        print("❌ Erreur : Le jeton TWITTER_BEARER_TOKEN est manquant dans le fichier .env.")
        return False

    try:
        # 2. Création de l'API client (méthode recommandée pour l'API v2)
        client = tweepy.Client(bearer_token=BEARER_TOKEN)

        # 3. Test de recherche (API v2)
        # Remplacez "Python" par un terme générique pour le test
        tweets = client.search_recent_tweets("IA France", max_results=10)

        if tweets.data:
            print("✅ Connexion Twitter API (v2) réussie!")
            print(f"📊 {len(tweets.data)} tweets récupérés.")

            # Afficher le premier tweet
            first_tweet = tweets.data[0]
            print(f"🐦 Premier tweet: {first_tweet.text[:100]}...")

            return True
        else:
            print("❌ Connexion réussie, mais aucun tweet n'a été récupéré pour ce terme.")
            return False

    except Exception as e:
        # Ceci peut capturer les erreurs d'authentification (jeton invalide)
        print(f"❌ Erreur de connexion (Jetons invalides ?): {e}")
        return False


if __name__ == "__main__":
    test_twitter_connection()