import pandas as pd
import requests
import os
import tweepy
from typing import List, Dict, Optional
import json
from tqdm import tqdm


class DataCollector:
    def __init__(self, config_path: str = "config/api_config.json"):
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        """Chargement de la configuration API"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {}
            print("⚠️  Fichier de configuration non trouvé")

    def download_sentiment140(self, output_path: str = "data/raw/sentiment140.csv"):
        """Téléchargement du dataset Sentiment140"""
        print("📥 Téléchargement de Sentiment140...")

        # URL du dataset (à adapter selon votre source)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/sentiment140/sentiment140.csv"

        try:
            # Téléchargement avec barre de progression
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f, tqdm(
                    desc="Sentiment140",
                    total=total_size,
                    unit='iB',
                    unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)

            # Chargement et formatage
            df = pd.read_csv(output_path, encoding='latin-1',
                             names=['target', 'ids', 'date', 'flag', 'user', 'text'])
            df['sentiment'] = df['target'].map({0: 'négatif', 4: 'positif'})
            df['language'] = 'en'

            # Sauvegarde formatée
            df[['text', 'sentiment', 'language']].to_csv(output_path, index=False)
            print(f"✅ Sentiment140 sauvegardé: {len(df)} tweets")

            return df

        except Exception as e:
            print(f"❌ Erreur téléchargement Sentiment140: {e}")
            return None

    def load_twitter_airline_sentiment(self):
        """Chargement du dataset Twitter Airline Sentiment"""
        print("📥 Chargement Twitter Airline Sentiment...")

        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/twitter-airline-sentiment.csv"

        try:
            df = pd.read_csv(url)
            df['sentiment'] = df['airline_sentiment'].map({
                'negative': 'négatif',
                'neutral': 'neutre',
                'positive': 'positif'
            })
            df['language'] = 'en'
            df['text'] = df['text'].astype(str)

            output_path = "data/raw/twitter_airline.csv"
            df[['text', 'sentiment', 'language']].to_csv(output_path, index=False)
            print(f"✅ Twitter Airline sauvegardé: {len(df)} tweets")

            return df

        except Exception as e:
            print(f"❌ Erreur chargement Twitter Airline: {e}")
            return None

    def create_french_dataset(self):
        """Création d'un dataset français de base"""
        print("🇫🇷 Création du dataset français...")

        french_tweets = [
            # Tweets positifs
            {"text": "J'adore ce produit, il est fantastique !", "sentiment": "positif"},
            {"text": "Superbe expérience utilisateur, je recommande", "sentiment": "positif"},
            {"text": "Excellent service client, très satisfait", "sentiment": "positif"},
            {"text": "Produit de grande qualité, bravo", "sentiment": "positif"},
            {"text": "Livraison rapide et emballage soigné", "sentiment": "positif"},

            # Tweets négatifs
            {"text": "Je déteste ce service, horrible", "sentiment": "négatif"},
            {"text": "Très déçu par la qualité, à éviter", "sentiment": "négatif"},
            {"text": "Service client inexistant, scandaleux", "sentiment": "négatif"},
            {"text": "Produit cassé à la réception, colère", "sentiment": "négatif"},
            {"text": "Pire achat de ma vie, nul", "sentiment": "négatif"},

            # Tweets neutres
            {"text": "Le produit est correct, sans plus", "sentiment": "neutre"},
            {"text": "Rien de spécial à signaler", "sentiment": "neutre"},
            {"text": "Commande reçue dans les délais", "sentiment": "neutre"},
            {"text": "Le temps est agréable aujourd'hui", "sentiment": "neutre"},
            {"text": "J'ai reçu le colis comme prévu", "sentiment": "neutre"}
        ]

        df = pd.DataFrame(french_tweets)
        df['language'] = 'fr'

        output_path = "data/raw/french_base.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Dataset français créé: {len(df)} tweets")

        return df

    def collect_twitter_data(self, query: str, max_tweets: int = 1000):
        """Collecte de données Twitter en temps réel (optionnel)"""
        if not self.config.get('twitter_bearer_token'):
            print("❌ Token Twitter non configuré")
            return None

        print(f"🐦 Collecte Twitter: {query}")

        try:
            client = tweepy.Client(bearer_token=self.config['twitter_bearer_token'])

            tweets_data = []
            for tweet in tweepy.Paginator(client.search_recent_tweets,
                                          query=query + " lang:fr -is:retweet",
                                          max_results=100).flatten(limit=max_tweets):
                tweets_data.append({
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'language': 'fr'
                })

            df = pd.DataFrame(tweets_data)
            output_path = f"data/raw/twitter_live_{query.replace(' ', '_')}.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Données Twitter collectées: {len(df)} tweets")

            return df

        except Exception as e:
            print(f"❌ Erreur collecte Twitter: {e}")
            return None

    def combine_all_datasets(self):
        """Combinaison de tous les datasets"""
        print("🔄 Combinaison des datasets...")

        datasets = []

        # Sentiment140
        sentiment140_path = "data/raw/sentiment140.csv"
        if os.path.exists(sentiment140_path):
            df_sentiment140 = pd.read_csv(sentiment140_path)
            datasets.append(df_sentiment140)

        # Twitter Airline
        airline_path = "data/raw/twitter_airline.csv"
        if os.path.exists(airline_path):
            df_airline = pd.read_csv(airline_path)
            datasets.append(df_airline)

        # Français de base
        french_path = "data/raw/french_base.csv"
        if os.path.exists(french_path):
            df_french = pd.read_csv(french_path)
            datasets.append(df_french)

        if not datasets:
            print("❌ Aucun dataset trouvé")
            return None

        # Combinaison
        combined_df = pd.concat(datasets, ignore_index=True)
        combined_df['source'] = 'combined'

        # Sauvegarde
        output_path = "data/processed/combined_dataset.csv"
        combined_df.to_csv(output_path, index=False)

        print(f"✅ Datasets combinés: {len(combined_df)} tweets")
        print(f"📊 Distribution: {combined_df['sentiment'].value_counts().to_dict()}")

        return combined_df


# Point d'entrée pour la collecte
if __name__ == "__main__":
    collector = DataCollector()

    # Téléchargement des datasets
    collector.download_sentiment140()
    collector.load_twitter_airline_sentiment()
    collector.create_french_dataset()

    # Combinaison
    combined_data = collector.combine_all_datasets()