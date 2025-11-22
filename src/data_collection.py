# src/data_collection.py
import pandas as pd
import os
try:
    import tweepy
except ImportError:
    tweepy = None

class TwitterCollector:
    def __init__(self, bearer_token=None):
        self.bearer_token = bearer_token
        if tweepy and bearer_token:
            self.client = tweepy.Client(bearer_token=bearer_token)
        else:
            self.client = None

    def get_recent_tweets(self, query, max_results=100):
        """Retourne un DataFrame: id,text,created_at"""
        rows = []
        if self.client:
            tweets = self.client.search_recent_tweets(query=query, max_results=max_results,
                                                      tweet_fields=['created_at','lang'])
            if tweets.data:
                for t in tweets.data:
                    if getattr(t, 'lang', None) == 'fr' or True:
                        rows.append({'id': t.id, 'text': t.text, 'created_at': t.created_at})
        else:
            # fallback : lire un CSV dans data/raw si présent
            csv_path = os.path.join('data', 'raw', 'tweets_sample.csv')
            if os.path.exists(csv_path):
                return pd.read_csv(csv_path)
            else:
                raise RuntimeError("Aucun client tweepy initialisé et pas de fichier data/raw/tweets_sample.csv")
        return pd.DataFrame(rows)
