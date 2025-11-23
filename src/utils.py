# src/utils.py

import time
from tweepy import TooManyRequests
import logging

# Configuration de base pour les logs (pour voir les messages d'attente)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def safe_twitter_call(func, *args, **kwargs):
    """
    Exécute une fonction d'appel à l'API Tweepy avec gestion des limites (TooManyRequests).
    Réessaie après 15 minutes d'attente.
    """

    # Nous utilisons une boucle pour réessayer au cas où la première attente ne suffirait pas
    max_retries = 3
    delay_minutes = 15

    for attempt in range(max_retries):
        try:
            # Tente d'exécuter la fonction Tweepy (par exemple, client.search_recent_tweets)
            return func(*args, **kwargs)

        except TooManyRequests as e:
            logging.warning(f"Limite d'API atteinte (Tentative {attempt + 1}/{max_retries}).")
            logging.warning(f"Attente de {delay_minutes} minutes...")

            if attempt < max_retries - 1:
                # Attendre (15 minutes * 60 secondes)
                time.sleep(delay_minutes * 60)
            else:
                logging.error("Échec après plusieurs tentatives. Levée de l'exception TooManyRequests.")
                raise e

        except Exception as e:
            # Gérer d'autres erreurs inattendues ici (réseau, authentification)
            logging.error(f"Erreur inattendue lors de l'appel API: {e}")
            raise e