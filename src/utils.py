# src/utils.py (MODIFICATION)

import time
from tweepy import TooManyRequests
import logging
from requests.exceptions import ConnectionError  # <-- AJOUTER CET IMPORT
# NOUVELLES LIGNES CORRECTES:
from urllib3.exceptions import MaxRetryError
from http.client import RemoteDisconnected # <-- CECI EST L'IMPORT CORRECT


# ... (le reste des imports et la configuration du logging)

def safe_twitter_call(func, *args, **kwargs):
    """
    Exécute une fonction d'appel à l'API Tweepy avec gestion des limites et des erreurs de connexion.
    """
    max_retries = 3
    delay_minutes = 15

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)

        except TooManyRequests as e:
            logging.warning(
                f"Limite d'API atteinte (Tentative {attempt + 1}/{max_retries}). Attente de {delay_minutes} minutes...")
            if attempt < max_retries - 1:
                time.sleep(delay_minutes * 60)
            else:
                logging.error("Échec après plusieurs tentatives de limite. Arrêt.")
                raise e

        except (ConnectionError, RemoteDisconnected, MaxRetryError) as e:  # <-- NOUVEAU BLOC DE GESTION
            logging.warning(
                f"Erreur de connexion/serveur inattendue (Tentative {attempt + 1}/{max_retries}). Réessai dans 10 secondes...")
            if attempt < max_retries - 1:
                # Si c'est une erreur de connexion, on n'attend pas 15 minutes, mais juste 10 secondes.
                time.sleep(10)
            else:
                logging.error("Échec persistant de la connexion. Arrêt.")
                raise e

        except Exception as e:
            # Gérer d'autres erreurs inattendues ici (réseau, authentification)
            logging.error(f"Erreur inattendue lors de l'appel API: {e}")
            raise e