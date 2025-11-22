# src/preprocessing.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Assure-toi d'avoir téléchargé les ressources NLTK une fois :
# >>> import nltk
# >>> nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')

class TweetPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('french')) | set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_tweet(self, tweet: str) -> str:
        if not isinstance(tweet, str):
            return ""
        t = tweet.lower()
        t = re.sub(r'http\S+', '', t)          # urls
        t = re.sub(r'@\w+', '', t)            # mentions
        t = re.sub(r'#\w+', '', t)            # hashtags (on peut garder le mot si on veut)
        t = re.sub(r'[^\w\s]', ' ', t)        # ponctuation -> espace
        t = re.sub(r'\d+', '', t)             # chiffres
        tokens = word_tokenize(t)
        tokens = [self.lemmatizer.lemmatize(tok) for tok in tokens
                  if tok not in self.stop_words and len(tok) > 2]
        return " ".join(tokens)
