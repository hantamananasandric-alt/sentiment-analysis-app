# src/models.py
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

class SentimentAnalyzer:
    def __init__(self, method='transformers'):
        self.method = method
        self.vectorizer = None
        self.model = None
        if method == 'transformers':
            # utilise un modèle pré-entraîné (ex: cardiffnlp/twitter-roberta-base-sentiment-latest)
            self.pipeline = pipeline("sentiment-analysis",
                                      model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                      tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                      return_all_scores=True)

    def predict(self, text: str) -> str:
        if self.method == 'transformers':
            results = self.pipeline(text[:512])[0]
            # choisir label dominant
            best = max(results, key=lambda x: x['score'])
            map_label = {'POS': 'positif', 'NEG': 'négatif', 'NEU': 'neutre'}
            # selon le modèle, label peut être 'positive'/'negative'/'neutral' ou 'LABEL_0' etc.
            lbl = best.get('label', '')
            # simple mapping robust
            if 'POS' in lbl.upper() or 'POSITIVE' in lbl.upper() or 'LABEL_2' in lbl:
                return 'positif'
            if 'NEG' in lbl.upper() or 'NEGATIVE' in lbl.upper() or 'LABEL_0' in lbl:
                return 'négatif'
            return 'neutre'
        else:
            if not self.vectorizer or not self.model:
                raise RuntimeError("Modèle ML non entraîné")
            vec = self.vectorizer.transform([text])
            return self.model.predict(vec)[0]

    def train_ml_model(self, X_texts, y_labels):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X = self.vectorizer.fit_transform(X_texts)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y_labels)

    def save_ml(self, path='models/ml_sentiment.joblib'):
        joblib.dump({'vectorizer': self.vectorizer, 'model': self.model}, path)

    def load_ml(self, path='models/ml_sentiment.joblib'):
        loaded = joblib.load(path)
        self.vectorizer = loaded['vectorizer']
        self.model = loaded['model']
