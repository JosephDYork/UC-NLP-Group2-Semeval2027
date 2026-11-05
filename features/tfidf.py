import numpy as np
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddings:
    def __init__(self, min_df=2, ngram_range=(1, 3)):
        self.path = "embeddings/tfidf_embeddings.npy"
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=lambda tokens: tokens,
            preprocessor=lambda tokens: tokens,
            token_pattern=None,
            lowercase=False,
            min_df=min_df,
            ngram_range=ngram_range,
        )

    def generate(self, rows):
        return normalize(self.vectorizer.fit_transform(rows.context_tokens))

    def save(self, rows):
        np.save(self.path, self.generate(rows).toarray())

    def load(self):
        return np.load(self.path)
