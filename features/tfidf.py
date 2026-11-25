import scipy.sparse
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddings:
    def __init__(self, min_df=2, ngram_range=(1, 1)):
        self.path = "embeddings/tfidf_embeddings.npz"
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
        scipy.sparse.save_npz(self.path, self.generate(rows).tocsr())

    def load(self):
        return scipy.sparse.load_npz(self.path).tocsr()
