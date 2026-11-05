from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class BaseModel:
    def __init__(
        self,
        min_df=2,
        random_state=42,
        feature_source=None,
        autoencoder=None,
        umap=None,
        normalize_vectors=True,
    ):
        self.min_df = min_df
        self.random_state = random_state
        self.feature_source = feature_source
        self.normalize_vectors = normalize_vectors
        self.embeddings = None
        self.reducers = tuple(
            reducer() if callable(reducer) else reducer
            for reducer in (autoencoder, umap)
            if reducer is not None
        )
        self.models_by_word = {}
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=lambda tokens: tokens,
            preprocessor=lambda tokens: tokens,
            token_pattern=None,
            min_df=min_df,
        )

    def vectorize(self, rows, fit=False):
        if self.feature_source is not None:
            if self.embeddings is None:
                self.embeddings = self.feature_source.load()
            vectors = self.embeddings[rows.index.to_numpy()]
            return normalize(vectors) if self.normalize_vectors else vectors

        if fit:
            return normalize(self.vectorizer.fit_transform(rows.context_tokens))

        return normalize(self.vectorizer.transform(rows.context_tokens))

    def reduce(self, vectors, fit=False):
        for reducer in self.reducers:
            vectors = (
                reducer.fit_transform(vectors) if fit else reducer.transform(vectors)
            )
        return vectors
