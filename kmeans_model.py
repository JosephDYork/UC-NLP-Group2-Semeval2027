import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


class KmeansBaselineModel:
    def __init__(self, max_k=8, max_iter=100, min_df=2, random_state=42):
        self.max_k = max_k
        self.min_df = min_df
        self.max_iter = max_iter
        self.random_state = random_state
        self.models_by_word = {}
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=lambda tokens: tokens,
            preprocessor=lambda tokens: tokens,
            token_pattern=None,
            min_df=min_df
        )

    def fit(self, train):
        X = normalize(self.vectorizer.fit_transform(train.context_tokens))
        for word in train.word.unique():
            x2 = X[train.word.eq(word).to_numpy()]
            k = self.choose_k(x2)
            model = self.build_kmeans(k).fit(x2)
            self.models_by_word[word] = model

        return self

    def predict(self, rows):
        x = normalize(self.vectorizer.transform(rows.context_tokens))
        predictions = np.empty(len(rows), dtype=object)

        for word in rows.word.unique():
            mask = rows.word.eq(word).to_numpy()
            model = self.models_by_word.get(word)
            labels = model.predict(x[mask])
            predictions[mask] = [f"{word}:{label}" for label in labels]

        return predictions.tolist()

    def choose_k(self, x):
        best_k, best_score, max_k = 1, 0, min(self.max_k, x.shape[0] - 1)
        for k in range(2, max_k + 1):
            labels = self.build_kmeans(k).fit_predict(x)
            if not len(np.unique(labels)) < 2:
                score = silhouette_score(x, labels, metric="cosine")
                if score > best_score:
                    best_k = k
                    best_score = score

        return best_k

    def build_kmeans(self, k):
        return KMeans(
            n_clusters=k,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
