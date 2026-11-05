import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .base import BaseModel


class KmeansModel(BaseModel):
    def __init__(self, max_k=8, max_iter=100, min_silhouette=0.02, **kwargs):
        super().__init__(**kwargs)
        self.max_k = max_k
        self.max_iter = max_iter
        self.min_silhouette = min_silhouette

    def fit(self, train):
        vectors = self.reduce(self.vectorize(train, fit=True), fit=True)
        for word in train.word.unique():
            word_vectors = vectors[train.word.eq(word).to_numpy()]
            self.models_by_word[word] = self.build_kmeans(
                self.choose_k(word_vectors)
            ).fit(word_vectors)
        return self

    def predict(self, rows):
        vectors = self.reduce(self.vectorize(rows), fit=False)
        predictions = np.empty(len(rows), dtype=object)
        for word in rows.word.unique():
            mask = rows.word.eq(word).to_numpy()
            labels = self.models_by_word[word].predict(vectors[mask])
            predictions[mask] = [f"{word}:{label}" for label in labels]
        return predictions.tolist()

    def choose_k(self, vectors):
        best_k, best_score = 1, self.min_silhouette
        for n_clusters in range(2, min(self.max_k, vectors.shape[0] - 1) + 1):
            labels = self.build_kmeans(n_clusters).fit_predict(vectors)
            if len(np.unique(labels)) >= 2:
                score = silhouette_score(vectors, labels, metric="cosine")
                if score > best_score:
                    best_k, best_score = n_clusters, score
        return best_k

    def build_kmeans(self, n_clusters):
        return KMeans(
            n_clusters=n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
