import numpy as np
import scipy.sparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .base import BaseModel


class ColumnSubsetKMeans:
    def __init__(self, model, columns):
        model.cluster_centers_ = np.ascontiguousarray(
            model.cluster_centers_[:, columns]
        )
        model.n_features_in_ = len(columns)
        self.model = model
        self.columns = columns

    def predict(self, vectors):
        return self.model.predict(vectors[:, self.columns])


def fit_kmeans(vectors, n_clusters, max_iter, random_state):
    model = KMeans(
        n_clusters=n_clusters, max_iter=max_iter, random_state=random_state
    ).fit(vectors)
    score = None
    if n_clusters >= 2 and len(np.unique(model.labels_)) >= 2:
        score = silhouette_score(vectors, model.labels_, metric="cosine")
    if scipy.sparse.issparse(vectors):
        model = ColumnSubsetKMeans(model, np.unique(vectors.tocsr().indices))
    return model, score


class KmeansModel(BaseModel):
    display_name = "K-means"

    def __init__(self, max_k=8, max_iter=100, min_silhouette=0.02, **kwargs):
        super().__init__(**kwargs)
        self.max_k = max_k
        self.max_iter = max_iter
        self.min_silhouette = min_silhouette

    @classmethod
    def search_space(cls, quick=False, n_dims=None):
        if quick:
            return cls.grid(max_k=[8], min_silhouette=[0.0, 0.05])
        else:
            return cls.grid(
                max_k=[3, 4, 6, 8],
                min_silhouette=[0.0, 0.1, 0.2],
                max_iter=[300],
            )

    def fit_words(self, words):
        fits = self.cached(
            {
                self.key(word, k): self.task(word, k)
                for word, positions in words.items()
                for k in self.candidate_ks(len(positions))
            }
        )
        best_k = {
            word: self.choose_k(word, len(positions), fits)
            for word, positions in words.items()
        }
        fits.update(
            self.cached(
                {
                    self.key(word, 1): self.task(word, 1)
                    for word, k in best_k.items()
                    if k == 1
                }
            )
        )
        self.models_by_word = {
            word: fits[self.key(word, k)][0] for word, k in best_k.items()
        }

    def predict_word(self, word, vectors, index):
        return self.models_by_word[word].predict(vectors)

    def task(self, word, n_clusters):
        return lambda: (
            fit_kmeans,
            self.word_vectors(word),
            n_clusters,
            self.max_iter,
            self.random_state,
        )

    def key(self, word, n_clusters):
        return ("kmeans", word, n_clusters, self.max_iter)

    def candidate_ks(self, n_rows):
        return range(2, min(self.max_k, n_rows - 1) + 1)

    def choose_k(self, word, n_rows, fits):
        best_k, best_score = 1, self.min_silhouette
        for n_clusters in self.candidate_ks(n_rows):
            score = fits[self.key(word, n_clusters)][1]
            if score is not None and score > best_score:
                best_k, best_score = n_clusters, score

        return best_k
