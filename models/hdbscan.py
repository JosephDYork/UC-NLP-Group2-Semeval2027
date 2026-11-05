import numpy as np
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

from .base import BaseModel


class HdbscanModel(BaseModel):
    def __init__(
        self,
        min_cluster_size=5,
        min_samples=1,
        min_cluster_ratio=None,
        normalize_reduced=True,
        cluster_selection_method="eom",
        allow_single_cluster=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.min_cluster_ratio = min_cluster_ratio
        self.normalize_reduced = normalize_reduced
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.training_vectors_by_word = {}

    def fit(self, train):
        vectors = self.reduce(self.vectorize(train, fit=True), fit=True)
        vectors = normalize(vectors) if self.normalize_reduced else vectors
        for word in train.word.unique():
            word_vectors = vectors[train.word.eq(word).to_numpy()]
            size = self.cluster_size_for(word_vectors)
            if len(word_vectors) < size:
                self.models_by_word[word] = None
                continue

            self.models_by_word[word] = HDBSCAN(
                min_cluster_size=size,
                min_samples=size
                if self.min_cluster_ratio is not None
                else self.min_samples,
                metric="precomputed",
                cluster_selection_method=self.cluster_selection_method,
                allow_single_cluster=self.allow_single_cluster,
            ).fit(cosine_distances(word_vectors).astype(np.float64))
            self.training_vectors_by_word[word] = word_vectors
        return self

    def cluster_size_for(self, vectors):
        if self.min_cluster_ratio is None:
            return self.min_cluster_size
        return max(int(self.min_cluster_ratio * len(vectors)), 2)

    def predict(self, rows):
        vectors = self.reduce(self.vectorize(rows), fit=False)
        vectors = normalize(vectors) if self.normalize_reduced else vectors
        predictions = np.empty(len(rows), dtype=object)
        for word in rows.word.unique():
            mask = rows.word.eq(word).to_numpy()
            model = self.models_by_word.get(word)
            if model is None:
                labels = np.zeros(mask.sum(), dtype=int)
            else:
                distances = cosine_distances(
                    vectors[mask], self.training_vectors_by_word[word]
                )
                nearest_training_rows = distances.argmin(axis=1)
                labels = model.labels_[nearest_training_rows]
            predictions[mask] = [
                f"{word}:{label}" if label >= 0 else f"{word}:noise:{index}"
                for label, index in zip(labels, rows.index.to_numpy()[mask])
            ]
        return predictions.tolist()
