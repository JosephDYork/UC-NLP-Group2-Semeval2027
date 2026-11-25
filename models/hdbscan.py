import numpy as np
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances

from .base import BaseModel


def fit_hdbscan_labels(
    vectors,
    min_cluster_size,
    min_samples,
    cluster_selection_method,
    allow_single_cluster,
):
    return (
        HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed",
            cluster_selection_method=cluster_selection_method,
            allow_single_cluster=allow_single_cluster,
        )
        .fit(cosine_distances(vectors).astype(np.float64))
        .labels_
    )


class HdbscanModel(BaseModel):
    display_name = "HDBSCAN"

    def __init__(
        self,
        min_cluster_size=5,
        min_samples=1,
        min_cluster_ratio=None,
        cluster_selection_method="eom",
        allow_single_cluster=True,
        merge_noise=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.min_cluster_ratio = min_cluster_ratio
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.merge_noise = merge_noise

    @classmethod
    def search_space(cls, quick=False, n_dims=None):
        common = cls.grid(
            cluster_selection_method=["eom"] if quick else ["eom", "leaf"],
            allow_single_cluster=[False] if quick else [True, False],
            merge_noise=[True] if quick else [False, True],
        )
        if quick:
            sizing = [{"min_cluster_ratio": r} for r in (0.04, 0.08)]
        else:
            sizing = [{"min_cluster_ratio": r} for r in (0.02, 0.04, 0.06, 0.08, 0.12)]
            sizing += cls.grid(
                min_cluster_ratio=[None],
                min_cluster_size=[3, 5, 10],
                min_samples=[1, 3, 5],
            )
        return [{**s, **c} for s in sizing for c in common]

    def fit_words(self, words):
        tasks, keys = {}, {}
        for word, positions in words.items():
            size = self.cluster_size_for(len(positions))
            if len(positions) < size:
                keys[word] = None
                continue
            min_samples = (
                size if self.min_cluster_ratio is not None else self.min_samples
            )
            key = (
                "hdbscan",
                word,
                size,
                min_samples,
                self.cluster_selection_method,
                self.allow_single_cluster,
            )
            tasks[key] = lambda word=word, size=size, min_samples=min_samples: (
                fit_hdbscan_labels,
                self.word_vectors(word),
                size,
                min_samples,
                self.cluster_selection_method,
                self.allow_single_cluster,
            )
            keys[word] = key

        labels = self.cached(tasks)
        self.models_by_word = {
            word: None if key is None else labels[key] for word, key in keys.items()
        }

    def cluster_size_for(self, n_rows):
        if self.min_cluster_ratio is None:
            return self.min_cluster_size
        return max(int(self.min_cluster_ratio * n_rows), 2)

    def predict_word(self, word, vectors, index):
        training_labels = self.models_by_word[word]
        if training_labels is None:
            return np.zeros(len(vectors), dtype=int)
        distances = cosine_distances(vectors, self.word_vectors(word))
        labels = training_labels[distances.argmin(axis=1)]
        return [
            label if label >= 0 else ("noise" if self.merge_noise else f"noise:{i}")
            for label, i in zip(labels, index)
        ]
