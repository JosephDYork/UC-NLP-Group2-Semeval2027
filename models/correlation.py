# BSD-3-Clause, Copyright (c) 2026, Garrafao. Same objective and search starts;

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseModel

MAX_SWEEPS = 100


def correlation_loss(weights, labels):
    rows, cols = np.triu_indices(len(weights), k=1)
    w = weights[rows, cols]
    same = labels[rows] == labels[cols]
    return float(np.where(same, np.maximum(-w, 0), np.maximum(w, 0)).sum())


def move_search(weights, labels, rng):
    n = len(weights)
    labels = np.unique(labels, return_inverse=True)[1]
    onehot = np.zeros((n, n + 1))
    onehot[np.arange(n), labels] = 1
    to_cluster = weights @ onehot
    sizes = np.bincount(labels, minlength=n + 1)
    n_open = labels.max() + 2
    for _ in range(MAX_SWEEPS):
        moved = False
        for i in rng.permutation(n):
            current = labels[i]
            row = to_cluster[i, :n_open]
            target = int(row.argmax())
            if target != current and row[target] - row[current] > 1e-9:
                labels[i] = target
                to_cluster[:, current] -= weights[:, i]
                to_cluster[:, target] += weights[:, i]
                sizes[current] -= 1
                sizes[target] += 1
                if sizes[target] == 1:
                    n_open += 1
                moved = True
        if not moved:
            break
    return np.unique(labels, return_inverse=True)[1]


def positive_components(weights, members=None):
    adjacency = weights >= 0
    np.fill_diagonal(adjacency, False)
    if members is not None:
        adjacency = adjacency[np.ix_(members, members)]
    return connected_components(csr_matrix(adjacency), directed=False)[1]


def fit_correlation_labels(vectors, threshold, threshold_quantile, s, seed):
    n = len(vectors)
    if n < 2:
        return np.zeros(n, dtype=int)
    similarities = cosine_similarity(vectors)
    if threshold_quantile is not None:
        threshold = np.quantile(
            similarities[np.triu_indices(n, k=1)], threshold_quantile
        )
    weights = similarities - threshold
    np.fill_diagonal(weights, 0.0)

    rng = np.random.default_rng(seed)
    starts = [positive_components(weights)]
    starts += [rng.integers(k, size=n) for k in range(2, s) for _ in range(2)]
    best_labels, best_loss = None, np.inf
    for start in starts:
        labels = move_search(weights, start, rng)
        loss = correlation_loss(weights, labels)
        if loss < best_loss:
            best_labels, best_loss = labels, loss

    labels = np.empty(n, dtype=int)
    next_label = 0
    for cluster in np.unique(best_labels):
        members = np.flatnonzero(best_labels == cluster)
        parts = positive_components(weights, members)
        labels[members] = parts + next_label
        next_label += parts.max() + 1
    return labels


class CorrelationClusteringModel(BaseModel):
    display_name = "Correlation"

    def __init__(self, threshold=0.575, threshold_quantile=None, s=10, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.threshold_quantile = threshold_quantile
        self.s = s

    @classmethod
    def search_space(cls, quick=False, n_dims=None):
        if quick:
            return cls.grid(threshold_quantile=[0.5], s=[5])
        absolute = [
            {"threshold": float(round(x, 3)), "threshold_quantile": None}
            for x in np.arange(0.45, 0.701, 0.025)
        ]
        relative = [
            {"threshold_quantile": float(round(q, 2))}
            for q in np.arange(0.40, 0.601, 0.05)
        ]
        return [{**t, "s": 10} for t in absolute + relative]

    def fit_words(self, words):
        threshold = None if self.threshold_quantile is not None else self.threshold
        keys = {
            word: ("correlation", word, threshold, self.threshold_quantile, self.s)
            for word in words
        }
        labels = self.cached(
            {
                key: lambda word=word: (
                    fit_correlation_labels,
                    self.word_vectors(word),
                    threshold,
                    self.threshold_quantile,
                    self.s,
                    self.random_state,
                )
                for word, key in keys.items()
            }
        )
        self.models_by_word = {word: labels[key] for word, key in keys.items()}

    def predict_word(self, word, vectors, index):
        similarities = cosine_similarity(vectors, self.word_vectors(word))
        return self.models_by_word[word][similarities.argmax(axis=1)]
