import numpy as np


class RandomBaselineModel:
    def __init__(self, max_k=8, random_state=42, **kwargs):
        self.max_k = max_k
        self.random_state = random_state
        self.k_by_word = {}

    def fit(self, train):
        self.k_by_word = {
            word: min(self.max_k, max(1, len(group)))
            for word, group in train.groupby("word")
        }
        return self

    def predict(self, rows):
        rng = np.random.default_rng(self.random_state)
        predictions = []

        for row in rows.itertuples():
            k = self.k_by_word.get(row.word, 1)
            predictions.append(f"{row.word}:{rng.integers(k)}")

        return predictions
