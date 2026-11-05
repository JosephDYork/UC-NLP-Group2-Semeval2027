import numpy as np
from sklearn.mixture import GaussianMixture

from .base import BaseModel


class GaussianMixtureModel(BaseModel):
    def __init__(
        self, max_components=8, covariance_type="diag", reg_covar=1e-5, **kwargs
    ):
        super().__init__(**kwargs)
        self.max_components = max_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar

    def build_gmm(self, n_components):
        return GaussianMixture(
            n_components=n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
        )

    def fit(self, train):
        vectors = self.reduce(self.vectorize(train, fit=True), fit=True)
        for word in train.word.unique():
            word_vectors = vectors[train.word.eq(word).to_numpy()]
            candidates = [
                self.build_gmm(n).fit(word_vectors)
                for n in range(1, min(self.max_components, len(word_vectors)) + 1)
            ]
            self.models_by_word[word] = min(
                candidates, key=lambda model: model.bic(word_vectors)
            )

        return self

    def predict(self, rows):
        vectors = self.reduce(self.vectorize(rows), fit=False)
        predictions = np.empty(len(rows), dtype=object)
        for word in rows.word.unique():
            mask = rows.word.eq(word).to_numpy()
            labels = self.models_by_word[word].predict(vectors[mask])
            predictions[mask] = [f"{word}:{label}" for label in labels]

        return predictions.tolist()
