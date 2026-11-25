from sklearn.mixture import GaussianMixture

from .base import BaseModel


def fit_gmm(vectors, n_components, covariance_type, reg_covar, random_state):
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=random_state,
    ).fit(vectors)
    return model, model.bic(vectors)


class GaussianMixtureModel(BaseModel):
    display_name = "GMM"

    def __init__(
        self, max_components=8, covariance_type="diag", reg_covar=1e-5, **kwargs
    ):
        super().__init__(**kwargs)
        self.max_components = max_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar

    @classmethod
    def search_space(cls, quick=False, n_dims=None):
        if quick:
            return cls.grid(
                max_components=[8], covariance_type=["diag"], reg_covar=[1e-3]
            )
        space = cls.grid(
            max_components=[4, 8],
            covariance_type=["diag", "spherical", "tied", "full"],
            reg_covar=[1e-3, 1e-2, 1e-1, 1.0],
        )

        if n_dims is not None and n_dims > 32:
            space = [p for p in space if p["covariance_type"] != "full"]

        return space

    def key(self, word, n_components):
        return ("gmm", word, n_components, self.covariance_type, self.reg_covar)

    def fit_words(self, words):
        candidates = {
            word: range(1, min(self.max_components, len(positions)) + 1)
            for word, positions in words.items()
        }
        fits = self.cached(
            {
                self.key(word, n): (
                    lambda word=word, n=n: (
                        fit_gmm,
                        self.word_vectors(word),
                        n,
                        self.covariance_type,
                        self.reg_covar,
                        self.random_state,
                    )
                )
                for word, ns in candidates.items()
                for n in ns
            }
        )
        self.models_by_word = {
            word: min((fits[self.key(word, n)] for n in ns), key=lambda fit: fit[1])[0]
            for word, ns in candidates.items()
        }

    def predict_word(self, word, vectors, index):
        return self.models_by_word[word].predict(vectors)
