import itertools

import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits


def single_threaded(function, *args):
    with threadpool_limits(limits=1):
        return function(*args)


class BaseModel:
    display_name = None

    def __init__(
        self,
        random_state=42,
        feature_source=None,
        normalize_vectors=True,
        n_jobs=1,
        cache=None,
        min_df=2,
    ):
        self.random_state = random_state
        self.feature_source = feature_source
        self.normalize_vectors = normalize_vectors
        self.n_jobs = n_jobs
        self.cache = {} if cache is None else cache
        self.min_df = min_df
        self.vectorizer = None
        self.training_vectors = None
        self.training_positions_by_word = {}
        self.models_by_word = {}

    @staticmethod
    def grid(**space):
        keys = list(space)
        return [
            dict(zip(keys, values)) for values in itertools.product(*space.values())
        ]

    @classmethod
    def search_space(cls, quick=False, n_dims=None):
        return [{}]

    def vector_getter(self, rows, fit=False):
        if self.feature_source is None:
            vectors = (
                self.fit_tfidf(rows)
                if fit
                else self.vectorizer.transform(rows.context_tokens)
            )
            vectors = normalize(vectors)
            return lambda positions: vectors[positions]

        source = self.feature_source.load()
        index = rows.index.to_numpy()
        normalize_vectors = self.normalize_vectors

        def get(positions):
            vectors = source[index[positions]]
            return normalize(vectors) if normalize_vectors else vectors

        return get

    def fit_tfidf(self, rows):
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=lambda tokens: tokens,
            preprocessor=lambda tokens: tokens,
            token_pattern=None,
            min_df=self.min_df,
        )
        return self.vectorizer.fit_transform(rows.context_tokens)

    @staticmethod
    def word_positions(rows):
        return rows.groupby("word", sort=False).indices

    def word_vectors(self, word):
        return self.training_vectors(self.training_positions_by_word[word])

    def fit(self, train):
        self.training_vectors = self.vector_getter(train, fit=True)
        self.training_positions_by_word = self.word_positions(train)
        self.models_by_word = {}
        self.fit_words(self.training_positions_by_word)
        return self

    def fit_words(self, words):
        raise NotImplementedError

    def predict(self, rows):
        vectors = self.vector_getter(rows)
        index = rows.index.to_numpy()
        predictions = np.empty(len(rows), dtype=object)
        for word, positions in self.word_positions(rows).items():
            if word in self.models_by_word:
                labels = self.predict_word(word, vectors(positions), index[positions])
            else:
                labels = np.zeros(len(positions), dtype=int)
            predictions[positions] = [f"{word}:{label}" for label in labels]
        return predictions.tolist()

    def predict_word(self, word, vectors, index):
        raise NotImplementedError

    def cached(self, tasks):
        missing = [key for key in tasks if key not in self.cache]

        def resolve(task):
            return task() if callable(task) else task

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(single_threaded)(*resolve(tasks[key])) for key in missing
        )

        self.cache.update(zip(missing, results))
        return {key: self.cache[key] for key in tasks}
