import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from evaluations import gold_clusters_per_word, score_b3f1_by_word
from features import ENCODERS, TfidfEmbeddings
from models import KmeansModel, SingleLabelBaselineModel
from reducers import AutoencoderReducer, UMAPReducer

RANDOM_STATE = 42

REDUCERS = {
    "none": {},
    "AE-20": {"autoencoder": {"n_components": 20}},
    "AE-64": {"autoencoder": {"n_components": 64, "hidden_dims": (512,)}},
    "AE-128": {"autoencoder": {"n_components": 128, "hidden_dims": (512,)}},
    "UMAP-10 (nn=15)": {
        "umap": {"n_components": 10, "n_neighbors": 15, "min_dist": 0.0}
    },
}


class PrecomputedFeatures:
    def __init__(self, vectors):
        self.vectors = vectors

    def load(self):
        return self.vectors


def reduced_dims(n_dims, reducer):
    for kind in ("umap", "autoencoder"):
        if kind in REDUCERS[reducer]:
            return REDUCERS[reducer][kind]["n_components"]
    return n_dims


def reduce_embeddings(raw, fit_rows, rows, reducer):
    config = REDUCERS[reducer]
    reducers = []
    if "autoencoder" in config:
        reducers.append(
            AutoencoderReducer(random_state=RANDOM_STATE, **config["autoencoder"])
        )
    if "umap" in config:
        reducers.append(UMAPReducer(random_state=RANDOM_STATE, **config["umap"]))
    if not reducers:
        return raw

    fit_vectors = normalize(raw[fit_rows.index.to_numpy()], copy=False)
    index = np.sort(rows.index.to_numpy())
    vectors = normalize(raw[index], copy=False)
    for reducer_model in reducers:
        reducer_model.fit(fit_vectors)
        fit_vectors = reducer_model.transform(fit_vectors)
        vectors = reducer_model.transform(vectors)
    reduced = np.zeros((len(raw), vectors.shape[1]), dtype=vectors.dtype)
    reduced[index] = vectors
    return reduced


def load_vectors(features, reducer, fit_rows, rows):
    return reduce_embeddings(ENCODERS[features]().load(), fit_rows, rows, reducer)


def build_model(model_class, params, vectors, reducer, **kwargs):
    return model_class(
        random_state=RANDOM_STATE,
        feature_source=PrecomputedFeatures(vectors),
        normalize_vectors=not REDUCERS[reducer],
        **params,
        **kwargs,
    )


def fit_and_score(model, fit_rows, eval_rows, gold_by_sentence):
    model.fit(fit_rows)
    predictions = model.predict(eval_rows)
    by_word = score_b3f1_by_word(eval_rows, predictions, gold_by_sentence)
    precision, recall, f1 = by_word.mean()
    n_clusters = (
        pd.Series(predictions, index=eval_rows.index)
        .groupby(eval_rows.word)
        .nunique()
        .mean()
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "avg_clusters_per_word": float(n_clusters),
        "gold_clusters_per_word": gold_clusters_per_word(eval_rows, gold_by_sentence),
        "word_f1": by_word.f1.to_dict(),
    }


ONE_CLUSTER = "One cluster per word"


def baselines(tfidf_source=None, n_jobs=1):
    return [
        (ONE_CLUSTER, SingleLabelBaselineModel()),
        (
            "TF-IDF + K-means",
            KmeansModel(
                random_state=RANDOM_STATE,
                feature_source=tfidf_source or TfidfEmbeddings(),
                n_jobs=n_jobs,
            ),
        ),
    ]


def baseline_scores(
    fit_rows, eval_rows, gold_by_sentence, tfidf_source=None, n_jobs=1
):
    return pd.DataFrame(
        [
            {
                "baseline": name,
                **fit_and_score(model, fit_rows, eval_rows, gold_by_sentence),
            }
            for name, model in baselines(tfidf_source, n_jobs)
        ]
    )
