import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from features import TfidfEmbeddings, XLLexemeEmbeddings, XLMRobertaEmbeddings
from reducers import AutoencoderReducer, UMAPReducer
from subtask1b import consecutive_jsd, gold_jsd, sense_counts, spearman_1b
from models import (
    GaussianMixtureModel,
    HdbscanModel,
    KmeansModel,
    RandomBaselineModel,
    SingleLabelBaselineModel,
)
from tune import (
    FEATURES,
    MODELS,
    REDUCERS,
    baseline_configurations,
    baseline_scores as tune_baseline_scores,
    build_model,
    reduce_embeddings,
    score as tune_score,
)

TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"^\w+(?:[-']\w+)*$", re.UNICODE)
SPLIT_RANDOM_STATE = 2027


def target_index(tokens, start, end):
    return max(
        [
            (i, min(token_end, end) - max(token_start, start))
            for i, (_, token_start, token_end) in enumerate(tokens)
        ],
        key=lambda item: item[1],
    )[0]


def tokenize(row):
    tokens = [
        (match.group(), match.start(), match.end())
        for match in TOKEN_RE.finditer(row.text)
    ]
    target = target_index(tokens, int(row.start), int(row.end))

    return [
        token.lower()
        for i, (token, _, _) in enumerate(tokens)
        if i != target and WORD_RE.match(token) is not None
    ]


def prepare_rows(rows):
    rows = rows.copy()
    rows["context_tokens"] = rows.apply(tokenize, axis=1)
    rows["target_form"] = rows.apply(
        lambda row: row.text[int(row.start) : int(row.end)].lower().strip(), axis=1
    )

    return rows


def build_train():
    return prepare_rows(pd.read_parquet("data/usages.parquet"))


def build_splits(random_state=SPLIT_RANDOM_STATE):
    rows = build_train()
    train_indices, held_out_indices = next(
        StratifiedShuffleSplit(
            n_splits=1, test_size=0.30, random_state=random_state
        ).split(rows, rows.word)
    )
    held_out = rows.iloc[held_out_indices]
    validation_indices, test_indices = next(
        StratifiedShuffleSplit(
            n_splits=1, test_size=0.50, random_state=random_state
        ).split(held_out, held_out.word)
    )
    return {
        "train": rows.iloc[train_indices],
        "validation": held_out.iloc[validation_indices],
        "test": held_out.iloc[test_indices],
    }


def bcubed_f1(gold_sets, predicted_labels):
    precision, recall = [], []
    for gold_i, predicted_i in zip(gold_sets, predicted_labels):
        predicted_i = {predicted_i}
        predicted_similarity = np.array(
            [float(predicted_i == {label}) for label in predicted_labels]
        )
        gold_similarity = np.array(
            [len(gold_i & gold_j) / len(gold_i | gold_j) for gold_j in gold_sets]
        )
        shared = (predicted_similarity * gold_similarity).sum()
        precision.append(shared / predicted_similarity.sum())
        recall.append(shared / gold_similarity.sum())
    p, r = float(np.mean(precision)), float(np.mean(recall))

    return p, r, 0.0 if p + r == 0 else 2 * p * r / (p + r)


def score_partition(rows, predictions, gold_by_sentence):
    predictions = np.asarray(predictions)
    word_scores = []
    for word, word_rows in rows.groupby("word", sort=False):
        mask = rows.word.eq(word).to_numpy()
        gold_sets = [
            {str(sense) for sense in row.label}
            for row in word_rows.merge(
                pd.Series(gold_by_sentence, name="label"),
                left_on="sentence_id",
                right_index=True,
                how="left",
            ).itertuples()
        ]
        word_scores.append(bcubed_f1(gold_sets, predictions[mask]))

    return np.mean(word_scores, axis=0)


def evaluate(model_class, fit_rows, evaluation_rows, **model_kwargs):
    gold = pd.read_parquet("data/subtask1.parquet")
    gold_by_sentence = gold.set_index("sentence_id").label.to_dict()
    model = model_class(random_state=42, **model_kwargs).fit(fit_rows)
    precision, recall, f1 = score_partition(
        evaluation_rows, model.predict(evaluation_rows), gold_by_sentence
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def train_validate(model_class, splits, **model_kwargs):
    return evaluate(model_class, splits["train"], splits["validation"], **model_kwargs)


def train_test(model_class, splits, **model_kwargs):
    return evaluate(
        model_class,
        pd.concat([splits["train"], splits["validation"]]),
        splits["test"],
        **model_kwargs,
    )


def jsd_spearman(model_class, rows, gold_by_sentence, **model_kwargs):
    model_kwargs.setdefault("random_state", 42)
    model = model_class(**model_kwargs).fit(rows)
    predictions = model.predict(rows)
    gold = gold_jsd(rows, gold_by_sentence)
    predicted = consecutive_jsd(sense_counts(rows, predictions))
    return spearman_1b(gold, predicted)


def main():
    splits = build_splits()
    all_rows = pd.concat(splits.values()).sort_index()
    gold_by_sentence = (
        pd.read_parquet("data/subtask1.parquet")
        .set_index("sentence_id")
        .label.to_dict()
    )

    fit_rows = pd.concat([splits["train"], splits["validation"]])
    test_baselines = tune_baseline_scores(fit_rows, splits["test"], gold_by_sentence)
    for index, (_, model_class, model_kwargs) in enumerate(baseline_configurations()):
        test_baselines.loc[index, "jsd_spearman"] = jsd_spearman(
            model_class, all_rows, gold_by_sentence, **model_kwargs
        )

    print("Baselines (validation)")
    print(
        test_baselines.to_string(
            index=False,
            formatters={
                "precision": "{:.3f}".format,
                "recall": "{:.3f}".format,
                "f1": "{:.3f}".format,
                "jsd_spearman": "{:.3f}".format,
            },
        )
    )

    champions = {
        "XL-LEXEME": (
            XLLexemeEmbeddings,
            [
                (
                    "K-means",
                    "AE-64",
                    KmeansModel,
                    lambda: {
                        "max_k": 10,
                        "min_silhouette": 0.05,
                        "max_iter": 100,
                        "autoencoder": AutoencoderReducer(
                            n_components=64, hidden_dims=(512,)
                        ),
                    },
                ),
                (
                    "GMM",
                    "UMAP-10 (nn=15)",
                    GaussianMixtureModel,
                    lambda: {
                        "max_components": 4,
                        "covariance_type": "full",
                        "reg_covar": 0.01,
                        "umap": UMAPReducer(
                            n_components=10, n_neighbors=15, min_dist=0.0
                        ),
                    },
                ),
                (
                    "HDBSCAN",
                    "AE-128",
                    HdbscanModel,
                    lambda: {
                        "min_cluster_size": 5,
                        "min_samples": 1,
                        "cluster_selection_method": "eom",
                        "allow_single_cluster": True,
                        "normalize_reduced": False,
                        "merge_noise": False,
                        "autoencoder": AutoencoderReducer(
                            n_components=128, hidden_dims=(512,)
                        ),
                    },
                ),
            ],
        ),
        "XLM-R": (
            XLMRobertaEmbeddings,
            [
                (
                    "K-means",
                    "AE-64",
                    KmeansModel,
                    lambda: {
                        "max_k": 4,
                        "min_silhouette": 0.15,
                        "max_iter": 100,
                        "autoencoder": AutoencoderReducer(
                            n_components=64, hidden_dims=(512,)
                        ),
                    },
                ),
                (
                    "GMM",
                    "None",
                    GaussianMixtureModel,
                    lambda: {
                        "max_components": 4,
                        "covariance_type": "spherical",
                        "reg_covar": 0.01,
                    },
                ),
                (
                    "HDBSCAN",
                    "AE-20",
                    HdbscanModel,
                    lambda: {
                        "min_cluster_size": 5,
                        "min_samples": 3,
                        "cluster_selection_method": "eom",
                        "allow_single_cluster": True,
                        "normalize_reduced": False,
                        "merge_noise": False,
                        "autoencoder": AutoencoderReducer(n_components=20),
                    },
                ),
            ],
        ),
    }

    feature_keys = {"XL-LEXEME": "xl_lexeme", "XLM-R": "xlm_roberta"}
    model_keys = {"K-means": "kmeans", "GMM": "gmm", "HDBSCAN": "hdbscan"}
    gold_1b = gold_jsd(all_rows, gold_by_sentence)

    def score_1b(model):
        predictions = model.fit(all_rows).predict(all_rows)
        return spearman_1b(
            gold_1b, consecutive_jsd(sense_counts(all_rows, predictions))
        )

    for encoder_name, (_, configurations) in champions.items():
        rows = []
        for model_name, reducer_name, model_class, build_kwargs in configurations:
            model_key = model_keys[model_name]
            feature_key = feature_keys[encoder_name]
            reducer_key = "none" if reducer_name == "None" else reducer_name
            params = {
                name: value
                for name, value in build_kwargs().items()
                if name not in {"autoencoder", "umap"}
            }
            raw = FEATURES[feature_key]().load()
            validation_vectors = reduce_embeddings(
                raw, splits["train"], REDUCERS[reducer_key]
            )
            validation_scores = tune_score(
                build_model(MODELS[model_key][1], params, validation_vectors),
                splits["train"],
                splits["validation"],
                gold_by_sentence,
            )
            rho_model = build_model(MODELS[model_key][1], params, validation_vectors)
            test_vectors = reduce_embeddings(raw, fit_rows, REDUCERS[reducer_key])
            test_scores = tune_score(
                build_model(MODELS[model_key][1], params, test_vectors),
                fit_rows,
                splits["test"],
                gold_by_sentence,
            )
            rows.append(
                {
                    "model": model_name,
                    "reducer": reducer_name,
                    "val_b3_f1": validation_scores["f1"],
                    "test_b3_f1": test_scores["f1"],
                    "jsd_spearman": score_1b(rho_model),
                }
            )

        print(f"\n{encoder_name} champions")
        print(
            pd.DataFrame(rows).to_string(
                index=False,
                formatters={
                    "val_b3_f1": "{:.3f}".format,
                    "test_b3_f1": "{:.3f}".format,
                    "jsd_spearman": "{:.3f}".format,
                },
            )
        )


if __name__ == "__main__":
    main()
