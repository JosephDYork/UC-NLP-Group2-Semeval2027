import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from features import TfidfEmbeddings, XLLexemeEmbeddings
from models import (
    GaussianMixtureModel,
    HdbscanModel,
    KmeansModel,
    RandomBaselineModel,
    SingleLabelBaselineModel,
)
from reducers import AutoencoderReducer, UMAPReducer

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


def main():
    splits = build_splits()

    baseline_configurations = [
        ("One cluster per target word", SingleLabelBaselineModel, {}),
        ("Random partition (up to 8 clusters)", RandomBaselineModel, {}),
        ("TF-IDF + K-means", KmeansModel, {"feature_source": TfidfEmbeddings()}),
    ]

    model_configurations = [
        ("K-means", KmeansModel, {}),
        (
            "HDBSCAN",
            HdbscanModel,
            {
                "min_cluster_ratio": 0.04,
                "allow_single_cluster": False,
                "normalize_reduced": False,
            },
        ),
        ("GMM", GaussianMixtureModel, {}),
    ]

    reducer_configurations = [
        ("Raw embeddings", lambda: {}),
        (
            "AE-128",
            lambda: {
                "autoencoder": AutoencoderReducer(n_components=128, hidden_dims=(512,))
            },
        ),
        (
            "AE-128 + UMAP-10",
            lambda: {
                "autoencoder": AutoencoderReducer(n_components=128, hidden_dims=(512,)),
                "umap": UMAPReducer(n_components=10, n_neighbors=5, min_dist=0.0),
            },
        ),
        (
            "CMCE AE-20 + UMAP-10",
            lambda: {
                "autoencoder": AutoencoderReducer(n_components=20),
                "umap": UMAPReducer(n_components=10, n_neighbors=5, min_dist=0.0),
            },
        ),
    ]

    validation_grid, test_grid = [], []
    for model_name, model_class, model_kwargs in model_configurations:
        validation_scores = {"model": model_name}
        # test_scores = {"model": model_name}
        for reducer_name, build_reducers in reducer_configurations:
            validation_result = train_validate(
                model_class,
                splits,
                feature_source=XLLexemeEmbeddings(),
                **model_kwargs,
                **build_reducers(),
            )
            # test_result = train_test(
            #     model_class,
            #     splits,
            #     feature_source=XLLexemeEmbeddings(),
            #     **model_kwargs,
            #     **build_reducers(),
            # )
            validation_scores[reducer_name] = validation_result["f1"]
            # test_scores[reducer_name] = test_result["f1"]
        validation_grid.append(validation_scores)
        # test_grid.append(test_scores)

    validation_baselines, test_baselines = [], []
    for name, model_class, model_kwargs in baseline_configurations:
        validation_baselines.append(
            {
                "baseline": name,
                **train_validate(model_class, splits, **model_kwargs),
            }
        )
        # test_baselines.append(
        #     {
        #         "baseline": name,
        #         **train_test(model_class, splits, **model_kwargs),
        #     }
        # )

    print("Baselines (validation)")
    print(
        pd.DataFrame(validation_baselines).to_string(
            index=False,
            formatters={
                "precision": "{:.3f}".format,
                "recall": "{:.3f}".format,
                "f1": "{:.3f}".format,
            },
        )
    )
    print("\nXL-LEXEME validation F1 by model and reducer")
    print(
        pd.DataFrame(validation_grid).to_string(
            index=False,
            formatters={name: "{:.3f}".format for name, _ in reducer_configurations},
        )
    )
    # print("\nBaselines (test)")
    # print(
    #     pd.DataFrame(test_baselines).to_string(
    #         index=False,
    #         formatters={
    #             "precision": "{:.3f}".format,
    #             "recall": "{:.3f}".format,
    #             "f1": "{:.3f}".format,
    #         },
    #     )
    # )
    # print("\nXL-LEXEME test F1 by model and reducer")
    # print(
    #     pd.DataFrame(test_grid).to_string(
    #         index=False,
    #         formatters={name: "{:.3f}".format for name, _ in reducer_configurations},
    #     )
    # )


if __name__ == "__main__":
    main()
