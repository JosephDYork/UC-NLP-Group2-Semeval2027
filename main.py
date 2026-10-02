import re

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from kmeans_model import KmeansBaselineModel
from random_model import RandomBaselineModel
from single_label_model import SingleLabelBaselineModel

TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"^\w+(?:[-']\w+)*$", re.UNICODE)


def target_index(tokens, start, end):
    return max(
        [
            (i, min(token_end, end) - max(token_start, start))
            for i, (_, token_start, token_end) in enumerate(tokens)
        ],
        key=lambda item: item[1],
    )[0]


def tokenize(row):
    tokens = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(row.text)]
    target = target_index(tokens, int(row.start), int(row.end))
    return [
        token.lower()
        for i, (token, _, _) in enumerate(tokens)
        if i != target and WORD_RE.match(token) is not None
    ]


def build_train():
    train = pd.read_parquet("data/usages.parquet")
    train["context_tokens"] = train.apply(tokenize, axis=1)
    train["target_form"] = train.apply(
        lambda row: row.text[int(row.start) : int(row.end)].lower().strip(), axis=1
    )
    return train


def bcubedF1(gold_sets, pred_labels):
    precision, recall = [], []
    for gold_i, pred_i in zip(gold_sets, pred_labels):
        same_cluster = np.array([pred_i == pred_j for pred_j in pred_labels])
        gold_similarity = np.array(
            [len(gold_i & gold_j) / len(gold_i | gold_j) for gold_j in gold_sets]
        )
        shared = gold_similarity[same_cluster].sum()
        precision.append(shared / same_cluster.sum())
        recall.append(shared / gold_similarity.sum())

    p, r = float(np.mean(precision)), float(np.mean(recall))
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f1


def crossval(model_class, train, **model_kwargs):
    gold = pd.read_parquet("data/subtask1.parquet")
    gold_by_sentence = gold.set_index("sentence_id").label.to_dict()
    cross_folds = KFold(shuffle=True, random_state=42).split(train)

    results = []
    for fold, (fit_idx, eval_idx) in enumerate(cross_folds, start=1):
        fit = train.iloc[fit_idx].reset_index(drop=True)
        evaluation = train.iloc[eval_idx].reset_index(drop=True)
        model = model_class(random_state=42 + fold, **model_kwargs).fit(fit)
        pred = model.predict(evaluation)

        gold_sets = [
            {f"{row.word}:{sense}" for sense in gold_by_sentence[row.sentence_id]}
            for row in evaluation.itertuples()
        ]
        p, r, f1 = bcubedF1(gold_sets, pred)
        results.append({"fold": fold, "precision": p, "recall": r, "f1": f1})

    return pd.DataFrame(results)


def main():
    results = []
    train = build_train()

    for name, model_class in [
        ("single label", SingleLabelBaselineModel),
        ("random", RandomBaselineModel),
        ("k-means", KmeansBaselineModel),
    ]:
        scores = crossval(model_class, train)
        summary = scores[["precision", "recall", "f1"]].agg(["mean"])
        results.append(
            {
                "model": name,
                "precision": summary.loc["mean", "precision"],
                "recall": summary.loc["mean", "recall"],
                "bCubedF1": summary.loc["mean", "f1"],
            }
        )

    comparison = pd.DataFrame(results)
    print("Baseline Comparison")
    print("===================")
    print(
        comparison.to_string(
            index=False,
            formatters={
                "precision": "{:.3f}".format,
                "recall": "{:.3f}".format,
                "bCubedF1": "{:.3f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()
