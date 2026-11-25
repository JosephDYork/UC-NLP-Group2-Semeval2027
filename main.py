import quiet  # noqa: F401
import argparse
from typing import NamedTuple

import pandas as pd

from build_datasets import (
    build_synthetic_splits,
    load_dataset,
    preprocess_dataset,
    subsample_splits,
    synthetic_gold_by_sentence,
)
from build_embeddings import build_embeddings, build_tfidf
from evaluations import bootstrap_ci, score_jsd_spearman
from experiment import (
    PrecomputedFeatures,
    baseline_scores,
    baselines,
    build_model,
    fit_and_score,
    load_vectors,
)
from features import TfidfEmbeddings
from models import MODELS


class Config(NamedTuple):
    model: str
    features: str
    reducer: str
    params: dict
    source: str


CONFIGS = [
    Config(
        "correlation",
        "xl_durel",
        "none",
        {"threshold": 0.55, "threshold_quantile": None, "s": 10},
        "results/tuning_20261111_022806 (all words)",
    ),
    Config(
        "kmeans",
        "xl_lexeme",
        "AE-64",
        {"max_k": 10, "min_silhouette": 0.05, "max_iter": 100},
        "earlier XL-LEXEME tuning",
    ),
    Config(
        "gmm",
        "xl_lexeme",
        "UMAP-10 (nn=15)",
        {"max_components": 4, "covariance_type": "full", "reg_covar": 0.01},
        "results/tuning_20261110_185943",
    ),
    Config(
        "hdbscan",
        "xl_lexeme",
        "AE-128",
        {
            "min_cluster_size": 5,
            "min_samples": 1,
            "cluster_selection_method": "eom",
            "allow_single_cluster": True,
            "merge_noise": False,
        },
        "earlier XL-LEXEME tuning",
    ),
]


def subtask_1b(model, all_rows, gold_by_sentence):
    predictions = model.fit(all_rows).predict(all_rows)
    return score_jsd_spearman(all_rows, predictions, gold_by_sentence)


def format_ci(word_f1):
    low, high = bootstrap_ci(word_f1)
    return f"[{low:.3f}, {high:.3f}]"


def evaluate_config(config, splits, gold_by_sentence, n_jobs):
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    fit_rows = pd.concat([train, validation])
    all_rows = pd.concat(splits.values())
    model_class = MODELS[config.model]

    def model(vectors):
        return build_model(
            model_class, config.params, vectors, config.reducer, n_jobs=n_jobs
        )

    vectors = load_vectors(config.features, config.reducer, train, all_rows)
    val = fit_and_score(model(vectors), train, validation, gold_by_sentence)
    vectors = load_vectors(config.features, config.reducer, fit_rows, all_rows)
    test = fit_and_score(model(vectors), fit_rows, test, gold_by_sentence)
    return {
        "model": model_class.display_name,
        "features": config.features,
        "reducer": config.reducer,
        "val_f1": val["f1"],
        "test_f1": test["f1"],
        "test_f1_95ci": format_ci(test["word_f1"]),
        "test_clusters": test["avg_clusters_per_word"],
        "jsd_spearman": subtask_1b(model(vectors), all_rows, gold_by_sentence),
    }


def evaluate_baselines(splits, gold_by_sentence, n_jobs):
    fit_rows = pd.concat([splits["train"], splits["validation"]])
    all_rows = pd.concat(splits.values())
    tfidf = PrecomputedFeatures(TfidfEmbeddings().load())
    scores = baseline_scores(fit_rows, splits["test"], gold_by_sentence, tfidf, n_jobs)
    scores["test_f1_95ci"] = scores.word_f1.map(format_ci)
    scores["jsd_spearman"] = [
        subtask_1b(model, all_rows, gold_by_sentence)
        for _, model in baselines(tfidf, n_jobs)
    ]
    return scores.rename(
        columns={"f1": "test_f1", "avg_clusters_per_word": "test_clusters"}
    )[["baseline", "test_f1", "test_f1_95ci", "test_clusters", "jsd_spearman"]]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[config.model for config in CONFIGS],
        default=[config.model for config in CONFIGS],
        help="which configurations to evaluate, by model (default: all)",
    )
    parser.add_argument(
        "--n-words",
        type=int,
        default=0,
        help="evaluate on this many randomly sampled words (default 0 = all words)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="parallel workers for per-word fits (-1 = all cores, 1 = serial)",
    )
    parser.add_argument(
        "--device", default="cuda", help="torch device for any embeddings built"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configs = [config for config in CONFIGS if config.model in args.models]

    preprocess_dataset()
    rows = load_dataset()
    build_tfidf(rows)
    build_embeddings(
        rows, sorted({config.features for config in configs}), device=args.device
    )

    splits = build_synthetic_splits(rows=rows)
    if args.n_words:
        splits = subsample_splits(splits, args.n_words)
    gold_by_sentence = synthetic_gold_by_sentence(pd.concat(splits.values()))
    splits = {
        name: split[["word", "sentence_id", "period_label"]]
        for name, split in splits.items()
    }
    del rows
    print(
        f"{splits['validation'].word.nunique()} words: "
        + ", ".join(f"{name} {len(split)}" for name, split in splits.items())
    )

    floats = {"float_format": "{:.3f}".format, "index": False}
    print("\nBaselines")
    print(evaluate_baselines(splits, gold_by_sentence, args.jobs).to_string(**floats))

    results = pd.DataFrame(
        [
            evaluate_config(config, splits, gold_by_sentence, args.jobs)
            for config in configs
        ]
    )
    print("\nConfigurations")
    print(results.to_string(**floats))


if __name__ == "__main__":
    main()
