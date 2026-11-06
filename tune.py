import argparse
import itertools
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from features import (
    CALEXLMREmbeddings,
    TfidfEmbeddings,
    XLLexemeEmbeddings,
    XLMRobertaEmbeddings,
)
from main import build_splits, score_partition
from models import (
    GaussianMixtureModel,
    HdbscanModel,
    KmeansModel,
    RandomBaselineModel,
    SingleLabelBaselineModel,
)
from reducers import AutoencoderReducer, UMAPReducer

RANDOM_STATE = 42

FEATURES = {
    "cale_xlmr": CALEXLMREmbeddings,
    "xl_lexeme": XLLexemeEmbeddings,
}
REDUCERS = {
    "none": {},
    "AE-20": {"autoencoder": {"n_components": 20}},
    "AE-64": {"autoencoder": {"n_components": 64, "hidden_dims": (512,)}},
    "AE-128": {"autoencoder": {"n_components": 128, "hidden_dims": (512,)}},
    "UMAP-10 (nn=5)": {"umap": {"n_components": 10, "n_neighbors": 5, "min_dist": 0.0}},
    "UMAP-10 (nn=15)": {
        "umap": {"n_components": 10, "n_neighbors": 15, "min_dist": 0.0}
    },
    "AE-128 + UMAP-10": {
        "autoencoder": {"n_components": 128, "hidden_dims": (512,)},
        "umap": {"n_components": 10, "n_neighbors": 5, "min_dist": 0.0},
    },
    "AE-20 + UMAP-10": {
        "autoencoder": {"n_components": 20},
        "umap": {"n_components": 10, "n_neighbors": 5, "min_dist": 0.0},
    },
}

QUICK_REDUCERS = ["none", "AE-20 + UMAP-10"]


def grid(**space):
    keys = list(space)
    return [dict(zip(keys, values)) for values in itertools.product(*space.values())]


def kmeans_space(quick):
    if quick:
        return grid(max_k=[8], min_silhouette=[0.0, 0.05])
    return grid(
        max_k=[4, 6, 8, 10],
        min_silhouette=[0.0, 0.02, 0.05, 0.1, 0.15],
        max_iter=[100, 300],
    )


def hdbscan_space(quick):
    common = grid(
        cluster_selection_method=["eom"] if quick else ["eom", "leaf"],
        allow_single_cluster=[False] if quick else [True, False],
        normalize_reduced=[False] if quick else [True, False],
        merge_noise=[True] if quick else [False, True],
    )
    if quick:
        sizing = [{"min_cluster_ratio": r} for r in (0.04, 0.08)]
    else:
        # Ratio-based sizing (min_samples is tied to the ratio inside the model)
        sizing = [{"min_cluster_ratio": r} for r in (0.02, 0.04, 0.06, 0.08, 0.12)]
        # Absolute sizing
        sizing += grid(
            min_cluster_ratio=[None], min_cluster_size=[3, 5, 10], min_samples=[1, 3, 5]
        )
    return [{**s, **c} for s in sizing for c in common]


def gmm_space(quick):
    if quick:
        return grid(max_components=[8], covariance_type=["diag"], reg_covar=[1e-3])
    return grid(
        max_components=[4, 8],
        covariance_type=["diag", "spherical", "full"],
        reg_covar=[1e-5, 1e-3, 1e-2],
    )


MODELS = {
    "kmeans": ("K-means", KmeansModel, kmeans_space),
    "hdbscan": ("HDBSCAN", HdbscanModel, hdbscan_space),
    "gmm": ("GMM", GaussianMixtureModel, gmm_space),
}


def is_valid(model_key, params, n_dims):
    if model_key == "gmm" and params["covariance_type"] == "full" and n_dims > 32:
        return False
    return True


class PrecomputedFeatures:
    def __init__(self, vectors):
        self.vectors = vectors

    def load(self):
        return self.vectors


def reduce_embeddings(raw, fit_rows, reducer_config):
    vectors = normalize(raw)
    reducers = []
    if "autoencoder" in reducer_config:
        reducers.append(
            AutoencoderReducer(
                random_state=RANDOM_STATE, **reducer_config["autoencoder"]
            )
        )
    if "umap" in reducer_config:
        reducers.append(
            UMAPReducer(random_state=RANDOM_STATE, **reducer_config["umap"])
        )
    if not reducers:
        return vectors

    fit_index = fit_rows.index.to_numpy()
    fit_vectors = vectors[fit_index]
    for reducer in reducers:
        reducer.fit(fit_vectors)
        fit_vectors = reducer.transform(fit_vectors)
        vectors = reducer.transform(vectors)
    return vectors


def score(model, fit_rows, eval_rows, gold_by_sentence):
    model.fit(fit_rows)
    predictions = model.predict(eval_rows)
    precision, recall, f1 = score_partition(eval_rows, predictions, gold_by_sentence)
    n_clusters = (
        pd.Series(predictions, index=eval_rows.index)
        .groupby(eval_rows.word)
        .nunique()
        .mean()
    )
    result = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "avg_clusters_per_word": float(n_clusters),
    }
    return result


def build_model(model_class, params, vectors):
    return model_class(
        random_state=RANDOM_STATE,
        feature_source=PrecomputedFeatures(vectors),
        normalize_vectors=False,
        **params,
    )


def run_search(args, splits, gold_by_sentence):
    trials = []
    reducer_names = QUICK_REDUCERS if args.quick else list(REDUCERS)
    for feature_name in args.features:
        raw = FEATURES[feature_name]().load()
        for reducer_name in reducer_names:
            start = time.time()
            vectors = reduce_embeddings(raw, splits["train"], REDUCERS[reducer_name])
            print(
                f"[{feature_name} | {reducer_name}] reduced to {vectors.shape[1]} dims "
                f"in {time.time() - start:.1f}s"
            )
            for model_key in args.models:
                model_name, model_class, space = MODELS[model_key]
                configs = [
                    p
                    for p in space(args.quick)
                    if is_valid(model_key, p, vectors.shape[1])
                ]
                start = time.time()
                for params in configs:
                    trial = {
                        "model": model_name,
                        "features": feature_name,
                        "reducer": reducer_name,
                        "params": json.dumps(params),
                    }
                    try:
                        trial.update(
                            score(
                                build_model(model_class, params, vectors),
                                splits["train"],
                                splits["validation"],
                                gold_by_sentence,
                            )
                        )
                    except Exception as error:  # keep going if one config blows up
                        trial["error"] = f"{type(error).__name__}: {error}"
                    trials.append(trial)
                print(
                    f"    {model_name:8s} {len(configs):4d} configs "
                    f"in {time.time() - start:.1f}s"
                )
    return pd.DataFrame(trials)


def evaluate_best_on_test(best, splits, gold_by_sentence):
    fit_rows = pd.concat([splits["train"], splits["validation"]])
    model_classes = {name: cls for name, cls, _ in MODELS.values()}
    rows = []
    for trial in best.itertuples():
        raw = FEATURES[trial.features]().load()
        vectors = reduce_embeddings(raw, fit_rows, REDUCERS[trial.reducer])
        model = build_model(
            model_classes[trial.model], json.loads(trial.params), vectors
        )
        rows.append(
            {
                "model": trial.model,
                "features": trial.features,
                "reducer": trial.reducer,
                "params": trial.params,
                "val_f1": trial.f1,
                **{
                    f"test_{k}": v
                    for k, v in score(
                        model, fit_rows, splits["test"], gold_by_sentence
                    ).items()
                },
            }
        )
    return pd.DataFrame(rows)


def baseline_configurations():
    return [
        ("One cluster per word", SingleLabelBaselineModel, {}),
        (
            "Random (up to 8 clusters)",
            RandomBaselineModel,
            {"random_state": RANDOM_STATE},
        ),
        (
            "TF-IDF + K-means",
            KmeansModel,
            {"random_state": RANDOM_STATE, "feature_source": TfidfEmbeddings()},
        ),
    ]


def baseline_scores(fit_rows, evaluation_rows, gold_by_sentence):
    rows = []
    for name, model_class, model_kwargs in baseline_configurations():
        rows.append(
            {
                "baseline": name,
                **score(
                    model_class(**model_kwargs),
                    fit_rows,
                    evaluation_rows,
                    gold_by_sentence,
                ),
            }
        )
    return pd.DataFrame(rows)


def format_params(params_json):
    params = json.loads(params_json)
    return ", ".join(f"{k}={v}" for k, v in params.items() if v is not None)


def to_markdown(frame, float_cols=("precision", "recall", "f1")):
    frame = frame.copy()
    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            digits = 2 if col.endswith("avg_clusters_per_word") else 3
            frame[col] = frame[col].map(
                lambda v: "" if pd.isna(v) else f"{v:.{digits}f}"
            )
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(path, args, trials, baselines, test_results, test_baselines, elapsed):
    ok = trials[trials["error"].isna()]
    failed = len(trials) - len(ok)
    ranked = ok.sort_values("f1", ascending=False)
    best = ranked.groupby("model", sort=False).head(1)

    display_cols = [
        "model",
        "features",
        "reducer",
        "params",
        "precision",
        "recall",
        "f1",
        "avg_clusters_per_word",
    ]

    def readable(frame):
        frame = frame[display_cols].copy()
        frame["params"] = frame["params"].map(format_params)
        return frame

    out = [
        "# Hyperparameter tuning report",
        "",
        f"- Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- Mode: {'quick' if args.quick else 'full'}",
        f"- Models: {', '.join(MODELS[m][0] for m in args.models)}",
        f"- Features: {', '.join(args.features)}",
        f"- Trials: {len(trials)} ({failed} failed)",
        f"- Runtime: {elapsed / 60:.1f} min",
        "- Selection metric: validation BCubed F1 (Subtask 1a), macro-averaged over target words.",
        "- Models are fit on the training split; validation gold labels are used only for selection.",
        "",
        "## Best configuration per model (validation)",
        "",
        to_markdown(readable(best)),
        "",
    ]

    if test_results is not None:
        test_display = test_results.copy()
        test_display["params"] = test_display["params"].map(format_params)
        out += [
            "## Best configurations on the held-out test split",
            "",
            "Refit on train + validation, then score Subtask 1a once on test.",
            "",
            to_markdown(test_display),
            "",
            "## Baselines (held-out test)",
            "",
            to_markdown(test_baselines),
            "",
        ]

    out += [
        "## Baselines (validation)",
        "",
        to_markdown(baselines),
        "",
    ]
    for metric, title in [("f1", "Best validation BCubed F1 (1a)")]:
        out += [f"## {title} by model and reducer", ""]
        for feature_name, group in ok.groupby("features"):
            pivot = (
                group.pivot_table(
                    index="model", columns="reducer", values=metric, aggfunc="max"
                )
                .reindex(columns=[r for r in REDUCERS if r in group.reducer.unique()])
                .reset_index()
            )
            out += [f"### Features: `{feature_name}`", "", to_markdown(pivot), ""]

    out += [f"## Top {args.top} configurations per model", ""]
    for model_name, group in ranked.groupby("model", sort=False):
        out += [
            f"### {model_name}",
            "",
            to_markdown(readable(group.head(args.top))),
            "",
        ]

    if failed:
        errors = trials[trials["error"].notna()][
            ["model", "features", "reducer", "params", "error"]
        ]
        out += ["## Failed configurations", "", to_markdown(errors), ""]

    path.write_text("\n".join(out))
    return best


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS), default=list(MODELS)
    )
    parser.add_argument(
        "--features", nargs="+", choices=list(FEATURES), default=list(FEATURES)
    )
    parser.add_argument(
        "--quick", action="store_true", help="tiny grid for smoke testing"
    )
    parser.add_argument(
        "--no-test", action="store_true", help="skip test-split evaluation"
    )
    parser.add_argument(
        "--top", type=int, default=10, help="rows per model in the top-N tables"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    start = time.time()
    splits = build_splits()
    gold = pd.read_parquet("data/subtask1.parquet")
    gold_by_sentence = gold.set_index("sentence_id").label.to_dict()

    trials = run_search(args, splits, gold_by_sentence)
    if "error" not in trials:
        trials["error"] = None
    baselines = baseline_scores(splits["train"], splits["validation"], gold_by_sentence)

    ok = trials[trials["error"].isna()].sort_values("f1", ascending=False)
    best = ok.groupby("model", sort=False).head(1)
    test_results = (
        None if args.no_test else evaluate_best_on_test(best, splits, gold_by_sentence)
    )

    test_baselines = (
        None
        if args.no_test
        else baseline_scores(
            pd.concat([splits["train"], splits["validation"]]),
            splits["test"],
            gold_by_sentence,
        )
    )

    out_dir = args.output_dir / f"tuning_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trials.sort_values("f1", ascending=False).to_csv(
        out_dir / "trials.csv", index=False
    )
    if test_results is not None:
        test_results.to_csv(out_dir / "best_test.csv", index=False)
        test_baselines.to_csv(out_dir / "baseline_test.csv", index=False)
    write_report(
        out_dir / "report.md",
        args,
        trials,
        baselines,
        test_results,
        test_baselines,
        time.time() - start,
    )

    summary = best[["model", "features", "reducer", "params", "f1"]].copy()
    summary["params"] = summary["params"].map(format_params)
    summary = summary.rename(columns={"f1": "val_f1"})
    if test_results is not None:
        summary["test_f1"] = test_results["test_f1"].to_numpy()
    print("\nBest configuration per model")
    print(summary.to_string(index=False, float_format="{:.3f}".format))
    print(f"\nFull report: {out_dir / 'report.md'}")
    print(f"All trials:  {out_dir / 'trials.csv'}")


if __name__ == "__main__":
    main()
