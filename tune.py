import quiet  # noqa: F401
import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from mdutils.mdutils import MdUtils
from tqdm import tqdm

from build_datasets import (
    build_synthetic_splits,
    subsample_splits,
    synthetic_gold_by_sentence,
)
from evaluations import N_BOOTSTRAP, bootstrap_ci
from experiment import (
    ONE_CLUSTER,
    PrecomputedFeatures,
    baseline_scores,
    build_model,
    fit_and_score,
    load_vectors,
    reduced_dims,
)
from features import ENCODERS, TfidfEmbeddings
from models import MODELS

DEFAULT_N_WORDS = 30
MAX_INITIAL_FAILURES = 10
SEARCH_REDUCERS = ["none", "AE-20", "AE-64", "AE-128"]
QUICK_REDUCERS = ["none", "AE-20"]
MODEL_BY_NAME = {cls.display_name: cls for cls in MODELS.values()}


def run_search(args, splits, gold_by_sentence, log_path):
    trials = []
    failures = 0
    reducers = args.reducers or (QUICK_REDUCERS if args.quick else SEARCH_REDUCERS)
    all_rows = pd.concat(splits.values())

    with log_path.open("w") as log:
        for features in args.features:
            n_dims = ENCODERS[features]().load().shape[1]
            feature_bar = tqdm(
                total=sum(
                    len(
                        MODELS[key].search_space(
                            args.quick, reduced_dims(n_dims, reducer)
                        )
                    )
                    for reducer in reducers
                    for key in args.models
                ),
                desc=features,
                unit="trial",
                position=0,
                dynamic_ncols=True,
            )
            for reducer in reducers:
                feature_bar.set_postfix_str(reducer)
                start = time.time()
                vectors = load_vectors(features, reducer, splits["train"], all_rows)
                tqdm.write(
                    f"[{features} | {reducer}] reduced to {vectors.shape[1]} dims "
                    f"in {time.time() - start:.1f}s"
                )
                cache = {}
                for key in args.models:
                    model_class = MODELS[key]
                    for params in tqdm(
                        model_class.search_space(args.quick, vectors.shape[1]),
                        desc=f"    {model_class.display_name}",
                        unit="config",
                        position=1,
                        leave=False,
                        dynamic_ncols=True,
                    ):
                        trial = {
                            "model": model_class.display_name,
                            "features": features,
                            "reducer": reducer,
                            "params": json.dumps(params),
                        }
                        try:
                            model = build_model(
                                model_class,
                                params,
                                vectors,
                                reducer,
                                n_jobs=args.jobs,
                                cache=cache,
                            )
                            trial.update(
                                fit_and_score(
                                    model,
                                    splits["train"],
                                    splits["validation"],
                                    gold_by_sentence,
                                )
                            )
                        except Exception as error:
                            trial["error"] = f"{type(error).__name__}: {error}"
                            failures += 1
                        trials.append(trial)
                        log.write(json.dumps(trial) + "\n")
                        log.flush()
                        feature_bar.update()
                        if failures >= MAX_INITIAL_FAILURES and failures == len(trials):
                            raise RuntimeError(
                                f"First {failures} trials all failed, last error: "
                                f"{trial['error']}"
                            )
            feature_bar.close()
            del vectors, cache
            gc.collect()

    trials = pd.DataFrame(trials)
    if "error" not in trials:
        trials["error"] = None
    return trials


def evaluate_best_on_test(best, splits, gold_by_sentence, n_jobs=1):
    fit_rows = pd.concat([splits["train"], splits["validation"]])
    all_rows = pd.concat(splits.values())
    rows = []
    for trial in best.itertuples():
        vectors = load_vectors(trial.features, trial.reducer, fit_rows, all_rows)
        model = build_model(
            MODEL_BY_NAME[trial.model],
            json.loads(trial.params),
            vectors,
            trial.reducer,
            n_jobs=n_jobs,
        )
        scores = fit_and_score(model, fit_rows, splits["test"], gold_by_sentence)
        rows.append(
            {
                "model": trial.model,
                "features": trial.features,
                "reducer": trial.reducer,
                "params": trial.params,
                "val_f1": trial.f1,
                **{f"test_{k}": v for k, v in scores.items()},
            }
        )
    return pd.DataFrame(rows)


def format_ci(ci, signed=False):
    if ci is None:
        return ""
    fmt = "{:+.3f}" if signed else "{:.3f}"
    return f"[{fmt.format(ci[0])}, {fmt.format(ci[1])}]"


def add_uncertainty(frame, baseline_word_f1, prefix=""):
    def ci(word_f1, baseline=None, signed=False):
        if not isinstance(word_f1, dict):
            return ""
        return format_ci(bootstrap_ci(word_f1, baseline), signed=signed)

    frame = frame.copy()
    word_f1 = frame[f"{prefix}word_f1"]
    frame[f"{prefix}f1_95ci"] = word_f1.map(ci)
    frame[f"{prefix}gain_vs_1cluster"] = frame[f"{prefix}f1"] - float(
        np.mean(list(baseline_word_f1.values()))
    )
    frame[f"{prefix}gain_95ci"] = word_f1.map(
        lambda w: ci(w, baseline_word_f1, signed=True)
    )
    return frame


def format_params(params_json):
    params = json.loads(params_json)
    return ", ".join(f"{k}={v}" for k, v in params.items() if v is not None)


def add_table(md, frame):
    frame = frame.copy()
    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            digits = 2 if col.endswith("clusters_per_word") else 3
            frame[col] = frame[col].map(
                lambda v: "" if pd.isna(v) else f"{v:.{digits}f}"
            )
    md.new_line()
    md.new_table(
        columns=len(frame.columns),
        rows=len(frame) + 1,
        text=[*map(str, frame.columns), *frame.astype(str).to_numpy().ravel()],
        text_align="left",
    )


def write_report(
    path, args, splits, trials, baselines, test_results, test_baselines, elapsed
):
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
    uncertainty_cols = ["f1_95ci", "gain_vs_1cluster", "gain_95ci"]

    def readable(frame, extra=()):
        frame = frame[display_cols + list(extra)].copy()
        frame["params"] = frame["params"].map(format_params)
        return frame

    def one_cluster_word_f1(frame):
        return frame.set_index("baseline").loc[ONE_CLUSTER, "word_f1"]

    def baseline_table(frame, word_f1_one):
        return add_uncertainty(frame, word_f1_one)[
            [
                "baseline",
                "precision",
                "recall",
                "f1",
                "f1_95ci",
                "avg_clusters_per_word",
                "gold_clusters_per_word",
            ]
        ]

    def section(level, title, frame, text=None):
        md.new_header(level, title, add_table_of_contents="n")
        if text:
            md.new_paragraph(text)
        add_table(md, frame)

    val_one = one_cluster_word_f1(baselines)
    gold_val = baselines["gold_clusters_per_word"].iloc[0]
    best_shown = add_uncertainty(best, val_one) if "word_f1" in best else best
    best_shown = best_shown.assign(gold_clusters_per_word=gold_val)
    gold_test = (
        ""
        if test_baselines is None
        else f", test {test_baselines['gold_clusters_per_word'].iloc[0]:.2f}"
    )

    md = MdUtils(
        file_name=str(path),
        title="Hyperparameter tuning report",
        title_header_style="atx",
    )
    md.new_list(
        [
            f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"Mode: {'quick' if args.quick else 'full'}",
            f"Models: {', '.join(MODELS[m].display_name for m in args.models)}",
            f"Features: {', '.join(args.features)}",
            f"Words: {splits['validation'].word.nunique()}",
            "Rows: "
            + ", ".join(f"{name} {len(rows)}" for name, rows in splits.items()),
            f"Trials: {len(trials)} ({failed} failed)",
            f"Runtime: {elapsed / 60:.1f} min",
            "Selection metric: validation BCubed F1 (Subtask 1a), macro-averaged over target words.",
            "Models are fit on the training split; validation gold labels are used only for selection.",
            f"Gold senses per word: validation {gold_val:.2f}{gold_test}",
            f"Confidence intervals: 95% percentile bootstrap over target words "
            f"({N_BOOTSTRAP} resamples). `gain_95ci` is the paired difference from the "
            "one-cluster baseline on the same words; an interval that excludes 0 means "
            "the gain holds up beyond which words were sampled.",
        ]
    )
    section(
        2,
        "Best configuration per model (validation)",
        readable(
            best_shown,
            extra=["gold_clusters_per_word"]
            + [c for c in uncertainty_cols if c in best_shown],
        ),
    )

    if test_results is not None:
        test_one = one_cluster_word_f1(test_baselines)
        test_display = add_uncertainty(test_results, test_one, prefix="test_")
        test_display = test_display.drop(columns="test_word_f1")
        test_display["params"] = test_display["params"].map(format_params)
        section(
            2,
            "Best configurations on the held-out test split",
            test_display,
            "Refit on train + validation, then score Subtask 1a once on test.",
        )
        section(
            2, "Baselines (held-out test)", baseline_table(test_baselines, test_one)
        )

    section(2, "Baselines (validation)", baseline_table(baselines, val_one))
    md.new_header(
        2,
        "Best validation BCubed F1 (1a) by model and reducer",
        add_table_of_contents="n",
    )
    for features, group in ok.groupby("features"):
        searched = [r for r in SEARCH_REDUCERS if r in group.reducer.unique()]
        pivot = (
            group.pivot_table(
                index="model", columns="reducer", values="f1", aggfunc="max"
            )
            .reindex(columns=searched + sorted(set(group.reducer) - set(searched)))
            .reset_index()
        )
        section(3, f"Features: `{features}`", pivot)

    md.new_header(
        2, f"Top {args.top} configurations per model", add_table_of_contents="n"
    )
    for model_name, group in ranked.groupby("model", sort=False):
        section(3, model_name, readable(group.head(args.top)))

    if failed:
        errors = trials[trials["error"].notna()][
            ["model", "features", "reducer", "params", "error"]
        ]
        section(2, "Failed configurations", errors)

    md.create_md_file()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reducers", nargs="+", choices=SEARCH_REDUCERS)
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-words", type=int, default=DEFAULT_N_WORDS)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS), default=["correlation"]
    )
    parser.add_argument(
        "--features", nargs="+", choices=list(ENCODERS), default=list(ENCODERS)
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()
    splits = build_synthetic_splits()
    if args.n_words:
        splits = subsample_splits(splits, args.n_words)
    gold_by_sentence = synthetic_gold_by_sentence(pd.concat(splits.values()))
    splits = {name: rows[["word", "sentence_id"]] for name, rows in splits.items()}
    print(
        f"{splits['validation'].word.nunique()} words: "
        + ", ".join(f"{name} {len(rows)}" for name, rows in splits.items())
    )

    out_dir = args.output_dir / f"tuning_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "n_words": args.n_words,
        "words": sorted(splits["validation"].word.unique().tolist()),
    }
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    tfidf_source = PrecomputedFeatures(TfidfEmbeddings().load())
    baselines = baseline_scores(
        splits["train"], splits["validation"], gold_by_sentence, tfidf_source, args.jobs
    )

    trials = run_search(args, splits, gold_by_sentence, out_dir / "trials.jsonl")
    trials.sort_values("f1", ascending=False).drop(
        columns="word_f1", errors="ignore"
    ).to_csv(out_dir / "trials.csv", index=False)

    ok = trials[trials["error"].isna()].sort_values("f1", ascending=False)
    if ok.empty:
        raise SystemExit(f"Every trial failed, see {out_dir / 'trials.csv'}")
    best = ok.groupby("model", sort=False).head(1)

    test_results = test_baselines = None
    if not args.no_test:
        test_results = evaluate_best_on_test(best, splits, gold_by_sentence, args.jobs)
        test_baselines = baseline_scores(
            pd.concat([splits["train"], splits["validation"]]),
            splits["test"],
            gold_by_sentence,
            tfidf_source,
            args.jobs,
        )
        test_results.drop(columns="test_word_f1").to_csv(
            out_dir / "best_test.csv", index=False
        )
        test_baselines.drop(columns="word_f1").to_csv(
            out_dir / "baseline_test.csv", index=False
        )

    write_report(
        out_dir / "report.md",
        args,
        splits,
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
