"""
Compare whole-sentence TF-IDF with local context windows using identical folds.
Run from repo root: uv run python window_experiments.py

Uses the baseline project's KmeansBaselineModel and bcubedF1. This is a
5-fold comparison matching the original main.py, not the newer 70/15/15 split.
"""

from pathlib import Path  # manage output folders
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold  # split data into folds
from main import build_train, bcubedF1  # reuse Joseph's data loading and evaluation
from kmeans_model import KmeansBaselineModel  # reuse the baseline model
from feature_engineering import extract_context  # use my context extraction

OUT = Path('results/laila_context_windows')  # where results are saved
WINDOWS = [None, 3, 5, 10]  # None means whole sentence


def score_by_word(frame, predictions, gold_lookup):
    """Calculate B-Cubed scores separately for each target word."""
    per_word = []

    for word, subset in frame.groupby('word', sort=True):  # evaluate each word separately
        indices = subset.index.to_numpy()
        labels = [predictions[i] for i in indices]  # predicted sense clusters

        gold_sets = []
        for row in subset.itertuples():
            senses = gold_lookup[row.sentence_id]  # actual annotated senses
            gold_sets.append({f'{word}:{sense}' for sense in senses})

        if not gold_sets or any(not s for s in gold_sets):
            raise ValueError(f'Empty gold labels for {word}')

        p, r, f1 = bcubedF1(gold_sets, labels)  # calculate clustering accuracy
        per_word.append({
            'word': word,
            'precision': p,
            'recall': r,
            'b3_f1': f1,
            'n': len(subset)
        })

    return pd.DataFrame(per_word)


def main():
    OUT.mkdir(parents=True, exist_ok=True)  # create results folder

    data = build_train().reset_index(drop=True)  # load and preprocess usages
    gold = pd.read_parquet('data/subtask1.parquet')  # load gold annotations
    gold_lookup = gold.set_index('sentence_id').label.to_dict()

    missing = set(data.sentence_id) - set(gold_lookup)
    if missing:
        raise ValueError(f'{len(missing)} usages missing gold annotations')

    # Use identical folds for a fair comparison
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(data))

    all_rows = []
    word_rows = []

    for window in WINDOWS:  # test each context size
        name = 'whole_sentence' if window is None else f'window_{window}'
        print('\nRunning', name, flush=True)

        prepared = data.copy()

        if window is not None:
            prepared['context_tokens'] = prepared.apply(
                lambda row: extract_context(row, window), axis=1
            )  # replace whole-sentence tokens with nearby words

        print('Mean context tokens:', round(prepared.context_tokens.map(len).mean(), 2))

        for fold, (train_idx, val_idx) in enumerate(folds, 1):
            train = prepared.iloc[train_idx].reset_index(drop=True)
            val = prepared.iloc[val_idx].reset_index(drop=True)

            model = KmeansBaselineModel(random_state=42 + fold)  # create K-Means model

            try:
                model.fit(train)  # train on training fold
                pred = model.predict(val)  # predict clusters on validation fold

            except ValueError as exc:
                # Retry if small windows leave too few TF-IDF features
                if 'empty vocabulary' not in str(exc).lower() and 'no terms remain' not in str(exc).lower():
                    raise

                print('  Empty TF-IDF vocabulary; retrying with min_df=1')
                model = KmeansBaselineModel(
                    random_state=42 + fold, min_df=1
                ).fit(train)
                pred = model.predict(val)

            per_word = score_by_word(val, pred, gold_lookup)  # evaluate each word

            for row in per_word.to_dict('records'):
                word_rows.append({'context': name, 'fold': fold, **row})

            all_rows.append({
                'context': name,
                'fold': fold,
                'precision': per_word.precision.mean(),
                'recall': per_word.recall.mean(),
                'b3_f1': per_word.b3_f1.mean(),  # average F1 across words
                'words_evaluated': len(per_word)
            })

            print(
                f'  Fold {fold}: macro B3-F1={per_word.b3_f1.mean():.4f}',
                flush=True
            )

    results = pd.DataFrame(all_rows)

    results.to_csv(OUT/'fold_results.csv', index=False)  # save fold scores
    pd.DataFrame(word_rows).to_csv(
        OUT/'per_word_results.csv', index=False
    )  # save scores for each target word

    summary = results.groupby('context', as_index=False)[
        ['precision', 'recall', 'b3_f1']
    ].mean()  # average scores across folds

    summary['delta_vs_whole_sentence'] = (
        summary.b3_f1 -
        float(summary.loc[
            summary.context == 'whole_sentence', 'b3_f1'
        ].iloc[0])
    )  # calculate improvement or decrease from baseline

    summary.to_csv(OUT/'comparison.csv', index=False)  # save final comparison

    print('\nEXPERIMENT RESULTS (macro over words, then mean over folds)')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nSaved to', OUT)


if __name__ == '__main__':
    main()  # start experiments
