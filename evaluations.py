import re
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

MIN_INSTANCES = 10
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42


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


def score_b3f1_by_word(rows, predictions, gold_by_sentence):
    predictions = np.asarray(predictions)
    sentence_ids = rows.sentence_id.to_numpy()
    word_scores = {}
    for word, positions in rows.groupby("word", sort=False).indices.items():
        gold_sets = [
            {str(sense) for sense in gold_by_sentence[sentence_id]}
            for sentence_id in sentence_ids[positions]
        ]
        word_scores[word] = bcubed_f1(gold_sets, predictions[positions])

    return pd.DataFrame.from_dict(
        word_scores, orient="index", columns=["precision", "recall", "f1"]
    )


def score_b3f1(rows, predictions, gold_by_sentence):
    return score_b3f1_by_word(rows, predictions, gold_by_sentence).mean().to_numpy()


def gold_clusters_per_word(rows, gold_by_sentence):
    senses = {}
    for word, sentence_id in zip(rows.word, rows.sentence_id):
        senses.setdefault(word, set()).update(
            str(sense) for sense in gold_by_sentence[sentence_id]
        )
    return float(np.mean([len(s) for s in senses.values()]))


def bootstrap_ci(word_f1, baseline_word_f1=None, n=N_BOOTSTRAP, level=0.95):
    words = sorted(word_f1)
    values = np.array([word_f1[w] for w in words])
    if baseline_word_f1 is not None:
        values = values - np.array([baseline_word_f1[w] for w in words])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return tuple(float(x) for x in np.percentile(means, [tail, 100 - tail]))


def as_label_set(label):
    if isinstance(label, (list, tuple, set, np.ndarray)):
        return {str(sense) for sense in label}
    return {str(label)}


def sense_counts(rows, labels, fractional=True):
    counts = defaultdict(lambda: defaultdict(Counter))
    for word, period, label in zip(rows.word, rows.period_label, labels):
        senses = as_label_set(label)
        weight = 1 / len(senses) if fractional else 1
        for sense in senses:
            counts[word][period][sense] += weight
    return counts


def period_sort_key(period_label):
    match = re.search(r"\d{4}", period_label)
    return (int(match.group()) if match else float("inf"), period_label)


def consecutive_jsd(counts):
    jsd = {}
    for word, by_period in counts.items():
        periods = sorted(
            (p for p, counter in by_period.items() if sum(counter.values()) > 0),
            key=period_sort_key,
        )
        senses = sorted({s for counter in by_period.values() for s in counter})
        for p1, p2 in zip(periods, periods[1:]):
            v1 = np.array([by_period[p1][s] for s in senses], dtype=float)
            v2 = np.array([by_period[p2][s] for s in senses], dtype=float)
            jsd[f"{word}_{p1}_{p2}"] = float(
                jensenshannon(v1 / v1.sum(), v2 / v2.sum(), base=2)
            )
    return jsd


def filter_min_instances(jsd, counts, min_instances=MIN_INSTANCES):
    filtered = {}
    for item, value in jsd.items():
        word, p1, p2 = item.rsplit("_", 2)
        if (
            sum(counts[word][p1].values()) >= min_instances
            and sum(counts[word][p2].values()) >= min_instances
        ):
            filtered[item] = value
    return filtered


def gold_jsd(rows, gold_by_sentence, fractional=True):
    counts = sense_counts(
        rows, [gold_by_sentence[s] for s in rows.sentence_id], fractional=fractional
    )
    return filter_min_instances(consecutive_jsd(counts), counts)


def spearman_1b(gold, predicted):
    missing = [item for item in gold if item not in predicted]
    if missing:
        raise KeyError(f"Predicted JSD missing for {missing[:3]}...")
    items = sorted(gold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr([gold[i] for i in items], [predicted[i] for i in items])
    return 0.0 if np.isnan(rho) else float(rho)


def score_jsd_spearman(rows, predictions, gold_by_sentence):
    gold = gold_jsd(rows, gold_by_sentence)
    predicted = consecutive_jsd(sense_counts(rows, predictions))
    return spearman_1b(gold, predicted)
