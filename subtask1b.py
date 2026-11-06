import re
import warnings
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


MIN_INSTANCES = 10


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
        warnings.simplefilter("ignore")  # constant input -> NaN, handled below
        rho, _ = spearmanr([gold[i] for i in items], [predicted[i] for i in items])
    # A constant prediction (e.g. one sense per word) gives an undefined rho.
    return 0.0 if np.isnan(rho) else float(rho)
