import quiet  # noqa: F401
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from evaluations import gold_jsd

USAGES_PATH = Path("data/usages.parquet")
GOLD_PATH = Path("data/subtask1.parquet")
SYNTHETIC_RAW_PATH = Path("synth_data/sythset_usages.parquet")
DATASET_PATH = Path("synth_data/preprocessed_dataset.parquet")
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
    return prepare_rows(pd.read_parquet(USAGES_PATH))


def load_dataset(path=DATASET_PATH):
    rows = pd.read_parquet(path)
    rows["context_tokens"] = rows.context_tokens.map(list)
    return rows


def gold_by_sentence():
    return pd.read_parquet(GOLD_PATH).set_index("sentence_id").label.to_dict()


def synthetic_gold_by_sentence(rows=None):
    if rows is None:
        rows = load_dataset()

    return rows.set_index("sentence_id").label.to_dict()


def synthetic_gold_jsd(rows=None):
    if rows is None:
        rows = load_dataset()

    return gold_jsd(rows, synthetic_gold_by_sentence(rows))


def _build_splits(rows, random_state):
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


def build_splits(random_state=SPLIT_RANDOM_STATE):
    return _build_splits(build_train(), random_state)


def build_synthetic_splits(random_state=SPLIT_RANDOM_STATE, rows=None):
    return _build_splits(load_dataset() if rows is None else rows, random_state)


def subsample_splits(splits, n_words, random_state=SPLIT_RANDOM_STATE):
    words = pd.Series(np.sort(splits["validation"].word.unique()))
    words = words.sample(frac=1, random_state=random_state).iloc[:n_words]

    return {name: rows[rows.word.isin(words)] for name, rows in splits.items()}


def bad_row_reasons(rows):
    start, end, text_length = (
        rows.start.astype(int),
        rows.end.astype(int),
        rows.text.str.len(),
    )
    tokens = rows.text.str.lower().str.findall(r"\w+")
    unique_ratio = tokens.map(lambda t: len(set(t)) / max(len(t), 1))
    invalid_span = (start < 0) | (end <= start) | (end > text_length)
    target = pd.Series(
        [text[s:e] if 0 <= s < e else "" for text, s, e in zip(rows.text, start, end)],
        index=rows.index,
    ).str.strip()

    return pd.DataFrame(
        {
            "invalid_span": invalid_span,
            "empty_target": target.eq(""),
            "partial_word": pd.Series(
                [
                    (s > 0 and text[s - 1].isalnum())
                    or (e < len(text) and text[e].isalnum())
                    for text, s, e in zip(rows.text, start, end)
                ],
                index=rows.index,
            )
            & ~invalid_span,
            "long_target": (
                target.str.split().str.len() > rows.lemma.str.split().str.len() + 2
            )
            | (target.str.len() > 50),
            "degenerate_text": ((unique_ratio < 0.3) & (tokens.map(len) >= 20))
            | (text_length > 1000),
            "duplicate": rows.duplicated(["text", "start", "end"]),
        }
    )


def preprocess_dataset(source=SYNTHETIC_RAW_PATH, output=DATASET_PATH, overwrite=False):
    if output.exists():
        if not overwrite:
            print(f"Keeping {output}, it exists")
            return output
        output.unlink()

    rows = pd.read_parquet(source)
    reasons = bad_row_reasons(rows)
    bad = reasons.any(axis=1)
    print(f"Read {len(rows)} rows from {source}")

    for reason, mask in reasons.items():
        print(f"  {reason:16s} {int(mask.sum()):6d}")

    print(f"Dropping {int(bad.sum())} rows (some match several reasons)")
    rows = rows[~bad].reset_index(drop=True)
    rows["period_label"] = rows["year"].astype(str)
    rows = prepare_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output)
    print(f"Wrote {len(rows)} rows to {output}")
    print("Rebuild the embeddings for these rows: build_embeddings.py --overwrite")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    preprocess_dataset(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
