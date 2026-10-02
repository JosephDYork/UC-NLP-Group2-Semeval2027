import pandas as pd
from tokenization import context_tokens


def build_train_set(n_neighbors):
    train = pd.read_parquet("data/usages.parquet")
    train["target_form"] = train.apply(lambda row: row.text[int(row.start) : int(row.end)].lower().strip(), axis=1)
    train["context_tokens"] = train.apply(lambda row: context_tokens(row, window=n_neighbors//2), axis=1)
    train["context_size"] = train.context_tokens.map(len)

    return train