from pathlib import Path

import pandas as pd


def build_usages_table():
    dataframes = {path.name: pd.read_parquet(path) for path in Path("data").iterdir()}
    usages = dataframes["usages.parquet"].copy()
    usages["target_form"] = usages.apply(
        lambda row: row.text[int(row.start) : int(row.end)].lower().strip(), axis=1
    )

    return usages
