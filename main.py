import pandas as pd
from pathlib import Path


def main():
    directory_path = Path("./data")
    filenames = [file for file in directory_path.iterdir() if file.is_file()]
    dataframes = {file_path.name:
        pd.read_parquet(file_path)
        for file_path in filenames
    }

    print("TRAINING INPUTS:")
    print("=======================")
    for name, df in sorted(dataframes.items()):
        print(f"{name}: Length: {len(df)}, Columns: {df.columns.to_list()}")
    print("=======================")

    # directory_path = Path("./examples")
    # filenames = [file for file in directory_path.iterdir() if file.is_file()]
    # dataframes = {file_path.name:
        # pd.read_parquet(file_path)
        # for file_path in filenames
    # }

    forms = []
    for row in dataframes["usages.parquet"].itertuples():
        start = int(row[6])
        end = int(row[7])
        text = str(row[3])
        if str(row[1]) == "motiv":
            forms.append(text[start:end].lower().strip())

    print(pd.unique(pd.Series(forms)))

    # print()
    # print("EXAMPLE OUTPUTS:")
    # print("=======================")
    # for name, df in sorted(dataframes.items()):
        # print(f"{name}: Length: {len(df)}, Columns: {df.columns.to_list()}")
    # print("=======================")


if __name__ == "__main__":
    main()
