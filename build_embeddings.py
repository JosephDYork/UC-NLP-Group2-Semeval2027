import quiet  # noqa: F401
import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from build_datasets import load_dataset
from features import ENCODERS, TfidfEmbeddings


def write_embeddings(rows, encoder, path, desc=None):
    path = Path(path)
    partial = path.with_suffix(".partial.npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-rows.text.str.len().to_numpy(), kind="stable")
    embeddings = None

    with tqdm(total=len(rows), desc=desc, unit="row", dynamic_ncols=True) as progress:
        for batch_start in range(0, len(rows), encoder.batch_size):
            positions = order[batch_start : batch_start + encoder.batch_size]
            batch = encoder.generate(rows.iloc[positions])

            if embeddings is None:
                embeddings = np.lib.format.open_memmap(
                    partial,
                    mode="w+",
                    dtype=batch.dtype,
                    shape=(len(rows), batch.shape[1]),
                )

            embeddings[positions] = batch
            progress.update(len(positions))

    embeddings.flush()
    del embeddings
    os.replace(partial, path)


def build_tfidf(rows, overwrite=False):
    tfidf = TfidfEmbeddings()
    path = Path(tfidf.path)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            return path

    print(f"Building tfidf, {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tfidf.save(rows)
    return path


def build_embeddings(rows, names=None, device="cuda", batch_size=None, overwrite=False):
    paths = {}
    names = list(ENCODERS) if names is None else names
    for name in names:
        encoder = ENCODERS[name](batch_size=batch_size, device=device)
        path = Path(encoder.path)
        paths[name] = path
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                continue

        print(f"Building {name}, {path}")
        write_embeddings(rows, encoder, path, desc=name)
        del encoder

    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--encoders",
        nargs="*",
        choices=list(ENCODERS),
        default=list(ENCODERS),
    )

    args = parser.parse_args()
    rows = load_dataset()

    build_tfidf(rows, overwrite=args.overwrite)
    build_embeddings(
        rows,
        names=args.encoders,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
