import argparse
import numpy as np
from pathlib import Path
from main import build_synthetic_train, build_train
from features import (
    TfidfEmbeddings,
    XLMRobertaEmbeddings,
    XLLexemeEmbeddings,
    CALEXLMREmbeddings,
)


def generate_target_embeddings(rows, embedding_model, output_path=None):
    rows = rows.copy()
    rows["_embedding_position"] = np.arange(len(rows))
    all_embeddings = None

    for target_form, form_rows in rows.groupby("target_form", sort=False):
        target_embeddings = embedding_model.generate(form_rows, target_form)
        if all_embeddings is None:
            shape = (len(rows), target_embeddings.shape[1])
            all_embeddings = (
                np.lib.format.open_memmap(
                    output_path, mode="w+", dtype=target_embeddings.dtype, shape=shape
                )
                if output_path is not None
                else np.empty(shape, dtype=target_embeddings.dtype)
            )
        all_embeddings[form_rows._embedding_position.to_numpy()] = target_embeddings

    if output_path is not None:
        all_embeddings.flush()

    return all_embeddings


def main():
    embedding_dir = Path("embeddings")
    embedding_dir.mkdir(parents=True, exist_ok=True)
    train = build_synthetic_train()

    print("Building TF-IDF embeddings...")
    TfidfEmbeddings().save(train)

    print(f"Building xlm-roberta embeddings...")
    model = XLMRobertaEmbeddings()
    embeddings = generate_target_embeddings(train, model)
    np.save(model.path, embeddings)

    print(f"Building xl-lexeme embeddings...")
    model = XLLexemeEmbeddings()
    embeddings = generate_target_embeddings(train, model)
    np.save(model.path, embeddings)

    print("Building CALE-XLM-R embeddings...")
    model = CALEXLMREmbeddings()
    embeddings = generate_target_embeddings(train, model)
    np.save(model.path, embeddings)


if __name__ == "__main__":
    main()
