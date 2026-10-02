from data import build_train_set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def print_section(title):
    print(title)
    print("=" * len(title))


def main():
    n_neighbors = 8
    train = build_train_set(n_neighbors)

    vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=lambda tokens: tokens,
        preprocessor=lambda tokens: tokens,
        token_pattern=None,
        lowercase=False,
        min_df=2,
    )

    X = vectorizer.fit_transform(train.context_tokens)
    X = normalize(X, norm="l2", copy=False)
    feature_names = np.array(vectorizer.get_feature_names_out())

    print_section("Train Set")
    print(f"Rows: {len(train):,}")
    print(f"Columns: {', '.join(train.columns)}")
    print(f"Target words: {train.word.nunique():,}")
    print(f"Period labels: {train.period_label.nunique():,}")
    print(
        f"Context window: {n_neighbors // 2} tokens per side, up to {n_neighbors} total"
    )
    print(
        "Context size: "
        f"min={train.context_size.min()}, "
        f"median={train.context_size.median():.0f}, "
        f"max={train.context_size.max()}"
    )
    print()
    print("Rows per target word:")
    print(train.word.value_counts().sort_index().to_string())
    print()

    print_section("Vectorized Output")
    print(f"Matrix shape: {X.shape[0]:,} rows x {X.shape[1]:,} features")
    print(f"Non-zero values: {X.nnz:,}")
    print(f"Density: {X.nnz / (X.shape[0] * X.shape[1]):.4%}")
    print(f"Vocabulary size: {len(feature_names):,}")
    print()

    sample = train.iloc[2]
    sample_vector = X[2]
    top_indices = sample_vector.toarray().ravel().argsort()[::-1][:10]
    top_terms = [feature_names[idx] for idx in top_indices if sample_vector[0, idx] > 0]

    print_section("Vectorized Row")
    print(f"Word: {sample.word}")
    print(f"Sentence ID: {sample.sentence_id}")
    print(f"Target form: {sample.target_form}")
    print(f"Context tokens: {sample.context_tokens}")
    print(f"Top TF-IDF terms: {', '.join(top_terms)}")


if __name__ == "__main__":
    main()
