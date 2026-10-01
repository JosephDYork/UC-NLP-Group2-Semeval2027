from data import build_usages_table
from tokenization import context_tokens, tokenize_span


def main():
    usages = build_usages_table()

    sample = usages.iloc[0]
    tokens = tokenize_span(sample.text)
    context = context_tokens(sample)
    metadata = "\n".join(
        [
            f"Word: {sample.word}",
            f"Target form: {sample.target_form}",
            f"Sentence ID: {sample.sentence_id}",
            f"Period: {sample.period_label}",
            f"Year: {sample.year}",
            f"Span: ({sample.start}, {sample.end})",
        ]
    )

    print(f"Example Usage")
    print(f"=============")
    print(f"{metadata}")
    print()
    print(f"Text:")
    print(f"{sample.text}")
    print()
    print(f"Tokens with offsets:")
    print(f"{tokens}")
    print()
    print(f"Context tokens:")
    print(f"{context}")


if __name__ == "__main__":
    main()
