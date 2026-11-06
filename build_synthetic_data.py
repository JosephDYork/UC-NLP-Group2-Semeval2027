import argparse
from pathlib import Path

from main import build_synthetic_train


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate generated synthetic usages into a parquet dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synth_data/sythset_usages.parquet"),
    )
    args = parser.parse_args()

    rows = build_synthetic_train()
    # These are deterministically rebuilt by prepare_rows() when the parquet is
    # loaded, so avoid storing duplicate tokenized text.
    rows = rows.drop(columns=["context_tokens", "target_form"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(args.output, index=False, compression="zstd")
    print(f"Wrote {len(rows):,} rows to {args.output}")


if __name__ == "__main__":
    main()
