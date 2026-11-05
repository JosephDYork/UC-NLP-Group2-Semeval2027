import re


def target_span(text, target_token, start=None, end=None):
    match = re.search(re.escape(target_token), text, re.IGNORECASE)
    if text[start:end].strip().casefold() != target_token.strip().casefold():
        raise ValueError(
            f"Target token {target_token!r} does not match the annotated span in {text!r}"
        )

    if match is None:
        raise ValueError(f"Target token {target_token!r} is not present in {text!r}")

    return match.span()
