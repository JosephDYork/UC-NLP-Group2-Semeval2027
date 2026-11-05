import re 
from types import SimpleNamespace  # creates a sample object for testing

TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)  # identify tokens and punctuation
WORD_RE = re.compile(r"^\w+(?:[-']\w+)*$", re.UNICODE)  # identify valid word tokens


def extract_context(row, window_size=5):
    """Extract nearby context words around a target word."""
    tokens = [
        (m.group(), m.start(), m.end())  # store token and character positions
        for m in TOKEN_RE.finditer(row.text)
    ]

    if not tokens:
        return []  # handle empty text

    start, end = int(row.start), int(row.end)  # target word's character positions

    target_index = max(
        range(len(tokens)),
        key=lambda i: max(
            0,
            min(tokens[i][2], end) - max(tokens[i][1], start)
        )
    )  # find token overlapping the target word

    left = max(0, target_index - window_size)  # start of context window
    right = min(len(tokens), target_index + window_size + 1)  # end of window

    return [
        token.lower()  # normalize to lowercase
        for i, (token, _, _) in enumerate(tokens)
        if left <= i < right  # select nearby tokens
        and i != target_index  # exclude target word
        and WORD_RE.fullmatch(token)  # exclude punctuation
    ]


if __name__ == "__main__":  # execute example only when running this file
    sentence = "She deposited her money at the bank yesterday."
    start = sentence.index("bank")  # find target position

    example = SimpleNamespace(
        text=sentence,
        start=start,
        end=start + len("bank")
    )  # create a sample usage

    for window in [3, 5, 10]:  # test three window sizes
        print(f"Window {window}:",
              extract_context(example, window))
