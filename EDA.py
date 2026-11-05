import re  # for finding words in text
from types import SimpleNamespace  # for creating a test example

TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)  # split text into tokens
WORD_RE = re.compile(r"^\w+(?:[-']\w+)*$", re.UNICODE)  # keep words, not punctuation


def extract_context(row, window_size=5):
    """Extract nearby context words around a target word."""

    tokens = [
        (m.group(), m.start(), m.end())  # token and its position in the sentence
        for m in TOKEN_RE.finditer(row.text)
    ]

    if not tokens:
        return []  # return empty if no tokens

    start, end = int(row.start), int(row.end)  # target word position

    target_index = max(
        range(len(tokens)),
        key=lambda i: max(
            0,
            min(tokens[i][2], end) - max(tokens[i][1], start)
        )
    )  # find the token matching the target word

    left = max(0, target_index - window_size)  # left window boundary
    right = min(len(tokens), target_index + window_size + 1)  # right boundary

    return [
        token.lower()  # convert words to lowercase
        for i, (token, _, _) in enumerate(tokens)
        if left <= i < right  # keep tokens inside the window
        and i != target_index  # exclude the target word
        and WORD_RE.fullmatch(token)  # exclude punctuation
    ]


if __name__ == "__main__":  # run this only when the file is executed directly
    sentence = "She deposited her money at the bank yesterday."
    start = sentence.index("bank")  # locate the target word

    example = SimpleNamespace(
        text=sentence,
        start=start,
        end=start + len("bank")
    )  # create a sample row

    for window in [3, 5, 10]:  # test different context sizes
        print(f"Window {window}:",
              extract_context(example, window))
