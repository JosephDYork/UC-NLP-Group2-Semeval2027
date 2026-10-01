import re


TOKENIZER_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)
WORD_FILTER_RE = re.compile(r"^\w+(?:[-']\w+)*$", re.UNICODE)


def tokenize_span(text):
    return [
        (match.group(), match.start(), match.end())
        for match in TOKENIZER_RE.finditer(text)
    ]


def target_token_index(tokens, start, end):
    overlaps = [
        (idx, min(token_end, end) - max(token_start, start))
        for idx, (token, token_start, token_end) in enumerate(tokens)
        if token_start < end and start < token_end
    ]

    if not overlaps:
        return None

    return max(overlaps, key=lambda item: item[1])[0]


def context_tokens(row, window=8):
    tokens = tokenize_span(row.text)
    target_idx = target_token_index(tokens, int(row.start), int(row.end))

    if target_idx is None:
        return []

    left = max(0, target_idx - window)
    right = min(len(tokens), target_idx + window + 1)
    window_tokens = tokens[left:right]

    context = []
    for idx, (token, token_start, token_end) in enumerate(window_tokens, start=left):
        if idx == target_idx:
            continue
        if WORD_FILTER_RE.match(token) is None:
            continue

        context.append(token.lower())

    return context
