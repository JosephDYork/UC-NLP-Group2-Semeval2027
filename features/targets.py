import torch
import re


def target_span(text, target_token, start=None, end=None):
    if start is None or end is None:
        return re.search(re.escape(target_token), text, re.IGNORECASE).span()

    segment = text[start:end]
    if segment.strip().casefold() != target_token.strip().casefold():
        raise ValueError(
            f"Target token {target_token!r} does not match the span in {text!r}"
        )

    start += len(segment) - len(segment.lstrip())
    end -= len(segment) - len(segment.rstrip())
    return start, end


def target_spans(rows):
    return [
        target_span(text, target_token, start, end)
        for text, target_token, start, end in rows[
            ["text", "target_form", "start", "end"]
        ].itertuples(index=False, name=None)
    ]


class OffsetTargets:
    def __init__(self, max_length=512):
        self.max_length = max_length

    def prepare(self, tokenizer, model):
        pass

    def encode(self, tokenizer, rows):
        targets = target_spans(rows)
        spans = torch.tensor(targets)
        encoded = tokenizer(
            rows.text.tolist(),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offsets = encoded.pop("offset_mapping")
        target_mask = (offsets[:, :, 0] < spans[:, 1:]) & (
            offsets[:, :, 1] > spans[:, :1]
        )

        return dict(encoded), target_mask


class MarkedTargets:
    def __init__(
        self,
        template="{before} <t> {target} </t> {after}",
        markers=["<t>", "</t>"],
        max_content_length=126,
    ):
        self.template = template
        self.markers = markers
        self.max_content_length = max_content_length

    def prepare(self, tokenizer, model):
        tokenizer.add_special_tokens({"additional_special_tokens": self.markers})
        model.resize_token_embeddings(len(tokenizer))

    def mark(self, text, start, end):
        return self.template.format(
            before=text[:start], target=text[start:end], after=text[end:]
        )

    def truncate(self, token_list, marker_ids):
        if len(token_list) <= self.max_content_length:
            return token_list

        length = self.max_content_length
        open_at = token_list.index(marker_ids[0])
        close_at = token_list.index(marker_ids[1])
        min_start = max(0, close_at - length + 1)
        max_start = min(open_at, len(token_list) - length)
        window_start = min(max(open_at - length // 2, min_start), max_start)

        return token_list[window_start : window_start + length]

    def encode(self, tokenizer, rows):
        marked_texts = [
            self.mark(text, start, end)
            for text, (start, end) in zip(rows.text, target_spans(rows))
        ]

        marker_ids = tokenizer.convert_tokens_to_ids(self.markers)
        input_ids = [
            [
                tokenizer.bos_token_id,
                *self.truncate(token_list, marker_ids),
                tokenizer.eos_token_id,
            ]
            for token_list in tokenizer(
                marked_texts, add_special_tokens=False
            ).input_ids
        ]

        encoded = tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        )

        ids = encoded["input_ids"]
        start = (ids == marker_ids[0]).int().argmax(dim=1, keepdim=True)
        end = (ids == marker_ids[1]).int().argmax(dim=1, keepdim=True)
        positions = torch.arange(ids.shape[1])
        target_mask = (positions > start) & (positions < end)

        return dict(encoded), target_mask
