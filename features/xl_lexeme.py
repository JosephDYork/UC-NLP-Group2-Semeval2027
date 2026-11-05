import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaModel

from .utils import target_span


class XLLexemeEmbeddings:
    def __init__(self, model_name="pierluigic/xl-lexeme", batch_size=32):
        self.device = "cuda"
        self.path = "embeddings/xl_lexeme_embeddings.npy"
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer: XLMRobertaTokenizer | None = None
        self.model: XLMRobertaModel | None = None

    def _load_model(self):
        if self.model is None:
            # This whole <t> and </t> thing is kind of a hack to avoid a WordTransformer requests conflict
            tokenizer = XLMRobertaTokenizer.from_pretrained(
                self.model_name, use_fast=True
            )
            tokenizer.add_special_tokens({"additional_special_tokens": ["<t>", "</t>"]})
            model = XLMRobertaModel.from_pretrained(self.model_name).to(self.device)
            model.resize_token_embeddings(len(tokenizer))
            model.eval()
            self.tokenizer = tokenizer
            self.model = model

        return self.tokenizer, self.model

    @staticmethod
    def _mark_target(text, target_token, start, end):
        start, end = target_span(text, target_token, start, end)
        return f"{text[:start]} <t> {text[start:end]} </t> {text[end:]}"

    @staticmethod
    def _truncate_around_markers(token_list, marker_ids, max_content_length=126):
        if len(token_list) <= max_content_length:
            return token_list

        min_start = max(0, token_list.index(marker_ids[1]) - max_content_length + 1)
        max_start = min(
            token_list.index(marker_ids[0]), len(token_list) - max_content_length
        )
        window_start = min(
            max(token_list.index(marker_ids[0]) - max_content_length // 2, min_start),
            max_start,
        )

        return token_list[window_start : window_start + max_content_length]

    def generate(self, rows, target_token):
        tokenizer, model = self._load_model()
        marked_texts = [
            self._mark_target(text, target_token, start, end)
            for text, start, end in rows[["text", "start", "end"]].itertuples(
                index=False, name=None
            )
        ]

        input_ids = []
        marker_ids = tokenizer.convert_tokens_to_ids(["<t>", "</t>"])
        token_lists = tokenizer(marked_texts, add_special_tokens=False).input_ids
        for token_list in token_lists:
            token_list = self._truncate_around_markers(token_list, marker_ids)
            input_ids.append(
                [
                    tokenizer.bos_token_id,
                    *token_list,
                    tokenizer.eos_token_id,
                ]
            )

        encoded = tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        )
        model_inputs = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**model_inputs).last_hidden_state

        embeddings = []
        for row_hidden, row_ids in zip(hidden, model_inputs["input_ids"]):
            start = (row_ids == marker_ids[0]).nonzero().item()
            end = (row_ids == marker_ids[1]).nonzero().item()
            embeddings.append(row_hidden[start + 1 : end].mean(dim=0))

        return torch.stack(embeddings).cpu().numpy()

    def load(self):
        return np.load(self.path)
