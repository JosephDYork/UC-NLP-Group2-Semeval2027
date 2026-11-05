import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaModel

from .utils import target_span


class XLMRobertaEmbeddings:
    def __init__(self, model_name="xlm-roberta-base", batch_size=32):
        self.device = "cuda"
        self.path = "embeddings/xlm_roberta_embeddings.npy"
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer: XLMRobertaTokenizer | None = None
        self.model: XLMRobertaModel | None = None

    def _load_model(self):
        if self.model is None:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(
                self.model_name, use_fast=True
            )
            self.model = (
                XLMRobertaModel.from_pretrained(self.model_name).to(self.device).eval()
            )

        return self.tokenizer, self.model

    def generate(self, rows, target_token):
        tokenizer, model = self._load_model()
        target_spans = [
            target_span(
                row.text,
                target_token,
                getattr(row, "start", None),
                getattr(row, "end", None),
            )
            for row in rows.itertuples()
        ]
        encoded = tokenizer(
            rows.text.tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        model_inputs = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**model_inputs).last_hidden_state.cpu().numpy()

        embeddings = []
        for (start, end), row_offsets, row_hidden in zip(
            target_spans, offsets.numpy(), hidden
        ):
            target_positions = np.flatnonzero(
                (row_offsets[:, 0] < end) & (row_offsets[:, 1] > start)
            )
            embeddings.append(row_hidden[target_positions].mean(axis=0))

        return np.vstack(embeddings)

    def load(self):
        return np.load(self.path)
