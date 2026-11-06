import numpy as np
import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizer

from .utils import target_span


class CALEXLMREmbeddings:
    def __init__(self, model_name="gabrielloiseau/CALE-XLM-R", batch_size=32):
        self.device = "cuda"
        self.path = "embeddings/cale_xlmr_embeddings.npy"
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

    @staticmethod
    def _mark_target(text, target_token, start, end):
        start, end = target_span(text, target_token, start, end)
        return f"{text[:start]}<t>{text[start:end]}</t>{text[end:]}"

    def generate(self, rows, target_token):
        tokenizer, model = self._load_model()
        marked_texts = [
            self._mark_target(text, target_token, start, end)
            for text, start, end in rows[["text", "start", "end"]].itertuples(
                index=False, name=None
            )
        ]

        embeddings = []
        for batch_start in range(0, len(marked_texts), self.batch_size):
            encoded = tokenizer(
                marked_texts[batch_start : batch_start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            model_inputs = {
                name: value.to(self.device) for name, value in encoded.items()
            }
            with torch.inference_mode():
                hidden = model(**model_inputs).last_hidden_state

            # CALE-XLM-R uses sentence-transformers mean-token pooling.
            attention_mask = model_inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(
                dim=1
            ).clamp_min(1)
            embeddings.append(pooled.cpu().numpy())

        return np.vstack(embeddings)

    def load(self):
        return np.load(self.path)
