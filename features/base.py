import numpy as np
import torch

from .targets import OffsetTargets


class EmbeddingsModel:
    model_name = None
    path = None
    tokenizer_class = None
    model_class = None
    batch_size = 128
    targets = OffsetTargets()

    def __init__(self, model_name=None, batch_size=None, device="cuda"):
        self.device = torch.device(device)
        self.model_name = model_name or self.model_name
        self.batch_size = batch_size or self.batch_size
        self.tokenizer = None
        self.model = None

    def load_model(self):
        if self.model is None:
            tokenizer = self.tokenizer_class.from_pretrained(
                self.model_name, use_fast=True
            )
            model = self.model_class.from_pretrained(self.model_name).to(self.device)
            self.targets.prepare(tokenizer, model)
            self.tokenizer, self.model = tokenizer, model.eval()

        return self.tokenizer, self.model

    def pool(self, hidden, attention_mask, target_mask):
        raise NotImplementedError

    @staticmethod
    def masked_mean(hidden, mask):
        mask = mask.unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1)

    def generate(self, rows):
        tokenizer, model = self.load_model()
        encoded, target_mask = self.targets.encode(tokenizer, rows)
        model_inputs = {name: value.to(self.device) for name, value in encoded.items()}
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=torch.bfloat16),
        ):
            hidden = model(**model_inputs).last_hidden_state

        with torch.inference_mode():
            pooled = self.pool(
                hidden.float(),
                model_inputs["attention_mask"].bool(),
                target_mask.to(self.device),
            )

        return pooled.cpu().numpy()

    def load(self):
        return np.load(self.path, mmap_mode="r")


class MeanPooledEmbeddingsModel(EmbeddingsModel):
    def pool(self, hidden, attention_mask, target_mask):
        return self.masked_mean(hidden, attention_mask)


class SpanPooledEmbeddingsModel(EmbeddingsModel):
    def pool(self, hidden, attention_mask, target_mask):
        return self.masked_mean(hidden, target_mask)


class ClassSpanPooledEmbeddingsModel(SpanPooledEmbeddingsModel):
    def pool(self, hidden, attention_mask, target_mask):
        cls_tok = torch.nn.functional.normalize(hidden[:, 0], dim=-1)
        target = torch.nn.functional.normalize(
            super().pool(hidden, attention_mask, target_mask), dim=-1
        )

        return torch.cat([cls_tok, target], dim=-1)
