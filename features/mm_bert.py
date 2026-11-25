from transformers import GemmaTokenizer, ModernBertModel

from .base import SpanPooledEmbeddingsModel


class MMBertEmbeddings(SpanPooledEmbeddingsModel):
    model_name = "jhu-clsp/mmBERT-base"
    path = "embeddings/mm_bert_embeddings.npy"
    tokenizer_class = GemmaTokenizer
    model_class = ModernBertModel
