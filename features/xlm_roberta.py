from transformers import XLMRobertaModel, XLMRobertaTokenizer

from .base import ClassSpanPooledEmbeddingsModel, SpanPooledEmbeddingsModel


class XLMRobertaEmbeddings(SpanPooledEmbeddingsModel):
    model_name = "xlm-roberta-large"
    path = "embeddings/xlm_roberta_large_embeddings.npy"
    tokenizer_class = XLMRobertaTokenizer
    model_class = XLMRobertaModel
    batch_size = 32


class XLMRobertaCLSEmbeddings(ClassSpanPooledEmbeddingsModel, XLMRobertaEmbeddings):
    path = "embeddings/xlm_roberta_large_cls_embeddings.npy"
