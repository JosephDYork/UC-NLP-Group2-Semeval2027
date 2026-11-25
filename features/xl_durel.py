from transformers import AutoTokenizer, XLMRobertaModel

from .targets import MarkedTargets
from .base import (
    ClassSpanPooledEmbeddingsModel,
    MeanPooledEmbeddingsModel,
    SpanPooledEmbeddingsModel,
)


class XLDURelEmbeddings(MeanPooledEmbeddingsModel):
    model_name = "sachinn1/xl-durel"
    path = "embeddings/xl_durel_embeddings.npy"
    tokenizer_class = AutoTokenizer
    model_class = XLMRobertaModel
    targets = MarkedTargets(template="{before}<t>{target} </t>{after}")


class XLDURelSpanEmbeddings(SpanPooledEmbeddingsModel, XLDURelEmbeddings):
    path = "embeddings/xl_durel_span_embeddings.npy"


class XLDURelCLSEmbeddings(ClassSpanPooledEmbeddingsModel, XLDURelEmbeddings):
    path = "embeddings/xl_durel_cls_embeddings.npy"
