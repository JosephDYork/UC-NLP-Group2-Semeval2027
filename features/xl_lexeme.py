from transformers import XLMRobertaModel, XLMRobertaTokenizer

from .base import ClassSpanPooledEmbeddingsModel, SpanPooledEmbeddingsModel
from .targets import MarkedTargets


class XLLexemeEmbeddings(SpanPooledEmbeddingsModel):
    model_name = "pierluigic/xl-lexeme"
    path = "embeddings/xl_lexeme_embeddings.npy"
    tokenizer_class = XLMRobertaTokenizer
    model_class = XLMRobertaModel
    targets = MarkedTargets(template="{before} <t> {target} </t> {after}")


class XLLexemeCLSEmbeddings(ClassSpanPooledEmbeddingsModel, XLLexemeEmbeddings):
    path = "embeddings/xl_lexeme_cls_embeddings.npy"
