from .targets import MarkedTargets, OffsetTargets
from .tfidf import TfidfEmbeddings
from .xl_lexeme import XLLexemeCLSEmbeddings, XLLexemeEmbeddings
from .xl_durel import XLDURelCLSEmbeddings, XLDURelEmbeddings, XLDURelSpanEmbeddings
from .xlm_roberta import XLMRobertaCLSEmbeddings, XLMRobertaEmbeddings
from .mm_bert import MMBertEmbeddings
from .base import (
    ClassSpanPooledEmbeddingsModel,
    EmbeddingsModel,
    MeanPooledEmbeddingsModel,
    SpanPooledEmbeddingsModel,
)

ENCODERS = {
    "xl_lexeme": XLLexemeEmbeddings,
    "xl_durel": XLDURelEmbeddings,
    "xl_durel_span": XLDURelSpanEmbeddings,
    "xlm_roberta_large": XLMRobertaEmbeddings,
    "mm_bert": MMBertEmbeddings,
    "xl_lexeme_cls": XLLexemeCLSEmbeddings,
    "xl_durel_cls": XLDURelCLSEmbeddings,
    "xlm_roberta_large_cls": XLMRobertaCLSEmbeddings,
}

__all__ = [
    "ENCODERS",
    "EmbeddingsModel",
    "MeanPooledEmbeddingsModel",
    "SpanPooledEmbeddingsModel",
    "ClassSpanPooledEmbeddingsModel",
    "MarkedTargets",
    "OffsetTargets",
    "TfidfEmbeddings",
    "XLLexemeEmbeddings",
    "XLLexemeCLSEmbeddings",
    "XLDURelEmbeddings",
    "XLDURelSpanEmbeddings",
    "XLDURelCLSEmbeddings",
    "XLMRobertaEmbeddings",
    "XLMRobertaCLSEmbeddings",
    "MMBertEmbeddings",
]
