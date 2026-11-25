from .base import BaseModel
from .correlation import CorrelationClusteringModel
from .gaussian import GaussianMixtureModel
from .hdbscan import HdbscanModel
from .kmeans import KmeansModel
from .single_label_baseline import SingleLabelBaselineModel

# Key -> model class, as tune.py and main.py refer to them.
MODELS = {
    "kmeans": KmeansModel,
    "hdbscan": HdbscanModel,
    "gmm": GaussianMixtureModel,
    "correlation": CorrelationClusteringModel,
}

__all__ = [
    "MODELS",
    "BaseModel",
    "CorrelationClusteringModel",
    "GaussianMixtureModel",
    "HdbscanModel",
    "KmeansModel",
    "SingleLabelBaselineModel",
]
