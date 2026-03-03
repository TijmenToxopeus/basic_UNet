from .base import PruneOutput, BasePruningMethod
from .l1_norm import L1NormPruning
from .l2_norm import L2NormPruning
from .pearson_correlation import PearsonCorrelationPruning
from .cosine_similarity import CosineSimilarityPruning


_METHODS = {
    "l1_norm": L1NormPruning(),
    "l2_norm": L2NormPruning(),
    "pearson_correlation": PearsonCorrelationPruning(),
    "cosine_similarity": CosineSimilarityPruning(),
    "cosine": CosineSimilarityPruning(),  # backward-compatible alias
    "correlation": PearsonCorrelationPruning(),  # backward-compatible alias
}


def get_method(name: str) -> BasePruningMethod:
    if name not in _METHODS:
        raise ValueError(f"Unknown pruning method '{name}'. Available: {list(_METHODS)}")
    return _METHODS[name]
