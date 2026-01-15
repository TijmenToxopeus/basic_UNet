from .base import PruneOutput, BasePruningMethod
from .l1_norm import L1NormPruning
from .similar_feature import SimilarFeaturePruning


_METHODS = {
    "l1_norm": L1NormPruning(),
    "correlation": SimilarFeaturePruning(),
}


def get_method(name: str) -> BasePruningMethod:
    if name not in _METHODS:
        raise ValueError(f"Unknown pruning method '{name}'. Available: {list(_METHODS)}")
    return _METHODS[name]
