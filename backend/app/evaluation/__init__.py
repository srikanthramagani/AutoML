"""
Evaluation package.
Provides cross-validation, metric computation, and model ranking.
"""
from .cross_validation import CrossValidator, CVResult
from .metrics import MetricsCalculator, MetricsResult
from .model_ranker import ModelRanker, RankedModel
from .fairness import FairnessAuditResult, FairnessAuditor

__all__ = [
    "CrossValidator",
    "CVResult",
    "MetricsCalculator",
    "MetricsResult",
    "ModelRanker",
    "RankedModel",
    "FairnessAuditResult",
    "FairnessAuditor",
]
