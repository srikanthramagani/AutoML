"""
ModelRanker
===========
Responsibility: Rank evaluated models and generate formatted leaderboard
with overfitting and generalization awareness.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from .cross_validation import CVResult


@dataclass
class RankedModel:
    """
    Entry in the ranked leaderboard.
    """
    rank: int
    model_name: str
    mean_score: float
    std_score: float
    scoring: str
    train_score: Optional[float] = None
    generalization_gap: Optional[float] = None
    fit_status: str = "well_fitted"


class ModelRanker:
    """
    Rank models by cross-validation score and generalization stability.
    """

    def __init__(
        self,
        higher_is_better: bool = True,
        penalize_overfit: bool = True,
        overfit_penalty_weight: float = 0.25,
    ) -> None:
        self.higher_is_better = higher_is_better
        self.penalize_overfit = penalize_overfit
        self.overfit_penalty_weight = overfit_penalty_weight

    def rank(
        self,
        cv_results: List[CVResult],
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> List[RankedModel]:
        """
        Rank models descending by mean score, factoring in generalization gap when provided.
        """
        valid_results = [result for result in cv_results if result.is_valid]
        if not valid_results:
            raise ValueError("No valid model evaluation results are available for ranking.")

        def sort_key(r: CVResult):
            diag = diagnostics.get(r.model_name) if diagnostics else None
            score = r.mean_score
            if self.penalize_overfit and diag is not None:
                gap = getattr(diag, "generalization_gap", 0.0)
                status = getattr(diag, "status", "well_fitted")
                if status == "overfitting" and gap > 0.12:
                    penalty = self.overfit_penalty_weight * (gap - 0.12)
                    score = (score - penalty) if self.higher_is_better else (score + penalty)
            return (score, -r.std_score)

        sorted_results = sorted(
            valid_results,
            key=sort_key,
            reverse=self.higher_is_better,
        )

        ranked = []
        for i, r in enumerate(sorted_results):
            diag = diagnostics.get(r.model_name) if diagnostics else None
            train_s = getattr(diag, "train_score", None) if diag else None
            gap_val = getattr(diag, "generalization_gap", None) if diag else None
            status_val = getattr(diag, "status", "well_fitted") if diag else "well_fitted"

            ranked.append(
                RankedModel(
                    rank=i + 1,
                    model_name=r.model_name,
                    mean_score=round(r.mean_score, 4),
                    std_score=round(r.std_score, 4),
                    scoring=r.scoring,
                    train_score=round(float(train_s), 4) if train_s is not None else None,
                    generalization_gap=round(float(gap_val), 4) if gap_val is not None else None,
                    fit_status=status_val,
                )
            )
        return ranked

    def leaderboard_str(self, ranked_models: List[RankedModel]) -> str:
        """
        Produce a formatted tabular leaderboard string.
        """
        has_train = any(m.train_score is not None for m in ranked_models)
        if has_train:
            header = f"{'Rank':<5} | {'Model Name':<28} | {'CV Mean':>10} | {'Train':>10} | {'Gap':>8} | {'Fit Status':<14}"
            sep = "-" * len(header)
            rows = [sep, header, sep]
            for m in ranked_models:
                train_disp = f"{m.train_score:.4f}" if m.train_score is not None else "-"
                gap_disp = f"{m.generalization_gap:.4f}" if m.generalization_gap is not None else "-"
                rows.append(
                    f"{m.rank:<5} | {m.model_name:<28} | {m.mean_score:>10.4f} | {train_disp:>10} | {gap_disp:>8} | {m.fit_status:<14}"
                )
            rows.append(sep)
            return "\n".join(rows)

        header = f"{'Rank':<5} | {'Model Name':<30} | {'CV Mean Score':>14} | {'Std Dev':>10}"
        sep = "-" * len(header)
        rows = [sep, header, sep]
        for m in ranked_models:
            rows.append(
                f"{m.rank:<5} | {m.model_name:<30} | {m.mean_score:>14.4f} | {m.std_score:>10.4f}"
            )
        rows.append(sep)
        return "\n".join(rows)
