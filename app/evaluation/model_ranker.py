"""
ModelRanker
===========
Responsibility: Rank evaluated models and generate formatted leaderboard.
"""

from dataclasses import dataclass
from typing import List
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


class ModelRanker:
    """
    Rank models by cross-validation score.
    """

    def __init__(self, higher_is_better: bool = True) -> None:
        self.higher_is_better = higher_is_better

    def rank(self, cv_results: List[CVResult]) -> List[RankedModel]:
        """
        Rank models descending by mean score, tie-breaking on lowest standard deviation.
        """
        valid_results = [result for result in cv_results if result.is_valid]
        if not valid_results:
            raise ValueError("No valid model evaluation results are available for ranking.")

        sorted_results = sorted(
            valid_results,
            key=lambda r: (r.mean_score, -r.std_score),
            reverse=self.higher_is_better,
        )
        ranked = [
            RankedModel(
                rank=i + 1,
                model_name=r.model_name,
                mean_score=round(r.mean_score, 4),
                std_score=round(r.std_score, 4),
                scoring=r.scoring,
            )
            for i, r in enumerate(sorted_results)
        ]
        return ranked

    def leaderboard_str(self, ranked_models: List[RankedModel]) -> str:
        """
        Produce a formatted tabular leaderboard string.
        """
        header = f"{'Rank':<5} | {'Model Name':<30} | {'CV Mean Score':>14} | {'Std Dev':>10}"
        sep = "-" * len(header)
        rows = [sep, header, sep]
        for m in ranked_models:
            rows.append(
                f"{m.rank:<5} | {m.model_name:<30} | {m.mean_score:>14.4f} | {m.std_score:>10.4f}"
            )
        rows.append(sep)
        return "\n".join(rows)
