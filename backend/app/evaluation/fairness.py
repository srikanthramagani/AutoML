"""Optional group fairness diagnostics for classification holdout predictions."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score


@dataclass
class FairnessAuditResult:
    status: str
    sensitive_columns: list[str]
    groups: Dict[str, Dict[str, float]]
    demographic_parity_difference: Optional[float] = None
    equal_opportunity_difference: Optional[float] = None
    note: Optional[str] = None


class FairnessAuditor:
    """Compute transparent group disparity metrics for binary classification."""

    def audit(
        self,
        sensitive_values: pd.DataFrame,
        y_true: Any,
        y_pred: Any,
        sensitive_columns: list[str],
    ) -> FairnessAuditResult:
        if len(sensitive_values) != len(y_true) or len(y_true) != len(y_pred):
            raise ValueError("Fairness inputs must contain the same number of rows.")
        if len(np.unique(y_true)) != 2:
            return FairnessAuditResult(
                status="not_available",
                sensitive_columns=sensitive_columns,
                groups={},
                note="Group audit currently supports binary classification only.",
            )

        groups: Dict[str, Dict[str, float]] = {}
        group_labels = sensitive_values.astype(str).agg(" | ".join, axis=1)
        positive_label = sorted(np.unique(y_true).tolist(), key=str)[-1]
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        for group in sorted(group_labels.unique()):
            mask = group_labels.to_numpy() == group
            group_true = y_true_arr[mask]
            group_pred = y_pred_arr[mask]
            groups[group] = {
                "sample_count": float(mask.sum()),
                "selection_rate": float(np.mean(group_pred == positive_label)),
                "true_positive_rate": float(
                    recall_score(
                        group_true,
                        group_pred,
                        labels=[positive_label],
                        average="macro",
                        zero_division=0,
                    )
                ),
            }

        selection_rates = [values["selection_rate"] for values in groups.values()]
        true_positive_rates = [values["true_positive_rate"] for values in groups.values()]
        return FairnessAuditResult(
            status="complete",
            sensitive_columns=sensitive_columns,
            groups=groups,
            demographic_parity_difference=round(max(selection_rates) - min(selection_rates), 4),
            equal_opportunity_difference=round(max(true_positive_rates) - min(true_positive_rates), 4),
            note="Differences are diagnostics, not automatic proof of discrimination or fairness.",
        )
