"""Evidence-based checks for tabular model evaluation integrity."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split


@dataclass
class AuditConfig:
    """Configurable thresholds and weights used by the reliability audit."""

    high_accuracy_threshold: float = 0.99
    high_r2_threshold: float = 0.99
    minimum_test_size: int = 20
    large_generalization_gap: float = 0.15
    direct_mapping_threshold: float = 0.98
    minimum_mapping_samples: int = 20
    weights: Dict[str, float] = field(default_factory=lambda: {
        "leakage": 25.0,
        "split_integrity": 20.0,
        "validation": 20.0,
        "preprocessing": 15.0,
        "dataset_adequacy": 10.0,
        "reproducibility": 5.0,
        "metric_validity": 5.0,
    })


class EvaluationReliabilityAuditor:
    """Inspect an existing evaluation and return findings with their evidence."""

    _GROUP_NAME = re.compile(r"(^|_)(user|patient|subject|customer|device|account|group|entity)_?id($|_)", re.I)
    _DATE_NAME = re.compile(r"(date|time|timestamp|datetime|created_at|event_at)", re.I)

    def __init__(self, config: Optional[AuditConfig] = None) -> None:
        self.config = config or AuditConfig()

    def audit(
        self,
        dataframe: pd.DataFrame,
        target_column: Optional[str],
        task_type: str,
        model_name: str,
        metrics: Dict[str, Any],
        train_score: Optional[float] = None,
        validation_score: Optional[float] = None,
        validation_strategy: str = "unknown",
        cv_folds: Optional[int] = None,
        test_size: float = 0.2,
        random_seed: Optional[int] = None,
        preprocessing_fold_local: Optional[bool] = None,
        dataset_name: str = "dataset",
        feature_columns: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        findings: List[Dict[str, Any]] = []
        components: Dict[str, Dict[str, Any]] = {}
        supervised = bool(target_column and target_column in dataframe.columns)
        task_is_classification = "classification" in task_type.lower()
        features = [c for c in (feature_columns or dataframe.columns) if c in dataframe.columns and c != target_column]
        y = dataframe[target_column] if supervised else None
        row_count = int(len(dataframe))
        duplicates = int(dataframe.duplicated().sum())
        duplicate_pct = round(100.0 * duplicates / max(row_count, 1), 2)

        dataset_hash = hashlib.sha256(
            pd.util.hash_pandas_object(dataframe, index=True).values.tobytes()
        ).hexdigest()

        def add_finding(
            finding_id: str,
            title: str,
            severity: str,
            evidence: str,
            explanation: str,
            impact: str,
            recommendation: str,
            verified: bool,
        ) -> None:
            findings.append({
                "id": finding_id,
                "title": title,
                "severity": severity,
                "evidence": evidence,
                "explanation": explanation,
                "potential_impact": impact,
                "recommended_correction": recommendation,
                "verified": verified,
            })

        mapping_features: List[Dict[str, Any]] = []
        if supervised:
            for column in features:
                feature = dataframe[column]
                valid = feature.notna() & y.notna()
                paired = pd.DataFrame({"feature": feature[valid], "target": y[valid]})
                if len(paired) < self.config.minimum_mapping_samples or paired.empty:
                    continue
                agreement = paired.groupby("feature", dropna=False)["target"].nunique()
                mapped_fraction = float(
                    paired["feature"].isin(agreement.index[agreement == 1]).sum()
                    / max(len(paired), 1)
                )
                covered = float(len(paired) / max(y.notna().sum(), 1))
                low_cardinality_mapping = (
                    len(agreement) <= max(20, int(len(paired) * 0.2))
                    and mapped_fraction >= self.config.direct_mapping_threshold
                )
                exact_target_match = bool(
                    pd.api.types.is_numeric_dtype(feature)
                    and pd.api.types.is_numeric_dtype(y)
                    and np.allclose(
                        pd.to_numeric(feature[valid], errors="coerce").to_numpy(dtype=float),
                        pd.to_numeric(y[valid], errors="coerce").to_numpy(dtype=float),
                        equal_nan=True,
                    )
                )
                correlation = None
                if pd.api.types.is_numeric_dtype(feature) and pd.api.types.is_numeric_dtype(y):
                    numeric = pd.DataFrame({"feature": feature, "target": y}).dropna()
                    if len(numeric) >= self.config.minimum_mapping_samples:
                        corr = numeric.corr(method="spearman").iloc[0, 1]
                        correlation = float(corr) if np.isfinite(corr) else None
                if exact_target_match or (
                    low_cardinality_mapping
                    and covered >= 0.9
                ) or (correlation is not None and abs(correlation) >= 0.995):
                    evidence_parts = [
                        f"{len(paired)} non-missing paired rows",
                        f"{mapped_fraction:.1%} of feature values map to a single target value",
                    ]
                    if exact_target_match:
                        evidence_parts.append("feature values exactly equal target values")
                    if correlation is not None:
                        evidence_parts.append(f"Spearman correlation {correlation:.4f}")
                    mapping_features.append({
                        "feature": str(column),
                        "mapping_fraction": round(mapped_fraction, 4),
                        "coverage": round(covered, 4),
                        "spearman_correlation": round(correlation, 4) if correlation is not None else None,
                        "exact_target_match": exact_target_match,
                        "evidence": "; ".join(evidence_parts),
                    })

        if mapping_features:
            add_finding(
                "target-proxy-feature",
                "Feature may directly encode the target",
                "High",
                " | ".join(f"{item['feature']}: {item['evidence']}" for item in mapping_features),
                "A feature value strongly determines or reproduces the target in this dataset. This relationship can be legitimate, so verify how and when the feature is created.",
                "If the feature is derived from the target or unavailable at prediction time, the reported score may not generalize.",
                "Trace feature provenance and prediction-time availability; rerun evaluation with confirmed target-derived features excluded.",
                False,
            )
            components["leakage"] = {"status": "warning", "evidence": "Potential target proxies require provenance review."}
        elif supervised:
            components["leakage"] = {"status": "pass", "evidence": "No near-deterministic feature-target proxy was detected by the configured checks."}
        else:
            components["leakage"] = {"status": "not_checked", "evidence": "A supervised target was not available."}

        train_indices = np.arange(row_count)
        cross_split_duplicates = 0
        conflicting_duplicate_labels = 0
        if row_count > 1:
            try:
                stratify = y if supervised and task_is_classification else None
                train_ids, test_ids = train_test_split(
                    train_indices,
                    test_size=test_size,
                    random_state=random_seed if random_seed is not None else 42,
                    stratify=stratify,
                )
            except ValueError:
                train_ids, test_ids = train_test_split(
                    train_indices, test_size=test_size,
                    random_state=random_seed if random_seed is not None else 42,
                )
            row_keys = pd.util.hash_pandas_object(dataframe, index=False).astype(str)
            train_keys = set(row_keys.iloc[train_ids])
            test_keys = set(row_keys.iloc[test_ids])
            cross_split_duplicates = len(train_keys.intersection(test_keys))

            if supervised and features:
                feature_keys = pd.util.hash_pandas_object(dataframe[features], index=False).astype(str)
                train_feature_keys = set(feature_keys.iloc[train_ids])
                test_feature_keys = set(feature_keys.iloc[test_ids])
                shared_feature_keys = train_feature_keys.intersection(test_feature_keys)
                for key in shared_feature_keys:
                    feature_rows = dataframe.loc[feature_keys == key, target_column].dropna()
                    if feature_rows.nunique() > 1:
                        conflicting_duplicate_labels += 1

        if duplicates:
            add_finding(
                "duplicate-rows",
                "Duplicate records found",
                "Medium",
                f"{duplicates} duplicate full rows ({duplicate_pct:.2f}% of records); {cross_split_duplicates} identical full-row values are present on both sides of the configured split.",
                "Repeated records can make evaluation less representative. Identical feature vectors with conflicting labels are also ambiguous training examples.",
                "Duplicates crossing the split can inflate performance; duplicates in one partition still overweight repeated observations.",
                "Review duplicate provenance and choose a documented deduplication or group-splitting policy; records were not removed.",
                cross_split_duplicates > 0,
            )
            components["split_integrity"] = {
                "status": "fail" if cross_split_duplicates else "warning",
                "evidence": f"{cross_split_duplicates} complete-row values overlap the train/test split.",
            }
        else:
            components["split_integrity"] = {"status": "pass", "evidence": "No exact duplicate full rows were found."}

        if conflicting_duplicate_labels:
            add_finding(
                "conflicting-duplicate-labels",
                "Repeated feature vectors have conflicting labels",
                "High",
                f"{conflicting_duplicate_labels} shared feature-vector values map to more than one target value.",
                "The same observed features have different outcomes, which may represent label noise, omitted context, or a duplicate entity crossing the split.",
                "Performance estimates may be unstable and a memorizing model may be rewarded or penalized unpredictably.",
                "Inspect the records and consider a domain-appropriate group split or label correction.",
                True,
            )

        date_columns = [
            str(column) for column in dataframe.columns
            if column != target_column and (
                self._DATE_NAME.search(str(column))
                or pd.api.types.is_datetime64_any_dtype(dataframe[column])
            )
        ]
        group_columns = [
            str(column) for column in dataframe.columns
            if column != target_column and self._GROUP_NAME.search(str(column))
            and dataframe[column].nunique(dropna=True) < row_count
        ]
        if date_columns and "time" not in validation_strategy.lower() and "chronolog" not in validation_strategy.lower():
            add_finding(
                "temporal-validation-review",
                "Potential time-dependent data needs chronological validation",
                "Medium",
                f"Date/time-like columns detected: {', '.join(date_columns[:8])}. Recorded strategy: {validation_strategy}.",
                "Date columns can indicate temporal ordering, but their presence alone does not establish a forecasting objective.",
                "A random split may allow later-period information to inform evaluation of earlier periods.",
                "Confirm the prediction objective; if forecasting or future generalization is required, use chronological holdout and time-series cross-validation.",
                False,
            )
            components["validation"] = {"status": "warning", "evidence": "Potential time columns were found, but task intent is not inferred from names alone."}
        elif group_columns and "group" not in validation_strategy.lower():
            add_finding(
                "group-validation-review",
                "Repeated entities may cross the split",
                "Medium",
                f"Identifier-like repeated columns detected: {', '.join(group_columns[:8])}. Recorded strategy: {validation_strategy}.",
                "Repeated users, devices, patients, or other entities can violate independent-row assumptions when split randomly.",
                "A test set containing entities seen in training may overstate performance on new entities.",
                "Confirm the entity definition and use GroupShuffleSplit or GroupKFold when the deployment objective is generalization to unseen entities.",
                False,
            )
            components["validation"] = {"status": "warning", "evidence": "Repeated identifier candidates were detected; group semantics need confirmation."}
        else:
            components["validation"] = {"status": "pass", "evidence": f"Recorded validation strategy: {validation_strategy}; no incompatible strategy was confirmed by available metadata."}

        if preprocessing_fold_local is True:
            components["preprocessing"] = {"status": "pass", "evidence": "The AutoML workflow fits preprocessing inside the training pipeline and each CV fold."}
        else:
            components["preprocessing"] = {"status": "not_checked", "evidence": "Preprocessing fit boundaries were not provided to the auditor."}

        if duplicates > 0:
            # Duplicate counts are reported independently; they are not proof of overlap.
            pass
        if row_count < max(self.config.minimum_test_size * 2, 20):
            add_finding(
                "small-dataset",
                "Dataset may be too small for stable evaluation",
                "Medium",
                f"{row_count} total records; configured test fraction {test_size:.1%} yields approximately {int(row_count * test_size)} test records.",
                "Small test sets have high sampling uncertainty. The minimum size is a configurable screening threshold, not a universal statistical guarantee.",
                "A small number of predictions can make near-perfect metrics unstable.",
                "Collect more independent observations or report uncertainty intervals and repeated evaluation where appropriate.",
                True,
            )
            components["dataset_adequacy"] = {"status": "warning", "evidence": f"{row_count} rows is below the configured screening size."}
        else:
            components["dataset_adequacy"] = {"status": "pass", "evidence": f"{row_count} rows; configured minimum screening size passed."}

        metric_values_valid = all(
            isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value))
            for value in metrics.values()
        )
        if metric_values_valid:
            for name, value in metrics.items():
                metric_name = name.lower()
                numeric_value = float(value)
                if metric_name in {
                    "accuracy", "balanced_accuracy", "f1", "f1_weighted", "f1_macro",
                    "precision", "precision_weighted", "precision_macro",
                    "recall", "recall_weighted", "recall_macro", "roc_auc", "pr_auc",
                } and not 0.0 <= numeric_value <= 1.0:
                    metric_values_valid = False
                    break
                if metric_name == "r2" and numeric_value > 1.0:
                    metric_values_valid = False
                    break
                if metric_name in {"mcc", "matthews_corrcoef"} and not -1.0 <= numeric_value <= 1.0:
                    metric_values_valid = False
                    break
                if metric_name in {
                    "mae", "mse", "rmse", "mean_absolute_error", "mean_squared_error",
                    "max_error", "median_absolute_error", "mape", "smape",
                } and numeric_value < 0:
                    metric_values_valid = False
                    break
        if not metrics or not metric_values_valid:
            components["metric_validity"] = {"status": "fail", "evidence": "No finite numeric evaluation metrics were available."}
            add_finding(
                "invalid-metrics",
                "Evaluation metrics are missing or invalid",
                "Critical",
                "At least one metric is missing, non-numeric, or non-finite.",
                "The reported evaluation cannot be interpreted reliably without finite metrics.",
                "The evaluation result is not suitable for model comparison.",
                "Inspect the evaluation output and recompute metrics from aligned ground-truth and prediction arrays.",
                True,
            )
        else:
            components["metric_validity"] = {"status": "pass", "evidence": f"{len(metrics)} reported metrics are finite numeric values."}

        if task_is_classification and "accuracy" in metrics:
            score = float(metrics["accuracy"])
            near_perfect = score >= self.config.high_accuracy_threshold
            if near_perfect:
                add_finding(
                    "high-classification-score",
                    "Near-perfect classification score merits context",
                    "Informational",
                    f"Holdout accuracy {score:.4f}; configured flag threshold {self.config.high_accuracy_threshold:.4f}; test set approximately {int(row_count * test_size)} rows.",
                    "High scores can arise from genuinely separable data. The score alone is not evidence of leakage.",
                    "On a small test set, a few observations can materially change the metric.",
                    "Review feature provenance, split design, and independent-sample coverage; do not infer leakage from accuracy alone.",
                    False,
                )
        elif "r2" in metrics and float(metrics.get("r2", -np.inf)) >= self.config.high_r2_threshold:
            add_finding(
                "high-regression-score",
                "Near-perfect regression score merits context",
                "Informational",
                f"Holdout R² {float(metrics['r2']):.4f}; configured flag threshold {self.config.high_r2_threshold:.4f}.",
                "Near-perfect R² may be valid for a deterministic relationship and does not establish leakage.",
                "Check target-derived inputs and whether test observations match the intended deployment population.",
                "Review feature provenance and evaluate on an independent split aligned with deployment.",
                False,
            )

        if train_score is not None and validation_score is not None:
            gap = float(train_score) - float(validation_score)
            if gap > self.config.large_generalization_gap:
                add_finding(
                    "generalization-gap",
                    "Training and validation scores differ substantially",
                    "Medium",
                    f"Training score {train_score:.4f}; validation score {validation_score:.4f}; gap {gap:.4f} exceeds configured {self.config.large_generalization_gap:.4f}.",
                    "A large gap is consistent with overfitting but can also result from small or shifted samples.",
                    "The selected model may perform worse on unseen data than its training score suggests.",
                    "Review learning curves, model complexity, and validation population; consider regularization or simpler models.",
                    False,
                )

        components["reproducibility"] = {
            "status": "pass" if random_seed is not None and dataset_hash else "not_checked",
            "evidence": f"Dataset SHA-256 {dataset_hash[:16]}…; random seed {random_seed}."
            if random_seed is not None else "Random seed metadata was unavailable.",
        }

        weights = self.config.weights
        total_weight = float(sum(weights.values()))
        earned = 0.0
        for component, weight in weights.items():
            status = components.get(component, {}).get("status", "not_checked")
            if status == "pass":
                earned += float(weight)
            elif status == "warning":
                earned += float(weight) * 0.5
        executed = sum(1 for component in weights if components.get(component, {}).get("status") in {"pass", "warning", "fail"})
        reliability_score = round(100.0 * earned / total_weight, 1) if executed >= 4 and total_weight > 0 else None

        severities = {item["severity"] for item in findings}
        if any(item["severity"] == "Critical" and item["verified"] for item in findings):
            status = "INVALID EVALUATION"
        elif any(item["severity"] == "High" and item["verified"] for item in findings):
            status = "SUSPICIOUS"
        elif row_count < 20 or reliability_score is None:
            status = "INCONCLUSIVE"
        elif findings:
            status = "TRUSTWORTHY WITH WARNINGS"
        else:
            status = "TRUSTED WITHIN CHECKED CONDITIONS"

        return {
            "audit_version": "1.0",
            "dataset": {
                "name": dataset_name,
                "fingerprint": dataset_hash,
                "samples": row_count,
                "features": len(features),
                "feature_list": [str(column) for column in features],
                "target_column": target_column,
                "task_type": task_type,
                "missing_value_pct": round(float(dataframe.isna().to_numpy().mean() * 100), 2) if dataframe.size else 0.0,
                "duplicate_record_count": duplicates,
                "duplicate_record_pct": duplicate_pct,
            },
            "evaluation": {
                "model_name": model_name,
                "training_score": train_score,
                "validation_score": validation_score,
                "test_metrics": metrics,
                "validation_strategy": validation_strategy,
                "cv_folds": cv_folds,
                "test_set_size": int(row_count * test_size),
                "random_seed": random_seed,
            },
            "findings": findings,
            "reliability": {
                "score": reliability_score,
                "status": status,
                "components": components,
                "formula": "100 × earned component weight / total configured weight. Pass earns full weight, warning earns half, failed or unchecked earns zero; score is withheld until at least four weighted checks execute.",
                "weights": weights,
                "executed_checks": executed,
                "total_checks": len(weights),
            },
            "summary": {
                "finding_count": len(findings),
                "critical_count": sum(item["severity"] == "Critical" for item in findings),
                "high_count": sum(item["severity"] == "High" for item in findings),
                "duplicate_train_test_overlap": cross_split_duplicates,
                "conflicting_duplicate_labels": conflicting_duplicate_labels,
                "potential_leakage_features": mapping_features,
            },
        }


def run_corrected_evaluation(
    dataframe: pd.DataFrame,
    target_column: str,
    fitted_pipeline: Any,
    task_type: str,
    excluded_features: Optional[List[str]] = None,
    feature_columns: Optional[List[str]] = None,
    random_seed: int = 43,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """Retrain a cloned pipeline on a new split and evaluate its independent holdout."""
    from .metrics import MetricsCalculator

    if target_column not in dataframe.columns:
        return {"status": "not_possible", "reason": "The selected target column is unavailable."}
    if "classification" not in task_type.lower() and task_type.lower() != "regression":
        return {"status": "not_possible", "reason": "Corrected evaluation supports classification and regression only."}
    if len(dataframe) < 10:
        return {"status": "not_possible", "reason": "At least 10 rows are required for an independent corrected split."}

    requested_exclusions = set(excluded_features or [])
    feature_columns = [
        column for column in (feature_columns or dataframe.columns)
        if column in dataframe.columns and column != target_column
    ]
    invalid_exclusions = requested_exclusions.difference(feature_columns)
    if invalid_exclusions:
        return {"status": "not_possible", "reason": "One or more requested feature exclusions are not present in the dataset."}
    feature_columns = [column for column in feature_columns if column not in requested_exclusions]
    if not feature_columns:
        return {"status": "not_possible", "reason": "No features remain after the requested exclusions."}

    X = dataframe[feature_columns]
    y = dataframe[target_column]
    if y.isna().any():
        return {"status": "not_possible", "reason": "Rows with missing targets must be resolved before corrected evaluation."}
    stratify = y if "classification" in task_type.lower() else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=stratify,
        )
    except ValueError as exc:
        return {"status": "not_possible", "reason": f"A valid corrected split could not be created: {exc}"}

    model = clone(fitted_pipeline)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    result = MetricsCalculator(task_type=task_type).compute(
        model_name="corrected_evaluation",
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        n_features=len(feature_columns),
    )
    return {
        "status": "complete",
        "metrics": result.metrics,
        "test_size": len(X_test),
        "train_size": len(X_train),
        "random_seed": random_seed,
        "validation_strategy": "stratified random holdout" if stratify is not None else "random holdout",
        "excluded_features": sorted(requested_exclusions),
        "feature_columns": [str(column) for column in feature_columns],
    }
