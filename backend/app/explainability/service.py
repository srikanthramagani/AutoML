"""Explanations based only on held-out rows and the fitted inference pipeline."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


class ExplainabilityService:
    def __init__(self, random_state: int = 42, max_rows: int = 30) -> None:
        self.random_state = random_state
        self.max_rows = max_rows

    def explain(
        self,
        pipeline: Any,
        X_train: pd.DataFrame,
        X_holdout: pd.DataFrame,
        y_holdout: pd.Series,
        task_type: str,
        scoring: str,
    ) -> Dict[str, Any]:
        """Return permutation importance and a local feature ablation for real rows.

        SHAP and LIME are attempted when installed and supported. Optional method
        errors are reported as unavailable; the deterministic permutation and
        ablation summaries remain usable without either dependency.
        """
        if len(X_holdout) < 2:
            return {"status": "unavailable", "reason": "At least two holdout rows are required."}

        metric = {
            "rmse": "neg_root_mean_squared_error",
            "mse": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        }.get(scoring.lower(), scoring)
        sample_n = min(self.max_rows, len(X_holdout))
        X_eval = X_holdout.iloc[:sample_n]
        y_eval = y_holdout.iloc[:sample_n]
        try:
            perm = permutation_importance(
                pipeline,
                X_eval,
                y_eval,
                scoring=metric,
                n_repeats=3,
                random_state=self.random_state,
                n_jobs=1,
            )
            feature_importance = [
                {"feature": str(name), "importance_mean": float(mean), "importance_std": float(std)}
                for name, mean, std in zip(X_eval.columns, perm.importances_mean, perm.importances_std)
            ]
            feature_importance.sort(key=lambda item: abs(item["importance_mean"]), reverse=True)
        except Exception as exc:
            feature_importance = []
            permutation_error = str(exc)
        else:
            permutation_error = None

        local_explanations = self._ablation_explanations(pipeline, X_train, X_eval, task_type)
        shap_result = self._try_shap(pipeline, X_train, X_eval, task_type)
        lime_result = self._try_lime(pipeline, X_train, X_eval, task_type)
        return {
            "status": "complete" if feature_importance or local_explanations else "partial",
            "method": "holdout permutation importance and single-feature ablation",
            "holdout_rows_used": int(sample_n),
            "global_feature_importance": feature_importance,
            "permutation_error": permutation_error,
            "local_explanations": local_explanations,
            "shap": shap_result,
            "lime": lime_result,
        }

    @staticmethod
    def _prediction_signal(pipeline: Any, frame: pd.DataFrame, task_type: str) -> np.ndarray:
        if "classification" in task_type:
            if hasattr(pipeline, "predict_proba"):
                proba = np.asarray(pipeline.predict_proba(frame))
                return proba[:, -1] if proba.ndim == 2 else proba
            predictions = np.asarray(pipeline.predict(frame))
            classes = list(getattr(pipeline, "classes_", np.unique(predictions)))
            return np.asarray([classes.index(value) for value in predictions], dtype=float)
        return np.asarray(pipeline.predict(frame), dtype=float).reshape(-1)

    def _ablation_explanations(self, pipeline: Any, X_train: pd.DataFrame, X_eval: pd.DataFrame, task_type: str):
        output = []
        for row_index in range(min(5, len(X_eval))):
            row = X_eval.iloc[[row_index]].copy()
            original = float(self._prediction_signal(pipeline, row, task_type)[0])
            impacts = []
            for column in row.columns:
                train_col = X_train[column] if column in X_train else row[column]
                replacement = train_col.median() if pd.api.types.is_numeric_dtype(train_col) else train_col.mode(dropna=True).iloc[0] if not train_col.mode(dropna=True).empty else None
                altered = row.copy()
                if (
                    pd.api.types.is_integer_dtype(altered[column])
                    and isinstance(replacement, (int, float, np.integer, np.floating))
                    and not isinstance(replacement, (bool, np.bool_))
                    and not float(replacement).is_integer()
                ):
                    altered[column] = altered[column].astype(float)
                altered.loc[:, column] = replacement
                changed = float(self._prediction_signal(pipeline, altered, task_type)[0])
                impacts.append({"feature": str(column), "replacement": str(replacement), "prediction_change": changed - original})
            impacts.sort(key=lambda item: abs(item["prediction_change"]), reverse=True)
            output.append({"row_index": str(X_eval.index[row_index]), "prediction_signal": original, "features": impacts[:10]})
        return output

    def _try_shap(self, pipeline: Any, X_train: pd.DataFrame, X_eval: pd.DataFrame, task_type: str):
        try:
            import shap
            background = X_train.sample(min(20, len(X_train)), random_state=self.random_state)
            sample = X_eval.iloc[:min(30, len(X_eval))]
            predict_fn = lambda values: self._prediction_signal(pipeline, pd.DataFrame(values, columns=X_train.columns), task_type)
            explanation = shap.Explainer(predict_fn, background.to_numpy())(sample.to_numpy())
            values = np.asarray(explanation.values)
            if values.ndim > 2:
                values = values.mean(axis=tuple(range(1, values.ndim - 1)))
            mean_abs = np.mean(np.abs(values), axis=0).reshape(-1)
            if values.ndim != 2 or values.shape[1] != X_train.shape[1]:
                return {"status": "unavailable", "reason": "SHAP returned an unsupported output shape."}
            global_rows = [{"feature": str(name), "mean_abs_shap": float(value)} for name, value in zip(X_train.columns, mean_abs)]
            global_rows.sort(key=lambda item: item["mean_abs_shap"], reverse=True)
            dependence = []
            for feature_index, name in sorted(enumerate(X_train.columns), key=lambda item: mean_abs[item[0]], reverse=True)[:5]:
                dependence.append({"feature": str(name), "points": [{"value": float(sample.iloc[i, feature_index]) if pd.api.types.is_numeric_dtype(sample.iloc[:, feature_index]) else str(sample.iloc[i, feature_index]), "shap_value": float(values[i, feature_index])} for i in range(len(sample))]})
            local = [{"row_index": str(sample.index[i]), "contributions": [{"feature": str(X_train.columns[j]), "value": float(values[i, j])} for j in np.argsort(np.abs(values[i]))[::-1][:10]]} for i in range(min(5, len(sample)))]
            return {"status": "complete", "global_importance": global_rows, "dependence": dependence, "local": local}
        except ImportError:
            return {"status": "unavailable", "reason": "Install backend/requirements-explainability.txt to enable SHAP."}
        except Exception as exc:
            return {"status": "unavailable", "reason": str(exc)}

    def _try_lime(self, pipeline: Any, X_train: pd.DataFrame, X_eval: pd.DataFrame, task_type: str):
        if not all(pd.api.types.is_numeric_dtype(X_train[column]) for column in X_train.columns):
            return {"status": "unavailable", "reason": "LIME tabular explanations currently require numeric input features."}
        try:
            from lime.lime_tabular import LimeTabularExplainer
            classification = "classification" in task_type
            predict_fn = (lambda values: pipeline.predict_proba(pd.DataFrame(values, columns=X_train.columns))) if classification else (lambda values: pipeline.predict(pd.DataFrame(values, columns=X_train.columns)))
            explainer = LimeTabularExplainer(
                X_train.to_numpy(dtype=float),
                feature_names=list(map(str, X_train.columns)),
                mode="classification" if classification else "regression",
                discretize_continuous=True,
                random_state=self.random_state,
            )
            result = explainer.explain_instance(X_eval.iloc[0].to_numpy(dtype=float), predict_fn, num_features=min(10, X_train.shape[1]))
            return {"status": "complete", "row_index": str(X_eval.index[0]), "feature_weights": [{"feature_rule": str(rule), "weight": float(weight)} for rule, weight in result.as_list()]}
        except ImportError:
            return {"status": "unavailable", "reason": "Install backend/requirements-explainability.txt to enable LIME."}
        except Exception as exc:
            return {"status": "unavailable", "reason": str(exc)}
