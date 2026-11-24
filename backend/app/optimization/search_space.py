"""
SearchSpace
===========
Responsibility: Define hyperparameter search spaces for each candidate model.
"""

from typing import Any, Callable, Dict

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    HAS_OPTUNA = False


class SearchSpace:
    """
    Optuna hyperparameter distributions per model name.
    """

    def __init__(self) -> None:
        self._spaces: Dict[str, Callable[[Any], Dict[str, Any]]] = {}
        if HAS_OPTUNA:
            self._register_defaults()

    def register(self, model_name: str, space_fn: Callable[[Any], Dict[str, Any]]) -> None:
        self._spaces[model_name] = space_fn

    def sample(self, model_name: str, trial: Any) -> Dict[str, Any]:
        if not HAS_OPTUNA or model_name not in self._spaces:
            return {}
        return self._spaces[model_name](trial)

    def _register_defaults(self) -> None:
        def _logistic_regression(trial) -> Dict[str, Any]:
            return {
                "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                "max_iter": trial.suggest_int("max_iter", 200, 1000, step=200),
            }

        def _ridge_classifier(trial) -> Dict[str, Any]:
            return {
                "alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)
            }

        def _knn(trial) -> Dict[str, Any]:
            return {
                "n_neighbors": trial.suggest_int("n_neighbors", 3, 15),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            }

        def _decision_tree(trial) -> Dict[str, Any]:
            return {
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            }

        def _random_forest(trial) -> Dict[str, Any]:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=25),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            }

        def _gradient_boosting(trial) -> Dict[str, Any]:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            }

        def _svc(trial) -> Dict[str, Any]:
            kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
            params = {
                "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
                "kernel": kernel,
            }
            if kernel == "rbf":
                params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
            return params

        def _ridge(trial) -> Dict[str, Any]:
            return {
                "alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)
            }

        def _lasso(trial) -> Dict[str, Any]:
            return {
                "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True)
            }

        def _elastic_net(trial) -> Dict[str, Any]:
            return {
                "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
            }

        def _svr(trial) -> Dict[str, Any]:
            return {
                "C": trial.suggest_float("C", 1e-2, 5e2, log=True),
                "epsilon": trial.suggest_float("epsilon", 0.005, 0.5, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            }

        def _xgboost(trial) -> Dict[str, Any]:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }

        def _kmeans(trial) -> Dict[str, Any]:
            return {
                "n_clusters": trial.suggest_int("n_clusters", 2, 8),
                "init": trial.suggest_categorical("init", ["k-means++", "random"]),
            }

        def _minibatch_kmeans(trial) -> Dict[str, Any]:
            return {
                "n_clusters": trial.suggest_int("n_clusters", 2, 8),
                "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            }

        def _dbscan(trial) -> Dict[str, Any]:
            return {
                "eps": trial.suggest_float("eps", 0.2, 1.5, step=0.1),
                "min_samples": trial.suggest_int("min_samples", 3, 10),
            }

        def _agglomerative(trial) -> Dict[str, Any]:
            return {
                "n_clusters": trial.suggest_int("n_clusters", 2, 8),
                "linkage": trial.suggest_categorical("linkage", ["ward", "complete", "average"]),
            }

        def _gaussian_mixture(trial) -> Dict[str, Any]:
            return {
                "n_components": trial.suggest_int("n_components", 2, 8),
                "covariance_type": trial.suggest_categorical("covariance_type", ["full", "tied", "diag"]),
            }

        def _isolation_forest(trial) -> Dict[str, Any]:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 250, step=25),
                "max_features": trial.suggest_float("max_features", 0.6, 1.0),
                "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
            }

        def _one_class_svm(trial) -> Dict[str, Any]:
            return {
                "nu": trial.suggest_float("nu", 0.01, 0.15),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
            }

        def _local_outlier_factor(trial) -> Dict[str, Any]:
            return {
                "n_neighbors": trial.suggest_int("n_neighbors", 10, 30, step=5),
            }

        def _time_series_gb(trial) -> Dict[str, Any]:
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor
                HistGradientBoostingRegressor(max_iter=100, learning_rate=0.08)
                return {
                    "max_iter": trial.suggest_int("max_iter", 100, 250, step=50),
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                }
            except (ImportError, TypeError):
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 250, step=50),
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                }

        def _time_series_xgb(trial) -> Dict[str, Any]:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 80, 240, step=40),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            }

        def _time_series_rf(trial) -> Dict[str, Any]:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 80, 200, step=40),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
            }

        def _time_series_ridge(trial) -> Dict[str, Any]:
            return {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
            }

        def _time_series_enet(trial) -> Dict[str, Any]:
            return {
                "alpha": trial.suggest_float("alpha", 0.001, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
            }

        self._spaces.update({
            "logistic_regression": _logistic_regression,
            "ridge_classifier": _ridge_classifier,
            "knn_classifier": _knn,
            "knn_regressor": _knn,
            "decision_tree": _decision_tree,
            "decision_tree_regressor": _decision_tree,
            "random_forest": _random_forest,
            "random_forest_regressor": _random_forest,
            "gradient_boosting": _gradient_boosting,
            "gradient_boosting_regressor": _gradient_boosting,
            "svc": _svc,
            "svr": _svr,
            "ridge": _ridge,
            "lasso": _lasso,
            "elastic_net": _elastic_net,
            "xgboost_classifier": _xgboost,
            "xgboost_regressor": _xgboost,
            "kmeans": _kmeans,
            "minibatch_kmeans": _minibatch_kmeans,
            "dbscan": _dbscan,
            "agglomerative": _agglomerative,
            "gaussian_mixture": _gaussian_mixture,
            "isolation_forest": _isolation_forest,
            "one_class_svm": _one_class_svm,
            "local_outlier_factor": _local_outlier_factor,
            "time_series_gradient_boosting": _time_series_gb,
            "time_series_xgboost": _time_series_xgb,
            "time_series_random_forest": _time_series_rf,
            "time_series_ridge": _time_series_ridge,
            "time_series_elastic_net": _time_series_enet,
        })
