"""
Time Series Forecasting Model Pool
==================================
Responsibility: Define and return candidate models tuned for temporal/time-series forecasting.
"""

from typing import Any, Dict, List
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:
    HistGradientBoostingRegressor = GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from xgboost import XGBRegressor


def get_time_series_models(random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Return candidate models specialized for temporal sequence forecasting.
    """
    try:
        gradient_estimator = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            random_state=random_state,
        )
    except (NameError, TypeError):
        gradient_estimator = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            random_state=random_state,
        )
    return [
        {
            "name": "time_series_gradient_boosting",
            "estimator": gradient_estimator,
            "tags": ["time_series", "boosting", "non-linear", "seasonality"],
        },
        {
            "name": "time_series_xgboost",
            "estimator": XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
            ),
            "tags": ["time_series", "xgboost", "boosting", "trend"],
        },
        {
            "name": "time_series_random_forest",
            "estimator": RandomForestRegressor(
                n_estimators=120,
                max_depth=8,
                random_state=random_state,
                n_jobs=-1,
            ),
            "tags": ["time_series", "tree", "ensemble"],
        },
        {
            "name": "time_series_ridge",
            "estimator": Ridge(
                alpha=10.0,
                random_state=random_state,
            ),
            "tags": ["time_series", "linear", "regularized"],
        },
        {
            "name": "time_series_elastic_net",
            "estimator": ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=random_state,
                max_iter=2000,
            ),
            "tags": ["time_series", "linear", "sparse"],
        },
    ]
