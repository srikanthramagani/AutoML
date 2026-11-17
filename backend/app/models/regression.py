"""
Regression Model Pool
=====================
Responsibility: Define and return the pool of candidate regression models.
"""

from typing import Any, Dict, List
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def get_regression_models(random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Return candidate regression models.
    """
    return [
        {
            "name": "linear_regression",
            "estimator": LinearRegression(),
            "tags": ["linear", "baseline"],
        },
        {
            "name": "ridge",
            "estimator": Ridge(random_state=random_state),
            "tags": ["linear", "regularized"],
        },
        {
            "name": "lasso",
            "estimator": Lasso(random_state=random_state, max_iter=2000),
            "tags": ["linear", "sparse"],
        },
        {
            "name": "elastic_net",
            "estimator": ElasticNet(random_state=random_state, max_iter=2000),
            "tags": ["linear", "regularized"],
        },
        {
            "name": "knn_regressor",
            "estimator": KNeighborsRegressor(n_neighbors=5),
            "tags": ["non-parametric", "distance-based"],
        },
        {
            "name": "decision_tree_regressor",
            "estimator": DecisionTreeRegressor(random_state=random_state),
            "tags": ["tree", "interpretable"],
        },
        {
            "name": "random_forest_regressor",
            "estimator": RandomForestRegressor(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
            "tags": ["tree", "ensemble", "bagging"],
        },
        {
            "name": "gradient_boosting_regressor",
            "estimator": GradientBoostingRegressor(random_state=random_state),
            "tags": ["tree", "ensemble", "boosting"],
        },
        {
            "name": "svr",
            "estimator": SVR(kernel="rbf"),
            "tags": ["svm", "kernel"],
        },
        {
            "name": "xgboost_regressor",
            "estimator": XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
            ),
            "tags": ["tree", "ensemble", "boosting", "xgboost"],
        },
    ]
