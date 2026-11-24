"""
Classification Model Pool
=========================
Responsibility: Define and return the pool of candidate classification models.
"""

from typing import Any, Dict, List
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_classification_models(random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Return candidate classification models.
    """
    return [
        {
            "name": "logistic_regression",
            "estimator": LogisticRegression(
                random_state=random_state, max_iter=1000, solver="lbfgs"
            ),
            "tags": ["linear", "baseline"],
        },
        {
            "name": "ridge_classifier",
            "estimator": RidgeClassifier(random_state=random_state),
            "tags": ["linear", "regularized"],
        },
        {
            "name": "knn_classifier",
            "estimator": KNeighborsClassifier(n_neighbors=5),
            "tags": ["non-parametric", "distance-based"],
        },
        {
            "name": "decision_tree",
            "estimator": DecisionTreeClassifier(random_state=random_state),
            "tags": ["tree", "interpretable"],
        },
        {
            "name": "random_forest",
            "estimator": RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
            "tags": ["tree", "ensemble", "bagging"],
        },
        {
            "name": "gradient_boosting",
            "estimator": GradientBoostingClassifier(random_state=random_state),
            "tags": ["tree", "ensemble", "boosting"],
        },
        {
            "name": "svc",
            "estimator": SVC(
                kernel="rbf", probability=True, random_state=random_state
            ),
            "tags": ["svm", "kernel"],
        },
        {
            "name": "gaussian_nb",
            "estimator": GaussianNB(),
            "tags": ["probabilistic", "baseline"],
        },
        {
            "name": "xgboost_classifier",
            "estimator": XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
            ),
            "tags": ["tree", "ensemble", "boosting", "xgboost"],
        },
    ]
