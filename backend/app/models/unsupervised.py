"""
Unsupervised Model Pool
=======================
Responsibility: Define and return candidate unsupervised models (Clustering and Anomaly Detection).
"""

from typing import Any, Dict, List
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
    MiniBatchKMeans,
)
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def get_clustering_models(random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Return candidate clustering models.
    """
    return [
        {
            "name": "kmeans",
            "estimator": KMeans(
                n_clusters=3,
                init="k-means++",
                n_init=15,
                random_state=random_state,
            ),
            "tags": ["clustering", "centroid-based", "partitioning"],
        },
        {
            "name": "minibatch_kmeans",
            "estimator": MiniBatchKMeans(
                n_clusters=3,
                batch_size=256,
                n_init=3,
                random_state=random_state,
            ),
            "tags": ["clustering", "fast", "scalable"],
        },
        {
            "name": "gaussian_mixture",
            "estimator": GaussianMixture(
                n_components=3,
                covariance_type="full",
                random_state=random_state,
            ),
            "tags": ["clustering", "probabilistic", "density"],
        },
        {
            "name": "agglomerative",
            "estimator": AgglomerativeClustering(
                n_clusters=3,
                linkage="ward",
            ),
            "tags": ["clustering", "hierarchical"],
        },
        {
            "name": "dbscan",
            "estimator": DBSCAN(
                eps=0.5,
                min_samples=5,
                n_jobs=-1,
            ),
            "tags": ["clustering", "density-based", "outlier-tolerant"],
        },
    ]


def get_anomaly_detection_models(random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Return candidate anomaly detection models.
    """
    return [
        {
            "name": "isolation_forest",
            "estimator": IsolationForest(
                n_estimators=100,
                contamination="auto",
                random_state=random_state,
                n_jobs=-1,
            ),
            "tags": ["anomaly_detection", "tree", "ensemble"],
        },
        {
            "name": "one_class_svm",
            "estimator": OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=0.05,
            ),
            "tags": ["anomaly_detection", "svm", "boundary"],
        },
        {
            "name": "local_outlier_factor",
            "estimator": LocalOutlierFactor(
                n_neighbors=20,
                novelty=True,
                n_jobs=-1,
            ),
            "tags": ["anomaly_detection", "density", "neighbors"],
        },
    ]


def get_unsupervised_models(random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Return candidate unsupervised models (defaulting to clustering).
    """
    return get_clustering_models(random_state=random_state)
