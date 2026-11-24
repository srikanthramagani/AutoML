"""
Unsupervised Model Pool
=======================
Responsibility: Define and return candidate unsupervised models (Clustering and Anomaly Detection).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
    MiniBatchKMeans,
    Birch,
)
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def _make_kmeans(n_clusters: int) -> KMeans:
    try:
        from sklearn import __version__ as sklearn_version
        version_parts = tuple(int(part) for part in sklearn_version.split(".")[:2])
        if version_parts >= (1, 4):
            return KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    except (TypeError, ValueError):
        pass
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=42)


class _NearestCorePredictor:
    """Inductive DBSCAN adapter that maps unseen rows to the nearest core sample."""

    def __init__(self, eps: float, min_samples: int = 5) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan_ = DBSCAN(eps=eps, min_samples=min_samples)
        self.core_neighbors_: Optional[NearestNeighbors] = None
        self.core_labels_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any = None) -> "_NearestCorePredictor":
        X_array = np.asarray(X)
        labels = self.dbscan_.fit_predict(X_array)
        core_indices = getattr(self.dbscan_, "core_sample_indices_", np.array([], dtype=int))
        valid_core_indices = np.asarray(core_indices, dtype=int)
        valid_core_indices = valid_core_indices[
            (valid_core_indices >= 0) & (valid_core_indices < len(X_array))
        ]
        if len(valid_core_indices) == 0:
            # Nearest-centroid fallback from all training rows preserves predictability.
            valid_core_indices = np.arange(len(X_array))
        core_labels = labels[valid_core_indices]
        self.core_neighbors_ = NearestNeighbors(n_neighbors=1).fit(X_array[valid_core_indices])
        self.core_labels_ = np.asarray(core_labels)
        self.labels_ = labels
        return self

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        return self.labels_

    def predict(self, X: Any) -> np.ndarray:
        X_array = np.asarray(X)
        if self.core_neighbors_ is None or self.core_labels_ is None:
            raise RuntimeError("Nearest-core DBSCAN predictor is not fitted.")
        indices = self.core_neighbors_.kneighbors(X_array, return_distance=False).reshape(-1)
        return self.core_labels_[indices]


class _InductiveAgglomerative:
    """Agglomerative clustering adapter with out-of-sample nearest-centroid prediction."""

    def __init__(self, n_clusters: int) -> None:
        self.n_clusters = n_clusters
        self.agglomerative_ = AgglomerativeClustering(n_clusters=n_clusters)
        self.centroid_classifier_: Optional[NearestCentroid] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any = None) -> "_InductiveAgglomerative":
        X_array = np.asarray(X)
        labels = np.asarray(self.agglomerative_.fit_predict(X_array), dtype=int)
        classifier = NearestCentroid()
        classifier.fit(X_array, labels)
        self.centroid_classifier_ = classifier
        self.labels_ = labels
        return self

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        return np.asarray(self.labels_, dtype=int)

    def predict(self, X: Any) -> np.ndarray:
        if self.centroid_classifier_ is None:
            raise RuntimeError("Inductive Agglomerative predictor is not fitted.")
        return np.asarray(self.centroid_classifier_.predict(np.asarray(X)), dtype=int)


def get_unsupervised_identifier_columns(df: pd.DataFrame) -> List[str]:
    """Return high-cardinality string columns that may safely label cluster rows."""
    identifiers: List[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
            non_null = series.dropna()
            unique_ratio = float(non_null.nunique()) / max(len(non_null), 1)
            if unique_ratio > 0.85:
                identifiers.append(str(column))
    return identifiers


def build_unsupervised_preprocessor(df: pd.DataFrame) -> Pipeline:
    """Build an isolated, inductive preprocessing pipeline for clustering."""
    identifier_columns = get_unsupervised_identifier_columns(df)
    features = df.drop(columns=[column for column in identifier_columns if column in df.columns]).copy()
    numeric_columns = [
        column for column in features.columns
        if pd.api.types.is_numeric_dtype(features[column].dtype)
    ]
    categorical_columns = [
        column for column in features.columns
        if (pd.api.types.is_object_dtype(features[column].dtype) or pd.api.types.is_string_dtype(features[column].dtype))
        and int(features[column].nunique(dropna=True)) <= 10
    ]

    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable clustering features were found after identifier detection.")

    transformers: List[Tuple[str, Pipeline, List[str]]] = []
    if numeric_columns:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_columns,
        ))
    if categorical_columns:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_columns,
        ))

    estimated_width = len(numeric_columns)
    for column in categorical_columns:
        estimated_width += int(features[column].nunique(dropna=True))
    steps: List[Tuple[str, Any]] = [
        ("features", ColumnTransformer(transformers=transformers, remainder="drop")),
    ]
    if estimated_width > 8:
        steps.append(("pca", PCA(n_components=0.95, random_state=42)))
    return Pipeline(steps=steps)


def _safe_cluster_metrics(X: np.ndarray, labels: np.ndarray) -> Tuple[int, float, float, float, float]:
    labels = np.asarray(labels)
    non_noise = labels != -1
    noise_ratio = float(np.mean(~non_noise)) if len(labels) else 1.0
    valid_labels = labels[non_noise]
    valid_X = X[non_noise]
    unique_labels = np.unique(valid_labels)
    n_clusters = int(len(unique_labels))
    if n_clusters < 2 or len(unique_labels) >= len(valid_labels):
        return n_clusters, -1.0, 0.0, 999.0, noise_ratio
    counts = np.bincount(valid_labels.astype(int)) if np.issubdtype(valid_labels.dtype, np.integer) else None
    if counts is not None and len(counts) and float(np.max(counts) / len(valid_labels)) > 0.90:
        return n_clusters, -1.0, 0.0, 999.0, noise_ratio
    try:
        silhouette = float(silhouette_score(valid_X, valid_labels, random_state=42))
        calinski = float(calinski_harabasz_score(valid_X, valid_labels))
        davies = float(davies_bouldin_score(valid_X, valid_labels))
        return n_clusters, silhouette, calinski, davies, noise_ratio
    except Exception:
        return n_clusters, -1.0, 0.0, 999.0, noise_ratio


def _valid_cluster_candidate(labels: np.ndarray) -> bool:
    labels = np.asarray(labels)
    non_noise = labels != -1
    unique_labels = np.unique(labels[non_noise])
    n_clusters = len(unique_labels)
    noise_ratio = float(np.mean(~non_noise)) if len(labels) else 1.0
    if n_clusters < 2 or noise_ratio > 0.30:
        return False
    counts = np.unique(labels[non_noise], return_counts=True)[1]
    if len(counts) and float(np.max(counts) / len(labels[non_noise])) > 0.90:
        return False
    return True


def run_inductive_clustering_suite(
    full_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_holdout: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """Run inductive clustering candidates and produce the clustering graph payload."""
    identifier_columns = get_unsupervised_identifier_columns(full_df)
    feature_df = full_df.drop(
        columns=[column for column in identifier_columns if column in full_df.columns]
    ).copy()
    train_df = X_train.drop(
        columns=[column for column in identifier_columns if column in X_train.columns]
    ).copy()
    holdout_df = (
        X_holdout.drop(columns=[column for column in identifier_columns if column in X_holdout.columns]).copy()
        if X_holdout is not None
        else None
    )

    # Preserve support for numeric-only and mixed-tabular datasets.
    numeric_columns = [column for column in train_df.columns if pd.api.types.is_numeric_dtype(train_df[column].dtype)]
    categorical_columns = [
        column for column in train_df.columns
        if (pd.api.types.is_object_dtype(train_df[column].dtype) or pd.api.types.is_string_dtype(train_df[column].dtype))
        and int(train_df[column].nunique(dropna=True)) <= 10
    ]
    feature_df = feature_df[numeric_columns + categorical_columns].copy()
    train_df = train_df[numeric_columns + categorical_columns].copy()
    if holdout_df is not None:
        holdout_df = holdout_df[numeric_columns + categorical_columns].copy()
    if train_df.empty or len(train_df.columns) == 0:
        raise ValueError("No usable clustering features were found.")

    preprocessor = build_unsupervised_preprocessor(train_df)
    try:
        X_train_processed = np.asarray(preprocessor.fit_transform(train_df), dtype=float)
        X_full_processed = np.asarray(preprocessor.transform(feature_df), dtype=float)
    except Exception:
        X_train_processed = np.asarray(build_unsupervised_preprocessor(train_df).fit_transform(train_df), dtype=float)
        X_full_processed = X_train_processed[:len(feature_df)]
    if not np.all(np.isfinite(X_train_processed)):
        fallback_preprocessor = Pipeline([
            step for step in preprocessor.steps
            if step[0] != "pca"
        ])
        X_train_processed = np.asarray(fallback_preprocessor.fit_transform(train_df), dtype=float)
        X_full_processed = np.asarray(fallback_preprocessor.transform(feature_df), dtype=float)
        preprocessor = fallback_preprocessor
    X_eval_processed = (
        np.asarray(preprocessor.transform(holdout_df), dtype=float)
        if holdout_df is not None
        else X_full_processed
    )
    eval_df = holdout_df if holdout_df is not None else feature_df
    eval_is_holdout = holdout_df is not None

    k_values = list(range(2, min(9, max(3, len(X_train_processed) // 5))))
    if not k_values:
        k_values = [2]
    k_values = [k for k in k_values if k < len(X_train_processed)]
    if not k_values:
        k_values = [2] if len(X_train_processed) >= 3 else []

    candidate_records: List[Dict[str, Any]] = []
    fitted_candidates: List[Tuple[str, int, Any, np.ndarray]] = []
    for k in k_values:
        candidates = [
            ("KMeans", _make_kmeans(k)),
            ("GaussianMixture", GaussianMixture(n_components=k, covariance_type="full", random_state=42)),
            ("Birch", Birch(n_clusters=k)),
            ("Agglomerative", _InductiveAgglomerative(n_clusters=k)),
        ]
        for model_name, estimator in candidates:
            try:
                if model_name == "Agglomerative":
                    estimator.fit(X_train_processed)
                    train_labels = np.asarray(estimator.labels_, dtype=int)
                    predictor = estimator
                elif model_name == "GaussianMixture":
                    estimator.fit(X_train_processed)
                    predictor = estimator
                else:
                    estimator.fit(X_train_processed)
                    predictor = estimator
                    train_labels = (
                        np.asarray(estimator.labels_, dtype=int)
                        if hasattr(estimator, "labels_")
                        else np.asarray(predictor.predict(X_train_processed), dtype=int)
                    )
                eval_labels = np.asarray(predictor.predict(X_eval_processed), dtype=int)
                if not _valid_cluster_candidate(eval_labels):
                    continue
                n_clusters, silhouette, calinski, davies, noise_ratio = _safe_cluster_metrics(
                    X_eval_processed, eval_labels
                )
                if n_clusters < 2 or silhouette == -1.0:
                    continue
                record = {
                    "model_name": model_name,
                    "k": int(k),
                    "silhouette_score": round(silhouette, 4),
                    "calinski_harabasz_score": round(calinski, 4),
                    "davies_bouldin_score": round(davies, 4),
                    "noise_ratio": round(noise_ratio, 4),
                }
                candidate_records.append(record)
                fitted_candidates.append((model_name, k, predictor, eval_labels))
            except Exception:
                continue

        try:
            neighbor_count = min(5, len(X_train_processed))
            nearest = NearestNeighbors(n_neighbors=neighbor_count).fit(X_train_processed)
            distances = nearest.kneighbors(X_train_processed, return_distance=True)[0]
            kth_distances = distances[:, -1]
            eps = float(np.percentile(kth_distances, 90))
            if not np.isfinite(eps) or eps <= 0:
                eps = float(np.percentile(kth_distances, 50)) if len(kth_distances) else 0.5
            dbscan = _NearestCorePredictor(eps=max(eps, 1e-6), min_samples=min(5, len(X_train_processed)))
            dbscan.fit(X_train_processed)
            db_labels = np.asarray(dbscan.predict(X_eval_processed), dtype=int)
            if _valid_cluster_candidate(db_labels):
                n_clusters, silhouette, calinski, davies, noise_ratio = _safe_cluster_metrics(
                    X_eval_processed, db_labels
                )
                if n_clusters >= 2 and silhouette != -1.0:
                    record = {
                        "model_name": "DBSCAN",
                        "k": int(n_clusters),
                        "silhouette_score": round(silhouette, 4),
                        "calinski_harabasz_score": round(calinski, 4),
                        "davies_bouldin_score": round(davies, 4),
                        "noise_ratio": round(noise_ratio, 4),
                    }
                    candidate_records.append(record)
                    fitted_candidates.append(("DBSCAN", n_clusters, dbscan, db_labels))
        except Exception:
            pass

    if candidate_records:
        silhouettes = [record["silhouette_score"] for record in candidate_records]
        calinski = [record["calinski_harabasz_score"] for record in candidate_records]
        davies = [record["davies_bouldin_score"] for record in candidate_records]

        def normalize(values: List[float]) -> np.ndarray:
            arr = np.asarray(values, dtype=float)
            span = float(np.max(arr) - np.min(arr))
            return np.ones_like(arr) if span <= 1e-12 else (arr - np.min(arr)) / span

        norm_silhouette = normalize(silhouettes)
        norm_calinski = normalize(calinski)
        norm_davies = normalize(davies)
        for index, record in enumerate(candidate_records):
            record["composite_score"] = round(
                0.45 * norm_silhouette[index]
                + 0.25 * norm_calinski[index]
                + 0.20 * (1.0 - norm_davies[index])
                + 0.10 * (1.0 - record["noise_ratio"]),
                4,
            )
            record["is_best"] = False
        best_index = int(np.argmax([record["composite_score"] for record in candidate_records]))
        candidate_records[best_index]["is_best"] = True
        best_model_name, best_k, best_predictor, best_eval_labels = fitted_candidates[best_index]

        if hasattr(best_predictor, "predict"):
            full_labels = np.asarray(best_predictor.predict(X_full_processed), dtype=int)
        elif hasattr(best_predictor, "fit_predict"):
            full_labels = np.asarray(best_predictor.fit_predict(X_full_processed), dtype=int)
        else:
            full_labels = np.asarray(best_eval_labels[:len(feature_df)], dtype=int)
    else:
        candidate_records = []
        best_model_name = "KMeans"
        best_k = k_values[0] if k_values else 2
        try:
            best_predictor = _make_kmeans(best_k).fit(X_train_processed)
            full_labels = np.asarray(best_predictor.predict(X_full_processed), dtype=int)
            best_eval_labels = np.asarray(best_predictor.predict(X_eval_processed), dtype=int)
        except Exception:
            best_predictor = _make_kmeans(2).fit(X_full_processed)
            full_labels = np.asarray(best_predictor.predict(X_full_processed), dtype=int)
            best_eval_labels = full_labels
        sentinel_clusters, _, _, _, _ = _safe_cluster_metrics(X_eval_processed, best_eval_labels)

    k_selection_curve: List[Dict[str, Any]] = []
    for k in range(2, 9):
        if k >= len(X_train_processed):
            break
        try:
            kmeans = _make_kmeans(k).fit(X_train_processed)
            k_labels = np.asarray(kmeans.predict(X_train_processed), dtype=int)
            n_clusters, silhouette, _, davies, _ = _safe_cluster_metrics(X_train_processed, k_labels)
            k_selection_curve.append({
                "k": int(k),
                "inertia": round(float(kmeans.inertia_), 4),
                "silhouette_score": round(silhouette, 4) if n_clusters >= 2 else -1.0,
                "davies_bouldin_score": round(davies, 4) if n_clusters >= 2 else 999.0,
            })
        except Exception:
            continue

    pca_scatter: List[Dict[str, Any]] = []
    try:
        pca = PCA(n_components=2, random_state=42)
        coordinates = pca.fit_transform(X_full_processed)
        holdout_start = len(feature_df) - len(eval_df) if eval_is_holdout else len(feature_df)
        for index in range(len(feature_df)):
            label_value = (
                str(feature_df.iloc[index].name)
                if len(identifier_columns) == 0
                else None
            )
            if identifier_columns:
                source_column = identifier_columns[0]
                label_value = full_df.iloc[index][source_column]
            split_value = (
                "holdout" if eval_is_holdout and index >= holdout_start
                else ("train" if eval_is_holdout else "full")
            )
            pca_scatter.append({
                "label": "" if label_value is None else str(label_value),
                "x": round(float(coordinates[index, 0]), 6),
                "y": round(float(coordinates[index, 1]), 6),
                "cluster": int(full_labels[index]),
                "split": split_value,
            })
    except Exception:
        pca_scatter = []

    cluster_summary: List[Dict[str, Any]] = []
    try:
        summary_numeric = feature_df[numeric_columns].copy()
        summary_numeric["_cluster"] = full_labels
        for cluster_id in sorted(np.unique(full_labels)):
            members = summary_numeric[summary_numeric["_cluster"] == cluster_id].drop(columns=["_cluster"])
            count = int(len(members))
            feature_means = {
                str(column): round(float(members[column].mean()), 6)
                for column in members.columns
                if np.isfinite(members[column].mean())
            }
            cluster_summary.append({
                "cluster_id": int(cluster_id),
                "count": count,
                "percentage": round(100.0 * count / max(len(full_labels), 1), 4),
                "feature_means": feature_means,
            })
    except Exception:
        cluster_summary = []

    if candidate_records:
        winner_record = candidate_records[best_index]
        holdout_metrics = {
            "n_clusters": float(winner_record["k"]),
            "silhouette_score": float(winner_record["silhouette_score"]),
            "calinski_harabasz_score": float(winner_record["calinski_harabasz_score"]),
            "davies_bouldin_score": float(winner_record["davies_bouldin_score"]),
            "noise_ratio": float(winner_record["noise_ratio"]),
        }
    else:
        holdout_metrics = {
            "n_clusters": 0.0,
            "silhouette_score": -1.0,
            "calinski_harabasz_score": 0.0,
            "davies_bouldin_score": 999.0,
            "noise_ratio": 1.0,
        }

    final_preprocessor = build_unsupervised_preprocessor(train_df)
    final_pipeline = Pipeline([
        ("preprocessor", final_preprocessor),
        ("estimator", best_predictor),
    ])
    final_pipeline.fit(train_df)

    comparison_graphs = {
        "model_comparison": candidate_records,
        "k_selection_curve": k_selection_curve,
        "pca_scatter_2d": pca_scatter,
        "cluster_summary": cluster_summary,
    }
    return {
        "pipeline": final_pipeline,
        "best_model": best_model_name,
        "best_params": {"n_clusters": int(best_k)},
        "metrics": holdout_metrics,
        "comparison_graphs": comparison_graphs,
        "full_labels": full_labels,
    }


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
            "estimator": _InductiveAgglomerative(n_clusters=3),
            "tags": ["clustering", "hierarchical"],
        },
        {
            "name": "birch",
            "estimator": Birch(n_clusters=3),
            "tags": ["clustering", "hierarchical", "scalable"],
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
