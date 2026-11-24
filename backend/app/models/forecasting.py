"""Leakage-aware direct multi-step forecasting for one numeric time series."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import optuna
except Exception:
    optuna = None


def _series_from_frame(frame: pd.DataFrame, target: str, datetime_column: Optional[str], frequency: Optional[str]) -> pd.Series:
    if target not in frame:
        raise ValueError(f"Target column {target!r} was not found.")
    values = pd.to_numeric(frame[target], errors="coerce")
    if datetime_column:
        if datetime_column not in frame:
            raise ValueError(f"Datetime column {datetime_column!r} was not found.")
        dates = pd.to_datetime(frame[datetime_column], errors="coerce")
        valid = dates.notna() & values.notna()
        series = pd.Series(values[valid].to_numpy(dtype=float), index=pd.DatetimeIndex(dates[valid]), name=target)
        series = series.groupby(level=0).mean().sort_index()
    else:
        series = pd.Series(values.dropna().to_numpy(dtype=float), name=target)
    if frequency:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("A frequency requires a datetime column.")
        series = series.asfreq(frequency)
    elif isinstance(series.index, pd.DatetimeIndex) and len(series) >= 3:
        inferred = pd.infer_freq(series.index)
        if inferred:
            series = series.asfreq(inferred)
    if isinstance(series.index, pd.DatetimeIndex):
        first_observed = series.first_valid_index()
        return series.loc[first_observed:] if first_observed is not None else series.iloc[0:0]
    return series.dropna()


def _metric_values(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    denominator = np.abs(actual) + np.abs(predicted)
    nonzero = actual != 0
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if np.any(nonzero) else None,
        "smape": float(np.mean(np.divide(2 * np.abs(actual - predicted), denominator, out=np.zeros_like(denominator), where=denominator != 0)) * 100),
    }


class DirectTimeSeriesForecaster:
    """Direct horizon-specific forecaster with Ridge and optional recurrent neural models."""

    def __init__(
        self,
        horizon: int = 7,
        lags: Tuple[int, ...] = (1, 2, 3, 7, 14),
        rolling_windows: Tuple[int, ...] = (3, 7),
        n_trials: int = 15,
        cv_folds: int = 4,
        random_state: int = 42,
        model_family: str = "ridge",
    ) -> None:
        if horizon < 1 or n_trials < 1 or cv_folds < 2:
            raise ValueError("horizon and n_trials must be positive; cv_folds must be at least 2.")
        if not lags or min(lags) < 1 or not rolling_windows or min(rolling_windows) < 1:
            raise ValueError("Lags and rolling windows must contain positive integers.")
        if model_family not in ("ridge", "lstm", "bilstm", "bilstm_stack", "auto"):
            raise ValueError("model_family must be ridge, lstm, bilstm, bilstm_stack, or auto.")
        self.horizon = horizon
        self.lags = tuple(sorted(set(lags)))
        self.rolling_windows = tuple(sorted(set(rolling_windows)))
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model_family = model_family
        self.models_: Dict[int, Any] = {}
        self.series_: Optional[pd.Series] = None
        self.target_: Optional[str] = None
        self.datetime_column_: Optional[str] = None
        self.frequency_: Optional[str] = None
        self.best_alphas_: Dict[int, float] = {}
        self.fitted_horizon_: int = 0

    def _sequence_windows(self, values: np.ndarray, origins: np.ndarray, horizon: int, window: int):
        windows = []
        targets = []
        valid_origins = []
        for origin in origins:
            origin = int(origin)
            target_index = origin + horizon
            if origin - window + 1 < 0 or target_index >= len(values):
                continue
            sequence = values[origin - window + 1:origin + 1]
            target = values[target_index]
            if np.isfinite(sequence).all() and np.isfinite(target):
                windows.append(sequence)
                targets.append(target)
                valid_origins.append(origin)
        if not windows:
            return np.empty((0, window, 1)), np.empty((0,), dtype=float), np.asarray(valid_origins)
        return (
            np.asarray(windows, dtype=float).reshape(-1, window, 1),
            np.asarray(targets, dtype=float),
            np.asarray(valid_origins),
        )

    def _fit_neural_model(self, values: np.ndarray, origins: np.ndarray, horizon: int, family: str):
        window = max(7, max(self.lags))
        X, y, valid_origins = self._sequence_windows(values, origins, horizon, window)
        if len(y) < 10:
            raise ValueError("Not enough observations to train a recurrent neural forecast.")

        try:
            import tensorflow as tf
            keras = getattr(tf, "keras", None)
            if keras is None:
                from tensorflow import keras
        except Exception as exc:
            raise RuntimeError("TensorFlow is unavailable; select Ridge fallback.") from exc

        input_scaler = StandardScaler()
        X_scaled = input_scaler.fit_transform(X.reshape(-1, window)).reshape(-1, window, 1)
        target_scaler = StandardScaler()
        y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)

        np.random.seed(self.random_state)
        try:
            tf.random.set_seed(self.random_state)
        except Exception:
            pass

        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(window, 1)))
        if family == "lstm":
            model.add(keras.layers.LSTM(32, return_sequences=False))
        elif family == "bilstm":
            model.add(keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=False)))
        else:
            model.add(keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True)))
            model.add(keras.layers.Bidirectional(keras.layers.LSTM(16, return_sequences=False)))
        model.add(keras.layers.Dense(16, activation="relu"))
        model.add(keras.layers.Dense(1))
        try:
            optimizer = keras.optimizers.Adam(learning_rate=0.003)
        except TypeError:
            optimizer = keras.optimizers.Adam(lr=0.003)
        model.compile(optimizer=optimizer, loss="mse")

        fit_kwargs = {
            "epochs": 40,
            "batch_size": 32,
            "shuffle": False,
            "verbose": 0,
        }
        if hasattr(keras, "callbacks"):
            try:
                fit_kwargs["callbacks"] = [
                    keras.callbacks.EarlyStopping(
                        monitor="loss",
                        patience=5,
                        restore_best_weights=True,
                    )
                ]
            except TypeError:
                fit_kwargs["callbacks"] = [
                    keras.callbacks.EarlyStopping(monitor="loss", patience=5)
                ]
        model.fit(X_scaled, y_scaled, **fit_kwargs)
        return {
            "keras_model": model,
            "input_scaler": input_scaler,
            "target_scaler": target_scaler,
            "window": window,
            "family": family,
        }

    def _predict_neural(self, trained: Dict[str, Any], values: np.ndarray, origin: int) -> float:
        window = int(trained["window"])
        if origin - window + 1 < 0:
            raise ValueError("Not enough history for the recurrent neural forecast.")
        sequence = np.asarray(values[origin - window + 1:origin + 1], dtype=float).reshape(1, window, 1)
        sequence_scaled = trained["input_scaler"].transform(sequence.reshape(1, window)).reshape(1, window, 1)
        try:
            prediction_scaled = trained["keras_model"].predict(sequence_scaled, verbose=0).reshape(-1)[0]
        except TypeError:
            prediction_scaled = trained["keras_model"].predict(sequence_scaled).reshape(-1)[0]
        prediction = trained["target_scaler"].inverse_transform([[prediction_scaled]])[0, 0]
        return float(prediction)

    def fit_evaluate(self, frame: pd.DataFrame, target: str, datetime_column: Optional[str] = None, frequency: Optional[str] = None, test_size: float = 0.2):
        series = _series_from_frame(frame, target, datetime_column, frequency)
        values = series.to_numpy(dtype=float)
        feature_values = series.ffill().to_numpy(dtype=float)
        min_history = max(max(self.lags), max(self.rolling_windows))
        split = int(len(values) * (1.0 - test_size))
        effective_horizon = min(self.horizon, len(values) - split)
        if split <= min_history + effective_horizon + self.cv_folds or effective_horizon < 1:
            raise ValueError("Not enough ordered observations for the selected history, horizon, CV folds, and holdout.")

        date_index = series.index if isinstance(series.index, pd.DatetimeIndex) else None
        self.frequency_ = frequency or (pd.infer_freq(date_index) if date_index is not None else None)
        models, cv_scores, histories, forecast_values, selected_alphas = {}, {}, {}, [], {}
        resolved_model_family = {}
        for horizon_step in range(1, effective_horizon + 1):
            origins = np.arange(min_history - 1, split - horizon_step)
            all_X = np.vstack([self._features(feature_values, int(origin), horizon_step, date_index) for origin in origins])
            all_y = values[origins + horizon_step]
            observed = np.isfinite(all_y)
            X, y = all_X[observed], all_y[observed]
            if len(y) <= self.cv_folds + 2:
                raise ValueError(f"Too few observed training targets for horizon {horizon_step}.")
            fold_count = min(self.cv_folds, max(2, len(origins) // 3))
            try:
                splits = list(TimeSeriesSplit(n_splits=fold_count, gap=max(0, horizon_step - 1)).split(X))
            except TypeError:
                splits = list(TimeSeriesSplit(n_splits=fold_count).split(X))
            if len(splits) < 2:
                raise ValueError(f"Insufficient time-series folds for horizon {horizon_step}.")

            if optuna is not None:
                study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.random_state + horizon_step))
                def objective(trial):
                    alpha = trial.suggest_float("alpha", 1e-5, 1e4, log=True)
                    fold_errors = []
                    for step, (train_idx, valid_idx) in enumerate(splits):
                        model = Ridge(alpha=alpha).fit(X[train_idx], y[train_idx])
                        error = float(np.sqrt(mean_squared_error(y[valid_idx], model.predict(X[valid_idx]))))
                        fold_errors.append(error)
                        trial.report(float(np.mean(fold_errors)), step=step)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    return float(np.mean(fold_errors))
                study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False, catch=(Exception,))
                if not study.best_trials:
                    raise RuntimeError(f"All Optuna trials failed for horizon {horizon_step}.")
                alpha = float(study.best_params["alpha"])
                cv_scores[str(horizon_step)] = [float(trial.value) for trial in study.trials if trial.value is not None]
                histories[str(horizon_step)] = [{"number": t.number, "value": t.value, "state": t.state.name, "params": t.params} for t in study.trials]
            else:
                alpha = 10.0
                cv_scores[str(horizon_step)] = []
                histories[str(horizon_step)] = [{
                    "number": 0,
                    "value": None,
                    "state": "FALLBACK",
                    "params": {"alpha": alpha},
                }]
            ridge_model = Ridge(alpha=alpha).fit(X, y)
            selected_alphas[str(horizon_step)] = alpha
            if self.model_family in ("lstm", "bilstm", "bilstm_stack"):
                try:
                    neural = self._fit_neural_model(values, origins[origins < split - horizon_step], horizon_step, self.model_family)
                    models[horizon_step] = neural
                    forecast_values.append(self._predict_neural(neural, feature_values, split - 1))
                    resolved_model_family[str(horizon_step)] = self.model_family
                except (RuntimeError, ValueError):
                    models[horizon_step] = ridge_model
                    forecast_values.append(float(ridge_model.predict(self._features(feature_values, split - 1, horizon_step, date_index).reshape(1, -1))[0]))
                    resolved_model_family[str(horizon_step)] = "ridge_fallback"
            else:
                models[horizon_step] = ridge_model
                forecast_values.append(float(ridge_model.predict(self._features(feature_values, split - 1, horizon_step, date_index).reshape(1, -1))[0]))
                resolved_model_family[str(horizon_step)] = "ridge"

        if date_index is not None:
            inferred = frequency or pd.infer_freq(date_index)
            offset = pd.tseries.frequencies.to_offset(inferred) if inferred else (date_index[-1] - date_index[-2])
            forecast_dates = [date_index[split - 1] + offset * step for step in range(1, effective_horizon + 1)]
        else:
            forecast_dates = list(range(split, split + effective_horizon))
        actual = values[split:split + effective_horizon]
        naive = np.repeat(values[split - 1], effective_horizon)
        rows = [{"index": str(forecast_dates[i]), "actual": float(actual[i]) if np.isfinite(actual[i]) else None, "predicted": forecast_values[i], "naive": float(naive[i]), "residual": float(actual[i] - forecast_values[i]) if np.isfinite(actual[i]) else None} for i in range(effective_horizon)]
        observed_holdout = np.isfinite(actual)
        if not np.any(observed_holdout):
            raise ValueError("No observed target values exist in the forecast holdout horizon.")
        metrics = {f"model_{key}": value for key, value in _metric_values(actual[observed_holdout], np.asarray(forecast_values)[observed_holdout]).items()}
        metrics.update({f"naive_{key}": value for key, value in _metric_values(actual[observed_holdout], naive[observed_holdout]).items()})
        # Refit the selected horizon models on all observed history only after
        # the holdout forecast and metrics have been finalized.
        all_feature_values = series.ffill().to_numpy(dtype=float)
        for horizon_step, model in models.items():
            origins = np.arange(min_history - 1, len(values) - horizon_step)
            all_X = np.vstack([self._features(all_feature_values, int(origin), horizon_step, date_index) for origin in origins])
            all_y = values[origins + horizon_step]
            observed = np.isfinite(all_y)
            if self.model_family in ("lstm", "bilstm", "bilstm_stack"):
                try:
                    models[horizon_step] = self._fit_neural_model(values, origins, horizon_step, self.model_family)
                except (RuntimeError, ValueError):
                    models[horizon_step] = Ridge(alpha=selected_alphas[str(horizon_step)]).fit(all_X[observed], all_y[observed])
            else:
                models[horizon_step] = Ridge(alpha=selected_alphas[str(horizon_step)]).fit(all_X[observed], all_y[observed])
        self.models_ = models
        self.fitted_horizon_ = len(models)
        self.series_ = series
        self.target_ = target
        self.datetime_column_ = datetime_column
        self.best_alphas_ = {int(key): value for key, value in selected_alphas.items()}
        return {
            "config": {
                "target": target,
                "datetime_column": datetime_column,
                "frequency": self.frequency_,
                "horizon": effective_horizon,
                "requested_horizon": self.horizon,
                "history_lags": self.lags,
                "rolling_windows": self.rolling_windows,
                "random_seed": self.random_state,
                "test_size": test_size,
                "cv_folds": self.cv_folds,
                "best_alpha_by_horizon": selected_alphas,
                "model_family": self.model_family,
                "resolved_model_family_by_horizon": resolved_model_family,
                "forecast_mode": "fixed-origin direct multi-step",
            },
            "metrics": metrics,
            "predictions": rows,
            "cv_scores": cv_scores,
            "optimization_history": histories,
        }

    def _features(self, values: np.ndarray, origin: int, horizon: int, dates: Optional[pd.DatetimeIndex]) -> np.ndarray:
        row = [float(values[origin - lag + 1]) for lag in self.lags]
        for window in self.rolling_windows:
            history = values[origin - window + 1:origin + 1]
            row.extend([float(np.mean(history)), float(np.std(history))])
        if dates is not None:
            target_pos = origin + horizon
            if target_pos < len(dates):
                future_date = dates[target_pos]
            else:
                inferred = self.frequency_ or pd.infer_freq(dates)
                offset = pd.tseries.frequencies.to_offset(inferred) if inferred else (dates[-1] - dates[-2])
                future_date = dates[-1] + offset * (target_pos - len(dates) + 1)
            row.extend([float(future_date.month), float(future_date.dayofweek), float(future_date.dayofyear)])
        return np.asarray(row, dtype=float)

    def predict_future(self, history_frame: Optional[pd.DataFrame] = None, horizon: Optional[int] = None) -> List[Dict[str, Any]]:
        """Forecast new future rows from stored history or appended observed history."""
        if not self.models_ or self.series_ is None or self.target_ is None:
            raise RuntimeError("Forecaster is not fitted. Call fit_evaluate before prediction.")
        steps = self.fitted_horizon_ if horizon is None else int(horizon)
        if steps < 1 or steps > self.fitted_horizon_:
            raise ValueError(f"horizon must be between 1 and the fitted horizon ({self.fitted_horizon_}).")
        series = self.series_.copy()
        if history_frame is not None:
            incoming = _series_from_frame(history_frame, self.target_, self.datetime_column_, self.frequency_)
            combined = pd.concat([series, incoming])
            if isinstance(combined.index, pd.DatetimeIndex):
                combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
            else:
                combined = pd.Series(np.concatenate([series.to_numpy(dtype=float), incoming.to_numpy(dtype=float)]), name=self.target_)
            series = combined
        values = series.to_numpy(dtype=float)
        features = series.ffill().to_numpy(dtype=float)
        min_history = max(max(self.lags), max(self.rolling_windows))
        origin = len(values) - 1
        if origin < min_history - 1:
            raise ValueError("Not enough observed history for the fitted lag features.")
        dates = series.index if isinstance(series.index, pd.DatetimeIndex) else None
        output = []
        for step in range(1, steps + 1):
            trained_model = self.models_[step]
            if isinstance(trained_model, dict) and "keras_model" in trained_model:
                prediction = self._predict_neural(trained_model, features, origin)
            else:
                prediction = float(trained_model.predict(self._features(features, origin, step, dates).reshape(1, -1))[0])
            if dates is not None:
                inferred = self.frequency_ or pd.infer_freq(dates)
                offset = pd.tseries.frequencies.to_offset(inferred) if inferred else (dates[-1] - dates[-2])
                date = dates[-1] + offset * step
                index = str(date)
            else:
                index = str(len(values) + step - 1)
            output.append({"horizon": step, "index": index, "predicted": prediction})
        return output
