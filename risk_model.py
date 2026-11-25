import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_string_dtype


TARGET = "accident_risk"
ID_COL = "id"

DEFAULT_DISCRETE_COLS = ("num_lanes", "speed_limit", "num_reported_accidents")


def risk_level(score: float) -> str:
    if score < 0.25:
        return "LOW"
    if score < 0.50:
        return "MODERATE"
    if score < 0.75:
        return "HIGH"
    return "VERY HIGH"


def risk_levels(scores: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(scores, dtype=float), [0.25, 0.50, 0.75])


@dataclass
class Preprocessor:
    numeric_cols: list[str]
    categorical_cols: list[str]
    categories: dict[str, list[str]]
    means: dict[str, float]
    stds: dict[str, float]
    feature_names: list[str]
    discrete_cols: list[str] = field(default_factory=list)
    discrete_values: dict[str, list[float]] = field(default_factory=dict)
    add_interactions: bool = False

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        discrete_cols=DEFAULT_DISCRETE_COLS,
        add_interactions: bool = True,
        max_discrete_values: int = 12,
    ) -> "Preprocessor":
        feature_df = df.drop(columns=[c for c in [TARGET] if c in df.columns])
        feature_df = feature_df.drop(columns=[c for c in [ID_COL] if c in feature_df.columns])

        categorical_cols = [
            col
            for col in feature_df.columns
            if is_string_dtype(feature_df[col]) or str(feature_df[col].dtype) == "category"
        ]
        numeric_cols = [
            col for col in feature_df.columns if col not in categorical_cols and (is_numeric_dtype(feature_df[col]) or is_bool_dtype(feature_df[col]))
        ]

        means = {}
        stds = {}
        for col in numeric_cols:
            values = feature_df[col].astype(float)
            means[col] = float(values.mean())
            std = float(values.std(ddof=0))
            stds[col] = std if std > 0 else 1.0

        categories = {
            col: sorted(feature_df[col].astype(str).fillna("missing").unique().tolist())
            for col in categorical_cols
        }

        discrete_cols = [c for c in (discrete_cols or []) if c in numeric_cols]
        discrete_values = {}
        for col in list(discrete_cols):
            values = sorted(float(v) for v in feature_df[col].dropna().unique())
            if len(values) > max_discrete_values:
                discrete_cols.remove(col)
                continue
            discrete_values[col] = values

        feature_names = numeric_cols[:]
        if add_interactions and "curvature" in numeric_cols:
            feature_names.append("curvature^2")
        for col in categorical_cols:
            feature_names.extend([f"{col}={value}" for value in categories[col]])
        for col in discrete_cols:
            feature_names.extend([f"{col}=={value:g}" for value in discrete_values[col]])

        return cls(
            numeric_cols,
            categorical_cols,
            categories,
            means,
            stds,
            feature_names,
            discrete_cols=discrete_cols,
            discrete_values=discrete_values,
            add_interactions=bool(add_interactions and "curvature" in numeric_cols),
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        n = len(df)

        if self.numeric_cols:
            numeric = np.empty((n, len(self.numeric_cols)), dtype=np.float32)
            for j, col in enumerate(self.numeric_cols):
                if col in df.columns:
                    values = df[col].astype(float).to_numpy()
                else:
                    values = np.full(n, self.means[col], dtype=float)
                numeric[:, j] = (values - self.means[col]) / self.stds[col]
            parts.append(numeric)

        if getattr(self, "add_interactions", False):
            if "curvature" in df.columns:
                curv = df["curvature"].astype(float).to_numpy()
            else:
                curv = np.full(n, self.means["curvature"], dtype=float)
            parts.append(((curv**2) - 0.33).reshape(-1, 1).astype(np.float32) * 3.0)

        for col in self.categorical_cols:
            values = df[col].astype(str).fillna("missing") if col in df.columns else pd.Series("", index=df.index)
            lookup = {value: idx for idx, value in enumerate(self.categories[col])}
            codes = values.map(lookup).to_numpy()
            encoded = np.zeros((n, len(self.categories[col])), dtype=np.float32)
            known = ~pd.isna(codes)
            encoded[np.flatnonzero(known), codes[known].astype(int)] = 1.0
            parts.append(encoded)

        for col in getattr(self, "discrete_cols", []):
            levels = np.asarray(self.discrete_values[col], dtype=float)
            if col in df.columns:
                values = df[col].astype(float).to_numpy()
            else:
                values = np.full(n, np.nan)
            parts.append((values[:, None] == levels[None, :]).astype(np.float32))

        return np.hstack(parts).astype(np.float32)


class NumpyMLPRegressor:

    def __init__(
        self,
        input_dim: int,
        hidden_layers=(128, 64, 32),
        seed: int = 42,
        output_activation: str = "sigmoid",
        output_init: float | None = None,
    ):
        self.hidden_layers = tuple(hidden_layers)
        self.output_activation = output_activation
        self.rng = np.random.default_rng(seed)
        layer_dims = (input_dim, *self.hidden_layers, 1)
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(layer_dims[:-1], layer_dims[1:]):
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(self.rng.normal(0, scale, size=(fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float32))
        if output_init is not None:
            self.weights[-1] *= 0.1
            if output_activation == "linear":
                self.biases[-1][:] = output_init
            else:
                p = float(np.clip(output_init, 1e-4, 1 - 1e-4))
                self.biases[-1][:] = np.log(p / (1 - p))
        self.history: list[dict[str, float]] = []
        self.best_epoch: int | None = None

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))

    def _output(self, z: np.ndarray) -> np.ndarray:
        if getattr(self, "output_activation", "sigmoid") == "linear":
            return z
        return self._sigmoid(z)

    def predict(self, x: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        outputs = []
        for start in range(0, len(x), batch_size):
            activation = x[start : start + batch_size]
            for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
                activation = self._relu(activation @ weight + bias)
            out = self._output(activation @ self.weights[-1] + self.biases[-1]).ravel()
            outputs.append(out)
        pred = np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32)
        if getattr(self, "output_activation", "sigmoid") == "linear":
            pred = np.clip(pred, 0.0, 1.0)
        return pred

    def _learning_rate(self, step: int, total_steps: int, base_lr: float, schedule: str, warmup_steps: int, min_lr_ratio: float) -> float:
        if schedule == "constant":
            return base_lr
        if step <= warmup_steps:
            return base_lr * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 15,
        batch_size: int = 2048,
        learning_rate: float = 0.001,
        dropout: float = 0.10,
        verbose: bool = True,
        weight_decay: float = 0.0,
        lr_schedule: str = "constant",
        warmup_epochs: float = 1.0,
        min_lr_ratio: float = 0.01,
        early_stopping_patience: int | None = None,
        restore_best: bool = False,
    ) -> list[dict[str, float]]:
        history = []
        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        linear_out = getattr(self, "output_activation", "sigmoid") == "linear"

        y_train = y_train.reshape(-1, 1).astype(np.float32)
        n_rows = x_train.shape[0]
        steps_per_epoch = int(np.ceil(n_rows / batch_size))
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(warmup_epochs * steps_per_epoch)

        best_val = np.inf
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(n_rows)
            for start in range(0, n_rows, batch_size):
                idx = order[start : start + batch_size]
                xb = x_train[idx]
                yb = y_train[idx]

                activations = [xb]
                pre_activations = []
                masks = []
                activation = xb

                for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
                    z = activation @ weight + bias
                    pre_activations.append(z)
                    activation = self._relu(z)
                    if dropout > 0:
                        mask = (self.rng.random(activation.shape, dtype=np.float32) >= dropout).astype(np.float32) / (1.0 - dropout)
                        activation = activation * mask
                    else:
                        mask = None
                    masks.append(mask)
                    activations.append(activation)

                z_out = activation @ self.weights[-1] + self.biases[-1]
                pred = self._output(z_out)
                activations.append(pred)

                if linear_out:
                    grad = (2.0 / len(xb)) * (pred - yb)
                else:
                    grad = (2.0 / len(xb)) * (pred - yb) * pred * (1.0 - pred)
                grad = grad.astype(np.float32)
                grad_w = [None] * len(self.weights)
                grad_b = [None] * len(self.biases)
                grad_w[-1] = activations[-2].T @ grad
                grad_b[-1] = grad.sum(axis=0, keepdims=True)

                for layer in range(len(self.weights) - 2, -1, -1):
                    grad = grad @ self.weights[layer + 1].T
                    if masks[layer] is not None:
                        grad = grad * masks[layer]
                    grad = grad * (pre_activations[layer] > 0)
                    grad_w[layer] = activations[layer].T @ grad
                    grad_b[layer] = grad.sum(axis=0, keepdims=True)

                step += 1
                lr = self._learning_rate(step, total_steps, learning_rate, lr_schedule, warmup_steps, min_lr_ratio)
                bc1 = 1 - beta1**step
                bc2 = 1 - beta2**step
                for i in range(len(self.weights)):
                    m_w[i] = beta1 * m_w[i] + (1 - beta1) * grad_w[i]
                    v_w[i] = beta2 * v_w[i] + (1 - beta2) * (grad_w[i] ** 2)
                    m_b[i] = beta1 * m_b[i] + (1 - beta1) * grad_b[i]
                    v_b[i] = beta2 * v_b[i] + (1 - beta2) * (grad_b[i] ** 2)

                    if weight_decay > 0:
                        self.weights[i] -= lr * weight_decay * self.weights[i]
                    self.weights[i] -= lr * (m_w[i] / bc1) / (np.sqrt(v_w[i] / bc2) + eps)
                    self.biases[i] -= lr * (m_b[i] / bc1) / (np.sqrt(v_b[i] / bc2) + eps)

            train_pred = self.predict(x_train[: min(50000, len(x_train))])
            val_pred = self.predict(x_val)
            history.append(
                {
                    "epoch": epoch,
                    "lr": float(lr),
                    "train_rmse": rmse(y_train[: len(train_pred)].ravel(), train_pred),
                    "val_rmse": rmse(y_val, val_pred),
                    "val_mae": mae(y_val, val_pred),
                    "val_r2": r2_score(y_val, val_pred),
                }
            )
            if verbose:
                row = history[-1]
                print(
                    f"Epoch {epoch}/{epochs} - "
                    f"lr={row['lr']:.2e} "
                    f"train_rmse={row['train_rmse']:.5f} "
                    f"val_rmse={row['val_rmse']:.5f} "
                    f"val_mae={row['val_mae']:.5f} "
                    f"val_r2={row['val_r2']:.5f}",
                    flush=True,
                )

            if history[-1]["val_rmse"] < best_val - 1e-6:
                best_val = history[-1]["val_rmse"]
                self.best_epoch = epoch
                epochs_without_improvement = 0
                if restore_best:
                    best_state = ([w.copy() for w in self.weights], [b.copy() for b in self.biases])
            else:
                epochs_without_improvement += 1
                if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (best epoch {self.best_epoch}).", flush=True)
                    break

        if restore_best and best_state is not None:
            self.weights, self.biases = best_state
        self.history = history
        return history


class EnsembleRegressor:
    def __init__(self, models: list[NumpyMLPRegressor]):
        self.models = models

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = np.vstack([model.predict(x) for model in self.models])
        return predictions.mean(axis=0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


def risk_level_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(risk_levels(y_true) == risk_levels(y_pred)))


def save_artifact(path: str | Path, preprocessor: Preprocessor, model: NumpyMLPRegressor) -> None:
    with Path(path).open("wb") as f:
        pickle.dump({"preprocessor": preprocessor, "model": model}, f)


def load_artifact(path: str | Path) -> tuple[Preprocessor, NumpyMLPRegressor]:
    with Path(path).open("rb") as f:
        artifact = pickle.load(f)
    return artifact["preprocessor"], artifact["model"]


def predict_dataframe(df: pd.DataFrame, model_path: str | Path = "road_accident_mlp.pkl") -> np.ndarray:
    preprocessor, model = load_artifact(model_path)
    return model.predict(preprocessor.transform(df))
