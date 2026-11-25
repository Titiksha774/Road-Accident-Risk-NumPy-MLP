import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE if (HERE / "train.csv").exists() else HERE.parent
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
EPOCHS = 10
BATCH_SIZE = 512
LEARNING_RATE = 0.001
HIDDEN = [64, 32]
TARGET = "accident_risk"


# 1. PROBLEM FORMULATION
# Predict accident_risk (0 to 1) of a road from its conditions -> supervised regression.

def risk_level(score):
    if score < 0.25:
        return "LOW"
    if score < 0.50:
        return "MODERATE"
    if score < 0.75:
        return "HIGH"
    return "VERY HIGH"


# 2. DATASET AND PREPROCESSING

def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def split_data(df):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    n_val = int(0.15 * len(df))
    n_test = int(0.15 * len(df))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


class Preprocessor:
    numeric = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents"]
    boolean = ["road_signs_present", "public_road", "holiday", "school_season"]
    categorical = ["road_type", "lighting", "weather", "time_of_day"]
    discrete = ["speed_limit", "num_reported_accidents"]

    def fit(self, df):
        self.mean = df[self.numeric].mean()
        self.std = df[self.numeric].std()
        self.categories = {c: sorted(df[c].astype(str).unique()) for c in self.categorical}
        self.levels = {c: sorted(df[c].unique()) for c in self.discrete}
        return self

    def transform(self, df):
        parts = [((df[self.numeric] - self.mean) / self.std).to_numpy(float)]
        parts.append(df[self.boolean].astype(float).to_numpy())
        for c in self.categorical:
            values = df[c].astype(str).to_numpy()
            parts.append(np.stack([values == v for v in self.categories[c]], axis=1).astype(float))
        for c in self.discrete:
            values = df[c].to_numpy()
            parts.append(np.stack([values == v for v in self.levels[c]], axis=1).astype(float))
        return np.hstack(parts).astype(np.float32)


# 3. MODEL DESIGN: Multi-Layer Perceptron (Dense-ReLU-Dense-ReLU-Dense-Sigmoid) in NumPy

class MLP:
    def __init__(self, n_inputs, hidden, target_mean):
        rng = np.random.default_rng(SEED)
        sizes = [n_inputs] + hidden + [1]
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros((1, b), np.float32) for b in sizes[1:]]
        self.W[-1] *= 0.1
        self.b[-1][:] = np.log(target_mean / (1 - target_mean))
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0
        self.rng = rng

    def forward(self, x):
        acts = [x]
        for i in range(len(self.W) - 1):
            x = np.maximum(0, x @ self.W[i] + self.b[i])
            acts.append(x)
        out = 1 / (1 + np.exp(-(x @ self.W[-1] + self.b[-1])))
        return out, acts

    def predict(self, x):
        return self.forward(x)[0].ravel()

    def train_step(self, x, y, lr):
        out, acts = self.forward(x)
        grad = 2 * (out - y) / len(x) * out * (1 - out)
        for i in reversed(range(len(self.W))):
            gW = acts[i].T @ grad
            gb = grad.sum(axis=0, keepdims=True)
            if i > 0:
                grad = (grad @ self.W[i].T) * (acts[i] > 0)
            self.adam(i, gW, gb, lr)

    def adam(self, i, gW, gb, lr, b1=0.9, b2=0.999, eps=1e-8):
        if i == len(self.W) - 1:
            self.t += 1
        self.mW[i] = b1 * self.mW[i] + (1 - b1) * gW
        self.vW[i] = b2 * self.vW[i] + (1 - b2) * gW ** 2
        self.mb[i] = b1 * self.mb[i] + (1 - b1) * gb
        self.vb[i] = b2 * self.vb[i] + (1 - b2) * gb ** 2
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        self.W[i] -= lr * (self.mW[i] / c1) / (np.sqrt(self.vW[i] / c2) + eps)
        self.b[i] -= lr * (self.mb[i] / c1) / (np.sqrt(self.vb[i] / c2) + eps)


# 5. EVALUATION METRICS

def metrics(y, p):
    levels = lambda s: np.digitize(s, [0.25, 0.5, 0.75])
    return {
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAE": float(np.mean(np.abs(y - p))),
        "R2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
        "Risk-level accuracy": float(np.mean(levels(y) == levels(p))),
    }


# 4. TRAINING AND VALIDATION

def train(model, x_train, y_train, x_val, y_val):
    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            model.train_step(x_train[batch], y_train[batch].reshape(-1, 1), LEARNING_RATE)
        train_rmse = metrics(y_train, model.predict(x_train))["RMSE"]
        val = metrics(y_val, model.predict(x_val))
        history.append({"epoch": epoch, "train_rmse": train_rmse, "val_rmse": val["RMSE"], "val_r2": val["R2"]})
        print(f"Epoch {epoch:2d}/{EPOCHS}  train RMSE={train_rmse:.4f}  val RMSE={val['RMSE']:.4f}  val R2={val['R2']:.4f}")
    return history


# 6. ANALYSIS: feature importance and plots

def feature_importance(model, prep, df, y):
    rng = np.random.default_rng(SEED)
    base = metrics(y, model.predict(prep.transform(df)))["RMSE"]
    scores = {}
    for col in prep.numeric + prep.boolean + prep.categorical:
        shuffled = df.copy()
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        scores[col] = metrics(y, model.predict(prep.transform(shuffled)))["RMSE"] - base
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def save_plots(history, y_test, p_test, importance):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - plots skipped")
        return
    epochs = [h["epoch"] for h in history]

    plt.figure(figsize=(6, 3.5))
    plt.plot(epochs, [h["train_rmse"] for h in history], "o-", label="train")
    plt.plot(epochs, [h["val_rmse"] for h in history], "o-", label="validation")
    plt.xlabel("epoch"); plt.ylabel("RMSE"); plt.title("Training curve"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_DIR / "training_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(4.5, 4.2))
    plt.scatter(y_test[:10000], p_test[:10000], s=2, alpha=0.2)
    plt.plot([0, 1], [0, 1], "r")
    plt.xlabel("actual risk"); plt.ylabel("predicted risk"); plt.title("Predicted vs actual (test set)")
    plt.tight_layout(); plt.savefig(OUT_DIR / "predicted_vs_actual.png", dpi=150); plt.close()

    names = list(importance)[::-1]
    plt.figure(figsize=(6, 3.8))
    plt.barh(names, [importance[n] for n in names])
    plt.xlabel("increase in RMSE when shuffled"); plt.title("Feature importance")
    plt.tight_layout(); plt.savefig(OUT_DIR / "feature_importance.png", dpi=150); plt.close()


def main():
    print("Step 2: loading and preprocessing data")
    df, kaggle_test = load_data()
    train_df, val_df, test_df = split_data(df)
    print(f"  train={len(train_df)}  validation={len(val_df)}  test={len(test_df)}")
    prep = Preprocessor().fit(train_df)
    x_train, x_val, x_test = (prep.transform(d) for d in (train_df, val_df, test_df))
    y_train, y_val, y_test = (d[TARGET].to_numpy(np.float32) for d in (train_df, val_df, test_df))
    print(f"  input features: {x_train.shape[1]}")

    print("Step 3: building the model")
    model = MLP(x_train.shape[1], HIDDEN, float(y_train.mean()))
    print(f"  layers: {x_train.shape[1]} -> {' -> '.join(map(str, HIDDEN))} -> 1")

    print(f"Step 4: training for {EPOCHS} epochs")
    history = train(model, x_train, y_train, x_val, y_val)

    print("Step 5: evaluating on the test set")
    p_test = model.predict(x_test)
    test_metrics = metrics(y_test, p_test)
    baseline = metrics(y_test, np.full_like(y_test, y_train.mean()))
    for k in test_metrics:
        print(f"  {k}: model={test_metrics[k]:.4f}   baseline (always predict mean)={baseline[k]:.4f}")

    print("Step 6: analysis")
    importance = feature_importance(model, prep, test_df, y_test)
    for k, v in importance.items():
        print(f"  {k:24s} {v:.4f}")
    save_plots(history, y_test, p_test, importance)

    example = pd.DataFrame([{
        "road_type": "highway", "num_lanes": 2, "curvature": 0.82, "speed_limit": 70,
        "lighting": "night", "weather": "rainy", "road_signs_present": False, "public_road": True,
        "time_of_day": "evening", "holiday": False, "school_season": True, "num_reported_accidents": 3,
    }])
    score = float(model.predict(prep.transform(example))[0])
    print(f"Example road: risk={score:.3f} -> {risk_level(score)}")

    pd.DataFrame({"id": kaggle_test["id"], TARGET: model.predict(prep.transform(kaggle_test))}).to_csv(OUT_DIR / "submission.csv", index=False)
    (OUT_DIR / "results.json").write_text(json.dumps({
        "test_metrics": test_metrics, "baseline_metrics": baseline, "history": history,
        "feature_importance": importance, "example_risk": score,
        "sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df), "features": int(x_train.shape[1])},
    }, indent=1))
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
