import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from risk_model import TARGET, NumpyMLPRegressor, Preprocessor, predict_dataframe
from train_model import evaluate, split_indices

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

df = pd.read_csv("train.csv")
y = df[TARGET].to_numpy(np.float32)
tr, va, ho = split_indices(len(df), 0.15, 0.15, 42)
x_v1 = Preprocessor.fit(df.iloc[tr], discrete_cols=None, add_interactions=False).transform(df)
x_v2 = Preprocessor.fit(df.iloc[tr]).transform(df)

rows = []
if Path("road_accident_mlp_v1_original.pkl").exists():
    r = evaluate(y[ho], predict_dataframe(df.iloc[ho], "road_accident_mlp_v1_original.pkl"))
    rows.append({"name": "0. Saved v1 model (50k rows, 5 epochs)", **r})

cfg = dict(features="v1", output_init=False, dropout=0.1, batch_size=2048, lr=1e-3, schedule="constant", epochs=15, restore=False)
steps = [
    ("1. Original code, full training data, 15 epochs", {}),
    ("2. + feature engineering", dict(features="v2")),
    ("3. + output-layer initialisation", dict(output_init=True)),
    ("4. + remove dropout", dict(dropout=0.0)),
    ("5. + batch 512, warm-up/cosine LR (2e-3), 20 epochs, best epoch", dict(batch_size=512, lr=2e-3, schedule="cosine", epochs=20, restore=True)),
]
for name, update in steps:
    cfg.update(update)
    x = x_v1 if cfg["features"] == "v1" else x_v2
    t0 = time.time()
    model = NumpyMLPRegressor(x.shape[1], seed=42, output_init=float(y[tr].mean()) if cfg["output_init"] else None)
    history = model.fit(
        x[tr], y[tr], x[va], y[va],
        epochs=cfg["epochs"], batch_size=cfg["batch_size"], learning_rate=cfg["lr"], dropout=cfg["dropout"],
        lr_schedule=cfg["schedule"], restore_best=cfg["restore"], verbose=False,
    )
    r = evaluate(y[ho], model.predict(x[ho]))
    rows.append({"name": name, **r, "seconds": time.time() - t0, "val_curve": [h["val_rmse"] for h in history]})
    print(f"{name}: RMSE={r['rmse']:.5f} R2={r['r2']:.5f}", flush=True)

metrics_path = RESULTS / "metrics.json"
if metrics_path.exists():
    final = json.loads(metrics_path.read_text())["metrics"].get("hold-out test")
    if final:
        rows.append({"name": "6. + 5-network ensemble (final v2 model)", **final})

(RESULTS / "ablation.json").write_text(json.dumps(rows, indent=1))
print("Saved results/ablation.json")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib not installed - skipping ablation plot")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
labels = [r["name"].split(". ", 1)[1] for r in rows]
import textwrap

labels = [textwrap.fill(label, 18) for label in labels]
values = [r["rmse"] for r in rows]
fig, ax = plt.subplots(figsize=(8.5, 3.8))
ax.bar(range(len(rows)), values, color=["#8d99ae"] + ["#2a6f97"] * (len(rows) - 1))
for i, r in enumerate(rows):
    ax.text(i, r["rmse"] + 0.0004, f"{r['rmse']:.4f}\nR²={r['r2']:.3f}", ha="center", fontsize=7.5)
ax.set_xticks(range(len(rows)), labels, fontsize=7)
ax.set_ylim(0.05, max(values) + 0.004)
ax.set_ylabel("hold-out RMSE (lower is better)")
ax.set_title("Ablation: effect of each change")
fig.tight_layout()
fig.savefig(RESULTS / "ablation.png")
print("Saved results/ablation.png")
