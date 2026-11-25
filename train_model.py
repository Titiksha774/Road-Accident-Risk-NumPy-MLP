import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from risk_model import (
    TARGET,
    EnsembleRegressor,
    NumpyMLPRegressor,
    Preprocessor,
    mae,
    predict_dataframe,
    r2_score,
    risk_level,
    risk_level_accuracy,
    risk_levels,
    rmse,
    save_artifact,
)

LEVEL_NAMES = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]

PRESETS = {
    "baseline": {
        "epochs": 15,
        "batch_size": 2048,
        "learning_rate": 0.001,
        "ensemble_size": 3,
        "hidden": "128,64,32",
        "dropout": 0.10,
        "weight_decay": 0.0,
        "lr_schedule": "constant",
        "feature_engineering": False,
        "output_init": False,
        "restore_best": False,
    },
    "improved": {
        "epochs": 20,
        "batch_size": 512,
        "learning_rate": 0.002,
        "ensemble_size": 5,
        "hidden": "128,64,32",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "lr_schedule": "cosine",
        "feature_engineering": True,
        "output_init": True,
        "restore_best": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy MLP for road accident risk prediction.")
    parser.add_argument("--train", default="train.csv", help="Training CSV path.")
    parser.add_argument("--test", default="test.csv", help="Test CSV path (Kaggle test, no labels).")
    parser.add_argument("--model", default="road_accident_mlp.pkl", help="Output model artifact path.")
    parser.add_argument("--submission", default="submission.csv", help="Output Kaggle-style submission CSV.")
    parser.add_argument("--report", default="training_report.md", help="Output Markdown report path.")
    parser.add_argument("--results-dir", default="results", help="Folder for plots and metrics.json.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="improved", help="Hyper-parameter preset.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Peak Adam learning rate.")
    parser.add_argument("--ensemble-size", type=int, default=None, help="Number of MLP models to train and average.")
    parser.add_argument("--hidden", default=None, help="Hidden layer sizes, e.g. 128,64,32.")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate on hidden layers.")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW decoupled weight decay.")
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default=None, help="Learning-rate schedule.")
    parser.add_argument("--validation-size", type=float, default=0.15, help="Fraction used for validation (early stopping / model selection).")
    parser.add_argument("--holdout-size", type=float, default=0.15, help="Fraction held out as an untouched labelled test set. Use 0 to disable.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick experiments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--compare-baseline", action="store_true", help="Also train the original baseline on the same split and report both.")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving figures.")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    args.hidden = tuple(int(h) for h in str(args.hidden).split(",") if h.strip())
    return args


def split_indices(n_rows: int, validation_size: float, holdout_size: float, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    hold_count = int(n_rows * holdout_size)
    val_count = int(n_rows * validation_size)
    hold_idx = order[:hold_count]
    val_idx = order[hold_count : hold_count + val_count]
    train_idx = order[hold_count + val_count :]
    return train_idx, val_idx, hold_idx


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "risk_level_accuracy": risk_level_accuracy(y_true, y_pred),
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    t, p = risk_levels(y_true), risk_levels(y_pred)
    cm = np.zeros((4, 4), dtype=int)
    np.add.at(cm, (t, p), 1)
    return cm


def train_ensemble(cfg, x_train, y_train, x_val, y_val, seed: int, label: str):
    models, histories = [], []
    for model_idx in range(max(1, cfg.ensemble_size)):
        model_seed = seed + model_idx * 101
        print(f"[{label}] Training model {model_idx + 1}/{cfg.ensemble_size} with seed {model_seed}...", flush=True)
        model = NumpyMLPRegressor(
            input_dim=x_train.shape[1],
            hidden_layers=cfg.hidden,
            seed=model_seed,
            output_init=float(y_train.mean()) if cfg.output_init else None,
        )
        history = model.fit(
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            dropout=cfg.dropout,
            weight_decay=cfg.weight_decay,
            lr_schedule=cfg.lr_schedule,
            restore_best=cfg.restore_best,
        )
        models.append(model)
        histories.append(history)
    model = models[0] if len(models) == 1 else EnsembleRegressor(models)
    return model, histories


def permutation_importance(model, preprocessor, df_eval: pd.DataFrame, y_eval: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    base = rmse(y_eval, model.predict(preprocessor.transform(df_eval)))
    cols = preprocessor.numeric_cols + preprocessor.categorical_cols
    scores = {}
    for col in cols:
        shuffled = df_eval.copy()
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        scores[col] = rmse(y_eval, model.predict(preprocessor.transform(shuffled))) - base
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def save_plots(results_dir: Path, histories, baseline_histories, y_hold, pred_hold, cm, importance, comparison) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots (pip install matplotlib).")
        return []

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    blue, orange, grey = "#2a6f97", "#e07a5f", "#8d99ae"
    saved = []

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, h in enumerate(histories):
        ax.plot([r["epoch"] for r in h], [r["val_rmse"] for r in h], color=blue, alpha=0.9 if i == 0 else 0.35, lw=1.6 if i == 0 else 1,
                label="Improved - validation" if i == 0 else None)
    h0 = histories[0]
    ax.plot([r["epoch"] for r in h0], [r["train_rmse"] for r in h0], color=blue, ls="--", lw=1, label="Improved - train (sample)")
    for i, h in enumerate(baseline_histories or []):
        ax.plot([r["epoch"] for r in h], [r["val_rmse"] for r in h], color=orange, alpha=0.9 if i == 0 else 0.35, lw=1.6 if i == 0 else 1,
                label="Baseline - validation" if i == 0 else None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE")
    ax.set_ylim(0.055, 0.075)
    ax.set_title("Learning curves (lower is better)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(results_dir / "learning_curves.png")
    plt.close(fig)
    saved.append("learning_curves.png")

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    sample = np.random.default_rng(0).choice(len(y_hold), size=min(15000, len(y_hold)), replace=False)
    ax.scatter(y_hold[sample], pred_hold[sample], s=3, alpha=0.15, color=blue, rasterized=True)
    ax.plot([0, 1], [0, 1], color=orange, lw=1.2)
    ax.set_xlabel("actual accident_risk")
    ax.set_ylabel("predicted accident_risk")
    ax.set_title("Predicted vs actual (hold-out test)")
    fig.tight_layout()
    fig.savefig(results_dir / "pred_vs_actual.png")
    plt.close(fig)
    saved.append("pred_vs_actual.png")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(pred_hold - y_hold, bins=80, color=blue)
    ax.axvline(0, color=orange, lw=1)
    ax.set_xlabel("prediction error (predicted - actual)")
    ax.set_ylabel("rows")
    ax.set_title("Residual distribution (hold-out test)")
    fig.tight_layout()
    fig.savefig(results_dir / "residuals.png")
    plt.close(fig)
    saved.append("residuals.png")

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(4), LEVEL_NAMES, rotation=20)
    ax.set_yticks(range(4), LEVEL_NAMES)
    ax.set_xlabel("predicted level")
    ax.set_ylabel("actual level")
    ax.set_title("Risk-level confusion matrix")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png")
    plt.close(fig)
    saved.append("confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(6, 3.8))
    names = list(importance)[::-1]
    ax.barh(names, [importance[n] for n in names], color=blue)
    ax.set_xlabel("increase in RMSE when feature is shuffled")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()
    fig.savefig(results_dir / "feature_importance.png")
    plt.close(fig)
    saved.append("feature_importance.png")

    if comparison:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
        import textwrap

        labels = list(comparison)
        tick_labels = [textwrap.fill(label, 16) for label in labels]
        colors = [orange, blue][: len(labels)] if len(labels) == 2 else [grey, orange, blue][: len(labels)]
        axes[0].bar(tick_labels, [comparison[k]["r2"] for k in labels], color=colors)
        axes[0].set_ylim(0.7, 0.9)
        axes[0].set_title("R² (higher is better)")
        axes[1].bar(tick_labels, [comparison[k]["rmse"] for k in labels], color=colors)
        axes[1].set_ylim(0.05, 0.085)
        axes[1].set_title("RMSE (lower is better)")
        for ax in axes:
            ax.tick_params(axis="x", labelsize=8)
        fig.tight_layout()
        fig.savefig(results_dir / "baseline_vs_improved.png")
        plt.close(fig)
        saved.append("baseline_vs_improved.png")
    return saved


def write_report(path: Path, args, preprocessor, histories, metrics, comparison, cm, importance, plots, split_sizes, train_seconds) -> None:
    arch = " -> ".join(["input features"] + [f"Dense({h}, ReLU)" for h in args.hidden] + ["Dense(1, sigmoid)"])
    lines = [
        "# Road Accident Risk Prediction Report",
        "",
        f"Preset: `{args.preset}`",
        "",
        "## Model",
        "",
        "- Algorithm: Multi-Layer Perceptron / Deep Neural Network implemented with NumPy",
        f"- Architecture: {arch}",
        f"- Ensemble: average of {args.ensemble_size} networks trained with different seeds",
        f"- Optimiser: Adam{'W' if args.weight_decay else ''}, peak learning rate {args.learning_rate}, "
        f"schedule `{args.lr_schedule}`, batch size {args.batch_size}, up to {args.epochs} epochs",
        f"- Dropout: {args.dropout}; output-layer initialisation at target mean: {args.output_init}; "
        f"keep best-validation epoch: {args.restore_best}",
        "- Output: `accident_risk` score between 0 and 1",
        "- Risk levels: LOW, MODERATE, HIGH, VERY HIGH",
        f"- Training time: {train_seconds:.0f} s",
        "",
        "## Data Split",
        "",
        f"- Train rows: {split_sizes['train']:,}",
        f"- Validation rows (early stopping / model selection): {split_sizes['val']:,}",
        f"- Hold-out test rows (never used for training or tuning): {split_sizes['holdout']:,}",
        "",
        "## Metrics",
        "",
        "| Split | RMSE | MAE | R2 | Risk-level accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, m in metrics.items():
        lines.append(f"| {split} | {m['rmse']:.5f} | {m['mae']:.5f} | {m['r2']:.5f} | {m['risk_level_accuracy']:.4f} |")

    if comparison:
        lines += ["", "## Baseline vs Improved (hold-out test)", "", "| Model | RMSE | MAE | R2 | Risk-level accuracy |", "|---|---:|---:|---:|---:|"]
        for name, m in comparison.items():
            lines.append(f"| {name} | {m['rmse']:.5f} | {m['mae']:.5f} | {m['r2']:.5f} | {m['risk_level_accuracy']:.4f} |")

    if cm is not None:
        lines += ["", "## Risk-level Confusion Matrix (hold-out test; rows = actual, columns = predicted)", "",
                  "| | " + " | ".join(LEVEL_NAMES) + " |", "|---|" + "---:|" * 4]
        for i, name in enumerate(LEVEL_NAMES):
            lines.append(f"| {name} | " + " | ".join(str(v) for v in cm[i]) + " |")

    lines += ["", "## Training History (validation, per epoch)", ""]
    for m_idx, history in enumerate(histories, start=1):
        lines += [f"### Model {m_idx}", "", "| Epoch | LR | Train RMSE | Validation RMSE | Validation MAE | Validation R2 |", "|---:|---:|---:|---:|---:|---:|"]
        for row in history:
            lines.append(
                f"| {row['epoch']} | {row.get('lr', args.learning_rate):.2e} | {row['train_rmse']:.5f} | {row['val_rmse']:.5f} | {row['val_mae']:.5f} | {row['val_r2']:.5f} |"
            )
        lines.append("")

    if importance:
        lines += ["## Permutation Feature Importance (hold-out test)", "", "| Feature | RMSE increase |", "|---|---:|"]
        for name, val in importance.items():
            lines.append(f"| {name} | {val:.5f} |")
        lines.append("")

    lines += ["## Features Used", "", ", ".join(preprocessor.feature_names), ""]
    if plots:
        lines += ["## Figures", ""] + [f"- `{args.results_dir}/{p}`" for p in plots] + [""]
    lines += [
        "## Interpretation",
        "",
        "The model estimates risk from the combined road, weather, lighting, timing, and accident-history conditions. "
        "Higher scores indicate situations that look more similar to historically higher-risk examples in the training data. "
        "Curvature, lighting (night), speed limit (60+), weather and the number of previously reported accidents carry almost all of the signal.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else Path.cwd() / path


def main() -> None:
    args = parse_args()
    model_path, submission_path, report_path, results_dir = map(resolve, (args.model, args.submission, args.report, args.results_dir))
    for output_path in (model_path, submission_path, report_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train)
    if args.max_rows:
        df = df.sample(n=min(args.max_rows, len(df)), random_state=args.seed).reset_index(drop=True)
    y = df[TARGET].to_numpy(dtype=np.float32)
    train_idx, val_idx, hold_idx = split_indices(len(df), args.validation_size, args.holdout_size, args.seed)
    split_sizes = {"train": len(train_idx), "val": len(val_idx), "holdout": len(hold_idx)}
    print(f"Split: {split_sizes}", flush=True)

    preprocessor = Preprocessor.fit(df.iloc[train_idx], discrete_cols=None, add_interactions=False) if not args.feature_engineering \
        else Preprocessor.fit(df.iloc[train_idx])
    x = preprocessor.transform(df)

    start = time.time()
    model, histories = train_ensemble(args, x[train_idx], y[train_idx], x[val_idx], y[val_idx], args.seed, args.preset)
    train_seconds = time.time() - start

    metrics = {"validation": evaluate(y[val_idx], model.predict(x[val_idx]))}
    cm, importance, pred_hold, comparison, baseline_histories = None, {}, None, {}, None
    if len(hold_idx):
        pred_hold = model.predict(x[hold_idx])
        metrics["hold-out test"] = evaluate(y[hold_idx], pred_hold)
        cm = confusion_matrix(y[hold_idx], pred_hold)
        print("Computing permutation importance...", flush=True)
        importance = permutation_importance(model, preprocessor, df.iloc[hold_idx], y[hold_idx], args.seed)

        if args.compare_baseline:
            base_cfg = argparse.Namespace(**{**vars(args), **PRESETS["baseline"]})
            base_cfg.hidden = tuple(int(h) for h in base_cfg.hidden.split(","))
            base_pre = Preprocessor.fit(df.iloc[train_idx], discrete_cols=None, add_interactions=False)
            xb = base_pre.transform(df)
            base_model, baseline_histories = train_ensemble(base_cfg, xb[train_idx], y[train_idx], xb[val_idx], y[val_idx], args.seed, "baseline")
            comparison["Original v1 settings, full data"] = evaluate(y[hold_idx], base_model.predict(xb[hold_idx]))
            comparison["Improved v2" if args.preset == "improved" else f"Retrained ({args.preset})"] = metrics["hold-out test"]
            old_pkl = resolve("road_accident_mlp_v1_original.pkl")
            if old_pkl.exists():
                comparison = {"Saved v1 model (50k-row quick run)": evaluate(y[hold_idx], predict_dataframe(df.iloc[hold_idx], old_pkl)), **comparison}

    save_artifact(model_path, preprocessor, model)

    test_df = pd.read_csv(args.test)
    test_pred = predict_dataframe(test_df, model_path)
    submission = pd.DataFrame({"id": test_df["id"], TARGET: np.clip(test_pred, 0, 1)})
    submission.to_csv(submission_path, index=False)

    plots = [] if args.no_plots or pred_hold is None else save_plots(results_dir, histories, baseline_histories, y[hold_idx], pred_hold, cm, importance, comparison)
    write_report(report_path, args, preprocessor, histories, metrics, comparison, cm, importance, plots, split_sizes, train_seconds)
    (results_dir / "metrics.json").write_text(
        json.dumps(
            {
                "preset": args.preset,
                "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
                "split_sizes": split_sizes,
                "metrics": metrics,
                "comparison": comparison,
                "confusion_matrix": cm.tolist() if cm is not None else None,
                "permutation_importance": importance,
                "histories": histories,
                "baseline_histories": baseline_histories,
                "train_seconds": train_seconds,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    example_score = float(test_pred[0])
    print(f"Saved model: {model_path}")
    print(f"Saved submission: {submission_path}")
    print(f"Saved report: {report_path}")
    for split, m in metrics.items():
        print(f"{split}: RMSE={m['rmse']:.5f} MAE={m['mae']:.5f} R2={m['r2']:.5f} level-acc={m['risk_level_accuracy']:.4f}")
    for name, m in comparison.items():
        print(f"  {name}: RMSE={m['rmse']:.5f} R2={m['r2']:.5f} level-acc={m['risk_level_accuracy']:.4f}")
    print(f"Example risk: {example_score:.3f} -> {risk_level(example_score)}")


if __name__ == "__main__":
    main()
