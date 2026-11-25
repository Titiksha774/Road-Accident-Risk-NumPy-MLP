# AI-Based Road Accident Risk Prediction

This project trains a Deep Learning / MLP model (written from scratch in NumPy) to estimate road accident risk from road, environmental, timing, and historical accident features.

**Version 2 (improved):** hold-out test R² **0.886** (RMSE 0.0560), up from R² 0.771 (RMSE 0.0796) for the model saved by version 1.
See `Mini_Project_Report.docx` for the full write-up and an explanation of every change.

Model: an ensemble of 5 MLPs (Dense 128 → 64 → 32 → 1, ReLU + sigmoid output), Adam with a cosine learning-rate schedule, trained on a 70/15/15 train/validation/hold-out split. Full numbers are in `training_report.md` and `results/metrics.json`.

## Data

The data comes from the Kaggle Playground Series competition *Predicting Road Accident Risk*. The CSV files are **not included** in this repository. Download `train.csv`, `test.csv` and `sample_submission.csv` from the competition page and put them in the project folder.

## Files

- `risk_model.py` - preprocessing, NumPy MLP model, prediction helpers
- `train_model.py` - trains the model and creates outputs
- `predict_example.py` - predicts one custom road situation
- `app.py` - optional Streamlit dashboard
- `road_accident_mlp.pkl` - trained improved model (v2)
- `road_accident_mlp_v1_original.pkl` - the original v1 model, kept for comparison
- `eda.py` - exploratory data analysis (writes `results/eda*.png` and `results/eda.json`)
- `ablation_study.py` - compares model variants on the hold-out split
- `results/` - plots and `metrics.json` written by training
- `training_report.md` - auto-generated training report (architecture, metrics, confusion matrix)
- `Mini_Project_Report.docx` - mini-project report
- `simple_version/` - a shorter single-file version of the pipeline with its own report and outputs

## Run Training

```bash
python train_model.py                      # improved model (default preset)
python train_model.py --compare-baseline   # also retrains the original v1 settings on the same split and compares
python train_model.py --preset baseline    # original v1 settings only
```

Quick test run:

```bash
python train_model.py --max-rows 50000 --epochs 5
```

All older options (`--epochs`, `--batch-size`, `--learning-rate`, `--ensemble-size`, `--max-rows`, `--seed`, ...) still work and override the preset.
New options: `--preset`, `--hidden`, `--dropout`, `--weight-decay`, `--lr-schedule`, `--holdout-size`, `--results-dir`, `--compare-baseline`, `--no-plots`.

## Outputs

Training creates:

- `road_accident_mlp.pkl`
- `submission.csv`
- `training_report.md`
- `results/*.png` and `results/metrics.json` (plots need `matplotlib`; they are skipped if it is not installed)

## Data split (v2)

`train.csv` is shuffled once (seed 42) and split 70 / 15 / 15 into train / validation / hold-out test.
Validation is used for keeping the best epoch; the hold-out test set is only used for the final evaluation.

## Risk Levels

- `0.00-0.24`: LOW
- `0.25-0.49`: MODERATE
- `0.50-0.74`: HIGH
- `0.75-1.00`: VERY HIGH

## Dashboard

Install Streamlit if needed:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

The dashboard lets a user enter road conditions and returns the predicted risk score and level.
