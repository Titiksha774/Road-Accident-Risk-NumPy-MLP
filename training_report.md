# Road Accident Risk Prediction Report

Preset: `improved`

## Model

- Algorithm: Multi-Layer Perceptron / Deep Neural Network implemented with NumPy
- Architecture: input features -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, sigmoid)
- Ensemble: average of 5 networks trained with different seeds
- Optimiser: Adam, peak learning rate 0.002, schedule `cosine`, batch size 512, up to 20 epochs
- Dropout: 0.0; output-layer initialisation at target mean: True; keep best-validation epoch: True
- Output: `accident_risk` score between 0 and 1
- Risk levels: LOW, MODERATE, HIGH, VERY HIGH
- Training time: 208 s

## Data Split

- Train rows: 362,428
- Validation rows (early stopping / model selection): 77,663
- Hold-out test rows (never used for training or tuning): 77,663

## Metrics

| Split | RMSE | MAE | R2 | Risk-level accuracy |
|---|---:|---:|---:|---:|
| validation | 0.05653 | 0.04394 | 0.88475 | 0.8356 |
| hold-out test | 0.05601 | 0.04381 | 0.88653 | 0.8340 |

## Risk-level Confusion Matrix (hold-out test; rows = actual, columns = predicted)

| | LOW | MODERATE | HIGH | VERY HIGH |
|---|---:|---:|---:|---:|
| LOW | 17185 | 4065 | 0 | 0 |
| MODERATE | 3080 | 36969 | 1877 | 0 |
| HIGH | 0 | 3050 | 9954 | 182 |
| VERY HIGH | 1 | 2 | 633 | 665 |

## Training History (validation, per epoch)

### Model 1

| Epoch | LR | Train RMSE | Validation RMSE | Validation MAE | Validation R2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.00e-03 | 0.05859 | 0.05842 | 0.04557 | 0.87694 |
| 2 | 1.99e-03 | 0.05796 | 0.05776 | 0.04507 | 0.87970 |
| 3 | 1.95e-03 | 0.05841 | 0.05814 | 0.04538 | 0.87809 |
| 4 | 1.88e-03 | 0.05789 | 0.05763 | 0.04494 | 0.88023 |
| 5 | 1.79e-03 | 0.05707 | 0.05695 | 0.04432 | 0.88306 |
| 6 | 1.68e-03 | 0.05709 | 0.05696 | 0.04431 | 0.88302 |
| 7 | 1.55e-03 | 0.05705 | 0.05690 | 0.04431 | 0.88325 |
| 8 | 1.41e-03 | 0.05690 | 0.05680 | 0.04423 | 0.88364 |
| 9 | 1.25e-03 | 0.05695 | 0.05688 | 0.04425 | 0.88334 |
| 10 | 1.09e-03 | 0.05695 | 0.05694 | 0.04422 | 0.88309 |
| 11 | 9.28e-04 | 0.05681 | 0.05684 | 0.04409 | 0.88351 |
| 12 | 7.67e-04 | 0.05664 | 0.05671 | 0.04407 | 0.88401 |
| 13 | 6.12e-04 | 0.05659 | 0.05670 | 0.04404 | 0.88407 |
| 14 | 4.69e-04 | 0.05658 | 0.05677 | 0.04407 | 0.88378 |
| 15 | 3.39e-04 | 0.05649 | 0.05669 | 0.04409 | 0.88409 |
| 16 | 2.29e-04 | 0.05646 | 0.05669 | 0.04407 | 0.88410 |
| 17 | 1.39e-04 | 0.05646 | 0.05673 | 0.04407 | 0.88393 |
| 18 | 7.36e-05 | 0.05641 | 0.05668 | 0.04405 | 0.88413 |
| 19 | 3.35e-05 | 0.05640 | 0.05668 | 0.04405 | 0.88416 |
| 20 | 2.00e-05 | 0.05639 | 0.05668 | 0.04406 | 0.88413 |

### Model 2

| Epoch | LR | Train RMSE | Validation RMSE | Validation MAE | Validation R2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.00e-03 | 0.05896 | 0.05871 | 0.04590 | 0.87571 |
| 2 | 1.99e-03 | 0.05821 | 0.05798 | 0.04519 | 0.87876 |
| 3 | 1.95e-03 | 0.05772 | 0.05749 | 0.04474 | 0.88082 |
| 4 | 1.88e-03 | 0.05726 | 0.05702 | 0.04437 | 0.88275 |
| 5 | 1.79e-03 | 0.05742 | 0.05723 | 0.04464 | 0.88188 |
| 6 | 1.68e-03 | 0.05719 | 0.05701 | 0.04439 | 0.88279 |
| 7 | 1.55e-03 | 0.05713 | 0.05699 | 0.04439 | 0.88288 |
| 8 | 1.41e-03 | 0.05702 | 0.05697 | 0.04427 | 0.88297 |
| 9 | 1.25e-03 | 0.05689 | 0.05684 | 0.04420 | 0.88350 |
| 10 | 1.09e-03 | 0.05687 | 0.05683 | 0.04417 | 0.88355 |
| 11 | 9.28e-04 | 0.05680 | 0.05680 | 0.04415 | 0.88367 |
| 12 | 7.67e-04 | 0.05695 | 0.05696 | 0.04434 | 0.88301 |
| 13 | 6.12e-04 | 0.05684 | 0.05691 | 0.04430 | 0.88320 |
| 14 | 4.69e-04 | 0.05663 | 0.05678 | 0.04413 | 0.88374 |
| 15 | 3.39e-04 | 0.05663 | 0.05679 | 0.04416 | 0.88371 |
| 16 | 2.29e-04 | 0.05652 | 0.05674 | 0.04408 | 0.88392 |
| 17 | 1.39e-04 | 0.05648 | 0.05672 | 0.04408 | 0.88398 |
| 18 | 7.36e-05 | 0.05646 | 0.05670 | 0.04407 | 0.88406 |
| 19 | 3.35e-05 | 0.05645 | 0.05671 | 0.04406 | 0.88404 |
| 20 | 2.00e-05 | 0.05645 | 0.05671 | 0.04407 | 0.88404 |

### Model 3

| Epoch | LR | Train RMSE | Validation RMSE | Validation MAE | Validation R2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.00e-03 | 0.05906 | 0.05887 | 0.04581 | 0.87502 |
| 2 | 1.99e-03 | 0.05815 | 0.05787 | 0.04514 | 0.87923 |
| 3 | 1.95e-03 | 0.05779 | 0.05752 | 0.04484 | 0.88069 |
| 4 | 1.88e-03 | 0.05780 | 0.05758 | 0.04497 | 0.88045 |
| 5 | 1.79e-03 | 0.05739 | 0.05718 | 0.04448 | 0.88210 |
| 6 | 1.68e-03 | 0.05717 | 0.05706 | 0.04439 | 0.88261 |
| 7 | 1.55e-03 | 0.05733 | 0.05730 | 0.04446 | 0.88162 |
| 8 | 1.41e-03 | 0.05697 | 0.05694 | 0.04429 | 0.88308 |
| 9 | 1.25e-03 | 0.05715 | 0.05703 | 0.04444 | 0.88270 |
| 10 | 1.09e-03 | 0.05685 | 0.05683 | 0.04416 | 0.88353 |
| 11 | 9.28e-04 | 0.05705 | 0.05698 | 0.04439 | 0.88290 |
| 12 | 7.67e-04 | 0.05686 | 0.05691 | 0.04418 | 0.88321 |
| 13 | 6.12e-04 | 0.05673 | 0.05680 | 0.04411 | 0.88366 |
| 14 | 4.69e-04 | 0.05660 | 0.05671 | 0.04409 | 0.88402 |
| 15 | 3.39e-04 | 0.05657 | 0.05672 | 0.04406 | 0.88399 |
| 16 | 2.29e-04 | 0.05651 | 0.05668 | 0.04406 | 0.88416 |
| 17 | 1.39e-04 | 0.05647 | 0.05670 | 0.04406 | 0.88406 |
| 18 | 7.36e-05 | 0.05645 | 0.05669 | 0.04406 | 0.88411 |
| 19 | 3.35e-05 | 0.05645 | 0.05669 | 0.04407 | 0.88412 |
| 20 | 2.00e-05 | 0.05644 | 0.05670 | 0.04406 | 0.88407 |

### Model 4

| Epoch | LR | Train RMSE | Validation RMSE | Validation MAE | Validation R2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.00e-03 | 0.05859 | 0.05856 | 0.04569 | 0.87634 |
| 2 | 1.99e-03 | 0.05766 | 0.05758 | 0.04481 | 0.88044 |
| 3 | 1.95e-03 | 0.05760 | 0.05752 | 0.04471 | 0.88071 |
| 4 | 1.88e-03 | 0.05747 | 0.05737 | 0.04463 | 0.88131 |
| 5 | 1.79e-03 | 0.05761 | 0.05756 | 0.04468 | 0.88052 |
| 6 | 1.68e-03 | 0.05742 | 0.05731 | 0.04456 | 0.88154 |
| 7 | 1.55e-03 | 0.05706 | 0.05695 | 0.04426 | 0.88304 |
| 8 | 1.41e-03 | 0.05714 | 0.05706 | 0.04441 | 0.88257 |
| 9 | 1.25e-03 | 0.05719 | 0.05705 | 0.04443 | 0.88263 |
| 10 | 1.09e-03 | 0.05702 | 0.05700 | 0.04425 | 0.88283 |
| 11 | 9.28e-04 | 0.05683 | 0.05683 | 0.04421 | 0.88353 |
| 12 | 7.67e-04 | 0.05675 | 0.05680 | 0.04416 | 0.88367 |
| 13 | 6.12e-04 | 0.05675 | 0.05680 | 0.04420 | 0.88364 |
| 14 | 4.69e-04 | 0.05658 | 0.05678 | 0.04414 | 0.88374 |
| 15 | 3.39e-04 | 0.05657 | 0.05675 | 0.04410 | 0.88385 |
| 16 | 2.29e-04 | 0.05653 | 0.05677 | 0.04410 | 0.88377 |
| 17 | 1.39e-04 | 0.05646 | 0.05674 | 0.04408 | 0.88390 |
| 18 | 7.36e-05 | 0.05646 | 0.05672 | 0.04411 | 0.88397 |
| 19 | 3.35e-05 | 0.05642 | 0.05671 | 0.04407 | 0.88402 |
| 20 | 2.00e-05 | 0.05641 | 0.05671 | 0.04407 | 0.88403 |

### Model 5

| Epoch | LR | Train RMSE | Validation RMSE | Validation MAE | Validation R2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.00e-03 | 0.05952 | 0.05928 | 0.04637 | 0.87326 |
| 2 | 1.99e-03 | 0.05788 | 0.05764 | 0.04489 | 0.88018 |
| 3 | 1.95e-03 | 0.05770 | 0.05757 | 0.04474 | 0.88049 |
| 4 | 1.88e-03 | 0.05738 | 0.05718 | 0.04446 | 0.88210 |
| 5 | 1.79e-03 | 0.05735 | 0.05721 | 0.04444 | 0.88196 |
| 6 | 1.68e-03 | 0.05733 | 0.05721 | 0.04448 | 0.88197 |
| 7 | 1.55e-03 | 0.05708 | 0.05691 | 0.04434 | 0.88320 |
| 8 | 1.41e-03 | 0.05744 | 0.05729 | 0.04464 | 0.88164 |
| 9 | 1.25e-03 | 0.05699 | 0.05686 | 0.04415 | 0.88342 |
| 10 | 1.09e-03 | 0.05721 | 0.05713 | 0.04447 | 0.88230 |
| 11 | 9.28e-04 | 0.05689 | 0.05684 | 0.04423 | 0.88351 |
| 12 | 7.67e-04 | 0.05678 | 0.05684 | 0.04419 | 0.88349 |
| 13 | 6.12e-04 | 0.05688 | 0.05692 | 0.04430 | 0.88316 |
| 14 | 4.69e-04 | 0.05664 | 0.05678 | 0.04415 | 0.88373 |
| 15 | 3.39e-04 | 0.05658 | 0.05674 | 0.04406 | 0.88390 |
| 16 | 2.29e-04 | 0.05652 | 0.05670 | 0.04407 | 0.88406 |
| 17 | 1.39e-04 | 0.05647 | 0.05671 | 0.04409 | 0.88402 |
| 18 | 7.36e-05 | 0.05646 | 0.05670 | 0.04407 | 0.88407 |
| 19 | 3.35e-05 | 0.05646 | 0.05670 | 0.04406 | 0.88406 |
| 20 | 2.00e-05 | 0.05645 | 0.05670 | 0.04407 | 0.88407 |

## Permutation Feature Importance (hold-out test)

| Feature | RMSE increase |
|---|---:|
| curvature | 0.08588 |
| speed_limit | 0.08188 |
| lighting | 0.07772 |
| weather | 0.02740 |
| num_reported_accidents | 0.01227 |
| holiday | 0.00022 |
| public_road | 0.00014 |
| num_lanes | 0.00008 |
| time_of_day | 0.00002 |
| road_type | 0.00002 |
| road_signs_present | 0.00001 |
| school_season | 0.00000 |

## Features Used

num_lanes, curvature, speed_limit, road_signs_present, public_road, holiday, school_season, num_reported_accidents, curvature^2, road_type=highway, road_type=rural, road_type=urban, lighting=daylight, lighting=dim, lighting=night, weather=clear, weather=foggy, weather=rainy, time_of_day=afternoon, time_of_day=evening, time_of_day=morning, num_lanes==1, num_lanes==2, num_lanes==3, num_lanes==4, speed_limit==25, speed_limit==35, speed_limit==45, speed_limit==60, speed_limit==70, num_reported_accidents==0, num_reported_accidents==1, num_reported_accidents==2, num_reported_accidents==3, num_reported_accidents==4, num_reported_accidents==5, num_reported_accidents==6

## Figures

- `results/learning_curves.png`
- `results/pred_vs_actual.png`
- `results/residuals.png`
- `results/confusion_matrix.png`
- `results/feature_importance.png`

## Interpretation

The model estimates risk from the combined road, weather, lighting, timing, and accident-history conditions. Higher scores indicate situations that look more similar to historically higher-risk examples in the training data. Curvature, lighting (night), speed limit (60+), weather and the number of previously reported accidents carry almost all of the signal.