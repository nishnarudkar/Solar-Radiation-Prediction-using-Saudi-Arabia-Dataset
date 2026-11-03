# Solar Radiation Prediction using Saudi Arabia Dataset

[![CI/CD](https://github.com/Aamir-Sarang31/Solar-Radiation-Prediction-using-Saudi-Arabia-Dataset/actions/workflows/ci.yml/badge.svg)](https://github.com/Aamir-Sarang31/Solar-Radiation-Prediction-using-Saudi-Arabia-Dataset/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-RandomizedSearchCV-F7931E.svg)](https://scikit-learn.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20%26%20Model%20Versioning-945DD6.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![Flask](https://img.shields.io/badge/Flask-Web%20Dashboard-000000.svg?logo=flask&logoColor=white)](app.py)
[![Render](https://img.shields.io/badge/Render-Live%20Demo-46E3B7.svg?logo=render&logoColor=white)](https://solar-radiation-prediction-uw40.onrender.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Research Paper & Project Context:** Solar Irradiance Forecasting & Renewable Energy Predictive Modeling using Multi-Station Meteorological Observations across Saudi Arabia.
> 
> 🌐 **Live Web Application:** [https://solar-radiation-prediction-uw40.onrender.com](https://solar-radiation-prediction-uw40.onrender.com)

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Benchmark Results](#-controlled-model-performance--benchmarks-10-fold-cross-validation)
- [Deep Learning Architectures](#-deep-learning-architectures)
- [Hyperparameter Tuning (RandomizedSearchCV)](#-automated-hyperparameter-tuning-randomizedsearchcv)
- [Data & Model Versioning (DVC)](#-data--model-versioning-dvc)
- [MLOps Pipeline](#-mlops-mlflow-experiment-tracking--registry-gating)
- [Project Structure](#-project-structure)
- [Quickstart Guide](#-quickstart-guide)
- [Web Application & Deployment](#-web-application--deployment)
  - [Interface Showcase](#-application-interface-showcase)
  - [Interactive Features](#-key-web-application-features)
  - [Deployment Guides (Docker & Render)](#-deployment-architectures--guides)
  - [REST API Reference](#-rest-api-endpoints)
- [Dataset](#-dataset)
- [CI/CD Pipeline](#-cicd-workflows)
- [Testing](#-testing)
- [Authors & Acknowledgements](#-authors--acknowledgements)
- [License](#-license)

---

## 🌟 Project Overview

This project forecasts **Global Horizontal Irradiance (GHI)** using meteorological observations collected from **43 weather monitoring stations across Saudi Arabia (2017–2021)**.

We systematically evaluate and benchmark **11 machine learning and deep learning models** under rigorous **10-Fold Cross-Validation** and **43-Fold Leave-One-Station-Out (LOSO) Cross-Validation**, with:

- 🧠 **3 Custom Deep Learning Architectures** (FT-Transformer, LSTM, 1D CNN)
- 📊 **8 Classical ML Baselines** (SVR, HGB, XGBoost, RF, ANN, LR, DT, KNN)
- 🔬 **Automated Hyperparameter Optimization** via Scikit-Learn `RandomizedSearchCV`
- 📦 **Data & Model Version Control** via DVC (Data Version Control)
- 📈 **End-to-end MLflow Experiment Tracking** with automated Model Registry promotion gating
- 🌐 **Interactive Flask Web Application** with real-time predictions and station maps
- ⚙️ **CI/CD Pipelines** with GitHub Actions

---

## 🏆 Key Features

| Feature | Description |
|---|---|
| **11-Model Controlled Benchmark** | All models evaluated under identical 10-fold CV partitions, same seed, and uniform scaling |
| **FT-Transformer Architecture** | Feature Tokenizer Transformer achieving **R² = 0.9919** — best-in-class for tabular solar data |
| **RandomizedSearchCV Tuning** | Scikit-Learn randomized search across continuous distributions for classical & PyTorch models |
| **Data & Model Versioning (DVC)** | DVC tracking for `dataset.csv` and model weights with reproducible `dvc.yaml` pipelines |
| **MLflow Experiment Tracking** | Every run logs parameters, metrics, plots, and model artifacts to SQLite |
| **Model Registry Promotion Gate** | Automated quality gate blocks deployment if candidate fails RMSE/R² thresholds |
| **Interactive Web Dashboard** | Flask app with Leaflet.js map, multi-model predictions, and Plotly visualizations |
| **Bright Neon Dark Mode** | Industry demo UI with sleek light/dark toggle, glowing HUD overlays, and persistent theme state |
| **Precision Vector Gradient Map** | Hardware-accelerated Canvas clipping to 3,174-pt geoBoundaries ADM0 data with dynamic GHI beacon pins |
| **CI/CD Integration** | GitHub Actions for linting, testing, smoke training, and promotion gate verification |

---

## 🧪 Controlled Model Performance & Benchmarks (10-Fold Cross-Validation)

All 11 models evaluated under the **exact same 10-fold CV partitions, same random seed (42), and uniform feature/target scaling** on `dataset.csv` (1,649 records across 43 stations). Sorted strictly by ascending RMSE (primary gate metric):

| Rank | Model Architecture | Category | MAE (Wh/m²) | 95% CI (MAE) | RMSE (Wh/m²) | 95% CI (RMSE) | R² Score | 95% CI (R²) | $p$-value (vs FT-Trans) | Inference (ms) |
| :---: | :--- | :--- | :--- | :---: | :--- | :---: | :--- | :---: | :---: | :---: |
| 🥇 | **FT-Transformer** | **Deep Learning** | **81.10 ± 7.21** | **[75.94, 86.26]** | **110.47 ± 12.04** | **[101.86, 119.08]** | **0.9919 ± 0.0023** | **[0.9902, 0.9935]** | **Baseline Champion** | **0.68** |
| 🥈 | Artificial Neural Network (ANN) | Neural Baseline | 129.13 ± 9.53 | [122.31, 135.95] | 172.50 ± 14.25 | [162.31, 182.70] | 0.9807 ± 0.0037 | [0.9781, 0.9833] | $2.00 \times 10^{-7}\ $ | 0.18 |
| 🥉 | Histogram Gradient Boosting (HGB) | Ensemble | 129.37 ± 12.36 | [120.53, 138.21] | 178.39 ± 22.46 | [162.32, 194.46] | 0.9792 ± 0.0055 | [0.9753, 0.9831] | $3.64 \times 10^{-6}\ $ | 0.79 |
| 4 | Support Vector Regression (SVR) | Classical ML | 114.11 ± 9.95 | [106.99, 121.23] | 188.36 ± 35.15 | [163.22, 213.51] | 0.9765 ± 0.0080 | [0.9708, 0.9822] | $1.23 \times 10^{-4}\ $ | 0.19 |
| 5 | Linear Regression (LR) | Classical ML | 134.28 ± 9.64 | [127.38, 141.18] | 194.14 ± 42.34 | [163.86, 224.43] | 0.9746 ± 0.0122 | [0.9659, 0.9833] | $1.59 \times 10^{-4}\ $ | 0.07 |
| 6 | XGBoost | Ensemble | 155.65 ± 11.10 | [147.71, 163.59] | 215.21 ± 20.70 | [200.40, 230.02] | 0.9697 ± 0.0067 | [0.9649, 0.9745] | $1.06 \times 10^{-7}\ $ | 0.33 |
| 7 | Random Forest (RF) | Ensemble | 163.60 ± 17.87 | [150.82, 176.38] | 230.28 ± 26.87 | [211.06, 249.51] | 0.9653 ± 0.0090 | [0.9589, 0.9717] | $1.49 \times 10^{-7}\ $ | 30.86 |
| 8 | 1D CNN | **Deep Learning** | 214.28 ± 19.76 | [200.14, 228.42] | 283.96 ± 25.74 | [265.55, 302.37] | 0.9471 ± 0.0133 | [0.9376, 0.9566] | $3.03 \times 10^{-10}\ $ | 1.32 |
| 9 | Solar LSTM | **Deep Learning** | 273.68 ± 40.51 | [244.70, 302.66] | 362.18 ± 55.91 | [322.18, 402.18] | 0.9117 ± 0.0335 | [0.8877, 0.9357] | $4.41 \times 10^{-9}\ $ | 0.52 |
| 10 | Decision Tree (DT) | Classical ML | 280.33 ± 23.14 | [263.78, 296.88] | 410.28 ± 51.76 | [373.25, 447.31] | 0.8895 ± 0.0312 | [0.8672, 0.9118] | $4.33 \times 10^{-8}\ $ | 0.08 |
| 11 | K-Nearest Neighbors (KNN) | Classical ML | 337.79 ± 22.74 | [321.52, 354.06] | 447.20 ± 27.45 | [427.56, 466.84] | 0.8699 ± 0.0236 | [0.8530, 0.8868] | $4.07 \times 10^{-11}\ $ | 2.94 |

> **Statistical Significance & Confidence Analysis:** Paired two-tailed Student's t-tests ($df = 9$) across all 10 cross-validation folds confirm that the performance superiority of **FT-Transformer** over every competing baseline is statistically significant at $p < 0.001\ (***)$ with exact $p$-values ranging from $3.64 \times 10^{-6}$ (vs HGB) down to $4.07 \times 10^{-11}$ (vs KNN). In addition, the 95% Confidence Interval for FT-Transformer's RMSE ($[101.86, 119.08]\text{ Wh/m}^2$) exhibits **zero overlap** with any baseline (nearest being ANN at $[162.31, 182.70]$), demonstrating decisive outperformance. Full t-statistics, exact p-values, and CIs are exported in [`results/statistical_significance.csv`](results/statistical_significance.csv).

### 🗺️ Controlled Spatial Generalization: 43-Fold Leave-One-Station-Out (LOSO) CV

To rigorously evaluate real-world generalization to completely **unseen geographical regions and distinct microclimates**, every model is evaluated under strict 43-Fold Leave-One-Station-Out (LOSO) Cross-Validation. In each fold, all observations from one complete weather station are held out for testing while training on the remaining 42 stations:

| Rank | Model Architecture | Category | MAE (Wh/m²) | 95% CI (MAE) | RMSE (Wh/m²) | 95% CI (RMSE) | R² Score | 95% CI (R²) | $p$-value (vs FT-Trans) | Inference (ms) |
| :---: | :--- | :--- | :--- | :---: | :--- | :---: | :--- | :---: | :---: | :---: |
| 🥇 | **FT-Transformer** | **Deep Learning** | **90.66 ± 33.82** | **[80.26, 101.07]** | **120.65 ± 54.64** | **[103.83, 137.47]** | **0.9803 ± 0.0352** | **[0.9694, 0.9911]** | **Baseline Champion** | **0.13** |
| 🥈 | Artificial Neural Network (ANN) | Neural Baseline | 140.69 ± 43.11 | [127.26, 154.11] | 181.66 ± 56.55 | [164.05, 199.27] | 0.9690 ± 0.0253 | [0.9611, 0.9769] | $2.91 \times 10^{-9}\ $ | 0.01 |
| 🥉 | Linear Regression (LR) | Classical ML | 140.85 ± 40.26 | [128.46, 153.24] | 189.93 ± 79.23 | [165.55, 214.32] | 0.9592 ± 0.0661 | [0.9388, 0.9795] | $1.70 \times 10^{-8}\ $ | 0.005 |
| 4 | Histogram Gradient Boosting (HGB) | Ensemble | 143.23 ± 47.80 | [128.52, 157.94] | 190.59 ± 71.02 | [168.73, 212.44] | 0.9626 ± 0.0413 | [0.9499, 0.9753] | $1.76 \times 10^{-8}\ $ | 0.04 |
| 5 | Support Vector Regression (SVR) | Classical ML | 136.55 ± 73.49 | [113.93, 159.17] | 200.18 ± 111.25 | [165.94, 234.41] | 0.9575 ± 0.0489 | [0.9425, 0.9726] | $2.09 \times 10^{-5}\ $ | 0.04 |
| 6 | XGBoost | Ensemble | 178.43 ± 70.57 | [156.71, 200.15] | 238.79 ± 113.20 | [203.95, 273.63] | 0.9403 ± 0.0692 | [0.9190, 0.9616] | $1.28 \times 10^{-8}\ $ | 0.04 |
| 7 | Random Forest (RF) | Ensemble | 188.66 ± 74.86 | [165.62, 211.70] | 251.29 ± 106.76 | [218.43, 284.14] | 0.9324 ± 0.0720 | [0.9102, 0.9545] | $9.46 \times 10^{-11}\ $ | 1.16 |
| 8 | 1D CNN | **Deep Learning** | 241.11 ± 74.89 | [218.06, 264.16] | 312.02 ± 92.24 | [283.64, 340.41] | 0.9030 ± 0.0902 | [0.8753, 0.9308] | $1.16 \times 10^{-18}\ $ | 0.21 |
| 9 | Solar LSTM | **Deep Learning** | 335.52 ± 90.88 | [307.55, 363.49] | 417.49 ± 112.91 | [382.74, 452.24] | 0.8496 ± 0.1056 | [0.8171, 0.8820] | $9.41 \times 10^{-21}\ $ | 0.13 |
| 10 | Decision Tree (DT) | Classical ML | 306.23 ± 125.86 | [267.50, 344.96] | 422.54 ± 178.48 | [367.61, 477.47] | 0.8169 ± 0.1826 | [0.7607, 0.8730] | $5.71 \times 10^{-14}\ $ | 0.01 |
| 11 | K-Nearest Neighbors (KNN) | Classical ML | 371.13 ± 107.23 | [338.13, 404.14] | 469.74 ± 134.50 | [428.35, 511.13] | 0.7890 ± 0.1911 | [0.7302, 0.8478] | $1.02 \times 10^{-21}\ $ | 0.07 |

> **Statistical Significance & Spatial Generalization Analysis:** Paired two-tailed Student's t-tests ($df = 42$) across all 43 held-out stations confirm that the spatial generalization superiority of **FT-Transformer** over every competing baseline is statistically significant at $p < 0.001\ (***)$ (e.g. vs ANN: $t = 7.49, p = 2.91 \times 10^{-9}$; vs LR: $t = 6.95, p = 1.70 \times 10^{-8}$; vs HGB: $t = 6.94, p = 1.76 \times 10^{-8}$; vs SVR: $t = 4.79, p = 2.09 \times 10^{-5}$). In addition, the 95% Confidence Interval for FT-Transformer's RMSE ($[103.83, 137.47]\text{ Wh/m}^2$) exhibits **zero overlap** with any baseline (nearest being ANN at $[164.05, 199.27]$), demonstrating decisive spatial outperformance with a **61.01 Wh/m²** (33.6%) error reduction. Full per-station data and t-test statistics are exported in [`results/loso_statistical_significance.csv`](results/loso_statistical_significance.csv).

---

## 🏗️ Deep Learning Architectures

### 1. FT-Transformer (Feature Tokenizer Transformer) — Champion Model

```
Input: (B, 21 features)
  ├── Numerical Feature Tokenizer: 21 × Linear(1, 64) projections → (B, 21, 64)
  ├── Prepend Learnable [CLS] Token → (B, 22, 64)
  ├── 2× Pre-LN TransformerEncoderLayer (d_model=64, 4 Heads, DimFF=48, Dropout=0.10)
  └── Extract [CLS] Token → LayerNorm → Linear(64, 1) → Output GHI (B, 1)
```

**Key Innovation:** Each numerical feature is independently projected into a learned embedding space via its own linear layer, then processed through multi-head self-attention — enabling the model to learn complex inter-feature interactions (e.g., how DNI modulates with temperature and pressure seasonally).

### 2. Solar 1D CNN

```
Input: (B, 1, 21)
  ├── Stem Conv1d(1, 64) + BatchNorm1d + GELU
  ├── Residual ConvBlock1D(64, 128) + Residual ConvBlock1D(128, 256)
  └── AdaptiveAvgPool1d(1) → Flatten → MLP Head → Output GHI (B, 1)
```

### 3. Solar LSTM

```
Input: (B, 21, 1)
  ├── Linear Feature Projection (1 → 32) + LayerNorm
  ├── 1-Layer Bidirectional LSTM(32, 64, Dropout=0.30)
  └── Global Average Pooling → LayerNorm → MLP Head → Output GHI (B, 1)
```

---

## 🔍 Automated Hyperparameter Tuning (RandomizedSearchCV)

The project includes an automated hyperparameter optimization pipeline using **Scikit-Learn's `RandomizedSearchCV`** paradigm with continuous probability distributions (`scipy.stats.loguniform`, `uniform`, `randint`).

### How It Works

1. **Classical Models (SVR, HGB, XGBoost, RF, ANN)**: Uses `sklearn.model_selection.RandomizedSearchCV` wrapped inside leakage-free `Pipeline` and `TransformedTargetRegressor` estimators evaluated on pre-computed K-Fold CV splits.
2. **Deep Learning Models (Transformer, LSTM, 1D CNN)**: Uses a randomized hyperparameter search loop sampling architecture hyperparameters (learning rate, weight decay, token dimension, attention heads) with full K-Fold cross-validation.
3. Every search run is tracked automatically in **MLflow**.
4. The best configuration is exported to a JSON file in `configs/` for reproducible training.

### Supported Models & Search Spaces

| Model | Tunable Hyperparameters | Distribution / Candidates |
|---|---|---|
| **FT-Transformer** | `d_model`, `nhead`, `num_layers`, `dim_feedforward`, `dropout`, `lr`, `weight_decay`, `batch_size` | Discrete [32, 48, 64], loguniform(1e-4, 5e-3) |
| **LSTM** | `embed_dim`, `hidden_dim`, `num_layers`, `dropout`, `bidirectional`, `lr`, `batch_size` | Discrete [16, 32, 64], loguniform(1e-4, 5e-3) |
| **1D CNN** | `base_channels`, `dropout`, `lr`, `weight_decay`, `batch_size` | Discrete [16, 32, 64], loguniform(1e-4, 5e-3) |
| **SVR** | `C`, `epsilon`, `gamma` | loguniform(1.0, 1000.0), loguniform(0.01, 1.0) |
| **HGB** | `max_depth`, `learning_rate`, `max_iter`, `min_samples_leaf`, `l2_regularization` | randint(3, 11), loguniform(0.01, 0.3) |
| **XGBoost** | `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `reg_alpha` | randint(3, 11), loguniform(0.01, 0.3) |
| **Random Forest** | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` | Discrete [50, 100, 150, 200], randint(5, 26) |
| **ANN (MLP)** | `hidden_layer_sizes`, `alpha`, `learning_rate_init` | Layer combinations, loguniform(1e-4, 10.0) |

### Usage

```bash
# Tune FT-Transformer across 10-Fold CV (30 iterations)
python src/tune.py --model transformer --trials 30 --cv-folds 10

# Tune SVR with RandomizedSearchCV (50 iterations)
python src/tune.py --model svr --trials 50 --cv-folds 10

# Retrain with tuned hyperparameters
python src/train.py --model transformer --config configs/transformer_best.json --cv kfold
```

Tuned configurations are saved to `configs/{model_name}_best.json` for reproducibility.

---

## 📦 Data & Model Versioning (DVC)

The project integrates **DVC (Data Version Control)** to define the ML pipeline as a reproducible directed acyclic graph (DAG). Dataset (`dataset.csv`) and model checkpoints (`model/`) remain in Git for immediate reproducibility after cloning.

### DVC Pipeline (`dvc.yaml`)

The complete ML workflow — from hyperparameter tuning through training to evaluation — is codified as a connected DAG with explicit dependency tracking:

```bash
# Visualize pipeline DAG
dvc dag

# Reproduce full pipeline (only re-runs stages whose dependencies changed)
dvc repro

# Reproduce a specific stage
dvc repro train
```

```
+------+     +---------+     +----------+
| tune | --> |  train  | --> | evaluate |
+------+     +---------+     +----------+
```

---

## 📊 MLOps: MLflow Experiment Tracking & Registry Gating

The project uses **MLflow** with an SQLite backend (`sqlite:///mlflow.db`) for experiment tracking, parameter logging, metric recording, and model registry governance.

### Logged Parameters
- **Architecture**: `model_architecture`, `d_model`, `n_heads`, `num_layers`, `hidden_dim`, `total_trainable_parameters`
- **Optimization**: `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `optimizer` (`AdamW`), `scheduler` (`CosineAnnealingLR`), `loss_function` (`MSELoss`)
- **Validation**: `cv_strategy` (`10-fold` or `43-fold LOSO`), `random_seed`, `num_features`

### Logged Metrics & Artifacts
- **Per-Fold & Summary Metrics**: MAE, MSE, RMSE, R², Training Time per fold (Mean ± Std).
- **Artifacts**: Checkpoints (`.pt`), Scaler objects (`.pkl`), Parity scatter plots, Residual error distribution charts.

### Model Registry Promotion Gate (CI/CD)

When a training run finishes:
1. `src/evaluate.py --gate` verifies candidate model metrics against quality thresholds (RMSE ≤ 300, R² ≥ 0.90).
2. Compares candidate RMSE against the current `@champion` model in the MLflow Model Registry.
3. If approved:
   - Registers new version in MLflow Model Registry (`SolarRadiationPredictor`).
   - Assigns the `@champion` model alias.
   - Automatically exports weights and scaler bundle to `model/production_dl_model.pt` and `model/production_dl_scaler.pkl`.
4. If candidate fails: blocks promotion in CI/CD.

```bash
# View all experiments, runs, and registered model versions
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
# Open http://127.0.0.1:5001
```

---

## 📁 Project Structure

```
Solar-Radiation-Prediction-using-Saudi-Arabia-Dataset/
│
├── .dvc/                          # DVC configuration and internal metadata
├── .dvcignore                     # DVC ignore rules
├── dvc.yaml                       # Reproducible DVC ML pipeline definition (tune → train → evaluate)
│
├── src/                           # Core source code
│   ├── __init__.py
│   ├── dataset.py                 # Data loading, feature definitions, CV splits
│   ├── train.py                   # MLflow-integrated training pipeline (10-Fold & LOSO CV)
│   ├── tune.py                    # RandomizedSearchCV hyperparameter optimization module
│   ├── evaluate.py                # MLflow Model Registry promotion gate
│   ├── export_results.py          # Granular CSV results exporter (samples, folds, stations)
│   ├── predict.py                 # SolarPredictor inference API
│   └── models/
│       ├── __init__.py            # MODEL_REGISTRY & build_model() factory
│       ├── transformer.py         # FT-Transformer architecture
│       ├── lstm.py                # Solar LSTM architecture
│       └── cnn1d.py               # Solar 1D CNN architecture
│
├── results/                       # Granular, publication-ready research CSV exports
│   ├── benchmark_summary.csv      # Cross-model master comparison summary (with 95% CIs)
│   ├── statistical_significance.csv # 10-Fold CV hypothesis testing (t-stats, p-values, 95% CIs)
│   ├── loso_statistical_significance.csv # 43-Fold LOSO hypothesis testing (t-stats, p-values, 95% CIs)
│   ├── paper_baselines.csv        # Published 8-model baseline results
│   ├── *_samples.csv              # Per-observation out-of-fold predictions & residuals
│   ├── *_folds.csv                # Per-fold performance metrics (MAE, RMSE, R², MBE)
│   └── *_stations.csv             # Station-by-station spatial metrics (all 43 stations)
│
├── tests/                         # Test suite (47 tests)
│   ├── __init__.py
│   ├── test_api.py                # Flask API endpoint tests
│   ├── test_models.py             # Forward/backward pass tests for all DL models
│   ├── test_preprocessing.py      # Dataset loading, CV split, and scaling tests
│   ├── test_registry_gate.py      # MLflow promotion gate tests
│   ├── test_results_export.py     # Results export & metrics calculation tests
│   └── test_tuning.py             # RandomizedSearchCV search space & tuning smoke tests
│
├── configs/                       # Tuned hyperparameter configurations (JSON)
│   ├── transformer_best.json      # Best FT-Transformer config (R² = 0.9919)
│   ├── cnn1d_best.json            # Best 1D CNN config (R² = 0.9471)
│   └── lstm_best.json             # Best Solar LSTM config (R² = 0.8930)
│
├── model/                         # Trained model checkpoints & scalers
│   ├── production_dl_model.pt     # Current production champion model
│   ├── production_dl_scaler.pkl   # Production scaler
│   ├── transformer_model.pt       # FT-Transformer checkpoint
│   ├── lstm_model.pt              # LSTM checkpoint
│   ├── cnn1d_model.pt             # 1D CNN checkpoint
│   └── *_classical_model.pkl      # Classical ML model checkpoints
│
├── Other Models/                  # Legacy pre-trained classical models
│
├── app.py                         # Flask web application & REST API server
├── wsgi.py                        # Production WSGI entry point (Gunicorn/Waitress)
├── templates/
│   └── index.html                 # Web frontend UI (Bootstrap 5, Leaflet.js, Plotly.js, Theme Switch)
├── static/
│   ├── style.css                  # Bright Neon Dark Mode & Solar Light design system
│   ├── app.js                     # Frontend interactive client, chart controllers & theme engine
│   └── saudi_boundary.js          # High-precision geoBoundaries ADM0 vector boundary coordinates
├── Screenshots/                   # Web application interface previews
│   ├── homepage.png               # Interactive Station Map (Normal Pins View)
│   ├── homepage_gradient.png      # Invariant Solar Radiation Gradient View
│   ├── Features_comparison.png    # Monthly resource feature trends & scale normalization
│   ├── Station_comparison.png     # Multi-station comparison & GHI summary statistics
│   └── Predictions.png            # Real-time GHI prediction interface (FT-Transformer)
│
├── dataset.csv                    # Saudi Arabia solar radiation dataset (1,649 records)
├── requirements.txt               # Production Python dependencies (Flask, PyTorch, sklearn)
├── requirements-dev.txt           # Development dependencies (MLflow, DVC, scipy, pytest)
├── Dockerfile                     # Optimized production container (Python 3.11-slim, CPU PyTorch)
├── render.yaml                    # Render.com Infrastructure-as-Code (IaC) deployment spec
├── .dockerignore                  # Docker build exclusions
├── .github/workflows/
│   ├── ci.yml                     # Continuous Integration & MLflow promotion gate
│   └── cd.yml                     # Continuous Deployment (Docker build & webhook)
│
├── Contribute.md                  # Contributing guidelines
├── Learn.md                       # Learning guide for newcomers
├── LICENSE                        # MIT License
└── README.md                      # This document
```

---

## 🚀 Quickstart Guide

### 1. Clone & Install

```bash
# Clone the repository
git clone https://github.com/Aamir-Sarang31/Solar-Radiation-Prediction-using-Saudi-Arabia-Dataset.git
cd Solar-Radiation-Prediction-using-Saudi-Arabia-Dataset

# Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install all dependencies (production + training/MLOps tools)
pip install -r requirements-dev.txt
```

### 2. Train Models

```bash
# Train all 11 models (classical + deep learning) with 10-Fold CV
python src/train.py --model all --epochs 25 --cv kfold

# Train only the FT-Transformer
python src/train.py --model transformer --epochs 25 --cv kfold

# Train with 43-Fold Leave-One-Station-Out CV
python src/train.py --model transformer --epochs 25 --cv loso

# Train with tuned hyperparameters
python src/train.py --model transformer --config configs/transformer_best.json --cv kfold
```

### 3. Run Hyperparameter Tuning

```bash
# Tune FT-Transformer (30 iterations, 10-Fold CV)
python src/tune.py --model transformer --trials 30 --cv-folds 10

# Tune SVR with RandomizedSearchCV (50 iterations)
python src/tune.py --model svr --trials 50

# Tune all supported models
python src/tune.py --model hgb --trials 50
python src/tune.py --model xgb --trials 50
```

### 4. Run Tests & Promotion Gate

```bash
# Run complete test suite (41 tests)
python -m pytest tests/ -v

# Run MLflow Model Registry Promotion Gate
python src/evaluate.py --gate --model transformer \
    --checkpoint model/transformer_model.pt \
    --scaler model/transformer_scaler.pkl
```

### 5. Launch Web Application

**Option A: Instant Live Web App (Render Cloud)**
- Access the live application directly in your browser: [https://solar-radiation-prediction-uw40.onrender.com](https://solar-radiation-prediction-uw40.onrender.com)

**Option B: Local Development (Flask)**
```bash
python app.py
# Open http://localhost:5000
```

**Option C: Containerized Deployment (Docker)**
```bash
# Build production Docker image
docker build -t solar-radiation-prediction:latest .

# Run container locally
docker run -d -p 5000:5000 --name solar-app solar-radiation-prediction:latest
# Open http://localhost:5000
```

### 6. View MLflow Dashboard

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://127.0.0.1:5000
```

---

## 🌐 Web Application & Deployment

The repository includes a production-ready, interactive web dashboard and REST API built with **Flask**, **Leaflet.js**, **Plotly.js**, and **Bootstrap 5**, fully containerized via **Docker** and deployed live on **Render.com**:

🔗 **Live Web Application:** [https://solar-radiation-prediction-uw40.onrender.com](https://solar-radiation-prediction-uw40.onrender.com)

### 📸 Application Interface Showcase

| Interactive Station Map (Pins View) | Invariant Solar Gradient View |
| :---: | :---: |
| ![Homepage & Map View](Screenshots/homepage.png) | ![Gradient Heatmap View](Screenshots/homepage_gradient.png) |
| *Interactive 43-station map (CARTO Positron) with station inspector & metadata* | *Continuous spatial solar irradiance gradient heatmap overlay* |

| Monthly Meteorological Feature Trends | Multi-Station Comparison & GHI Statistics |
| :---: | :---: |
| ![Monthly Resource Data](Screenshots/Features_comparison.png) | ![Station Comparison](Screenshots/Station_comparison.png) |
| *Per-station multi-parameter time series with dual-scale visualization (Standard vs Normalized)* | *Multi-station comparison across 2017–2021 with dynamic scale normalization & summary stats* |

| Real-Time GHI Prediction Interface |
| :---: |
| ![GHI Prediction](Screenshots/Predictions.png) |
| *Live 21-feature GHI inference with range indicators, dataset defaults, and champion FT-Transformer* |

---

### ✨ Key Web Application Features

1. **🌙 Bright Neon Dark Mode UI (Default) & Dynamic Theme Switch**
   - **Primary Default Theme**: High-contrast, state-of-the-art dark theme utilizing a radial slate-black background (`#10192e` to `#080c14`), glowing neon cyan (`#00F0FF`) headings and active tabs, electric magenta (`#FF00E5`) accents, and glowing neon green (`#00FF9D`) prediction outputs.
   - **Interactive Theme Switch**: Pill-shaped toggle with ☀️ Sun and 🌙 Moon icons, active status badge positioned dynamically (`Solar Light` on the left before the sun icon, `Neon Dark` on the right after the moon icon), clickable emoji icons, and automatic `localStorage` preference persistence.
   - **Full Component Retheming**: Translucent glassmorphic cards, themed Select2 dropdowns, illuminated data tables, and high-visibility range indicators.

2. **🗺️ Precision Geographic Station Map & Solar Gradient**
   - **High-Precision Vector Clipping**: Uses official 3,174-point **geoBoundaries (ADM0)** vector coordinates covering both mainland Saudi Arabia and the Farasan Islands, hardware-clipped via 2D Canvas `ctx.clip()` with zero raster blur or international border bleeding.
   - **Dynamic GHI Station Beacons**: In **Gradient View**, the 43 weather station pins transform into crisp, white-bordered circular beacons with true GHI color coding (green for lower irradiance, amber/yellow for mid-range, deep red for highest) with zero unwanted color haloing.
   - **Dual Watermark-Free Basemaps**: Dynamically swaps between **Esri World Dark Gray Base + Reference** in Neon Dark Mode and **Esri World Light Gray Base + Reference** in Solar Light Mode, ensuring clean English geography without third-party watermarks.
   - **Station Inspector**: Selecting any station pans smoothly and retrieves GPS coordinates, elevation, observation period, and historical averages.

3. **📊 Dynamic Themed Time Series & Scale Normalization**
   - Multi-parameter time-series plotting powered by Plotly.js with instant client-side in-memory caching and debounced parameter selection.
   - **Theme-Aware Charts**: Automatically switches between dark transparent backgrounds with vibrant neon traces (`#00F0FF`, `#FF00E5`, `#39FF14`, `#FFD600`) and the clean light theme with solar cobalt colors upon toggling the theme switch.
   - Filter by station and year (2017–2021 or All Years aggregate).
   - **Dual Scale Modes**:
     - *Standard Values*: Visualizes exact physical quantities in native units (°C, m/s, Wh/m², hPa).
     - *Normalized (0–100%)*: Min-max normalizes disparate metrics onto a unified vertical axis to clearly examine cross-parameter correlations.

4. **📈 Multi-Station Comparison & Statistical Benchmarks**
   - Compare multiple stations side-by-side across any meteorological parameter with dynamic splines and sub-50ms instant response via client-side caching and vectorized backend aggregation.
   - **Automated GHI Summary Statistics Table**: Dynamically calculates Mean, Minimum, Maximum, and Standard Deviation GHI for selected stations with themed table styling.

5. **⚡ Real-Time GHI Prediction Engine**
   - Accepts 21 meteorological features (temperatures, wind dynamics, diffuse/direct irradiance components, humidity, barometric pressure, and measurement uncertainties).
   - **Real-Time Input Validation**: Displays valid operating ranges derived from national empirical observations with high-contrast text.
   - **"Load Default Values" Button**: Electric magenta action button for one-click autofill with national dataset means.
   - **Glowing Prediction Result Box**: Displays predicted GHI with neon green glassmorphic banner and illuminated metrics.
   - Powered by the production champion **FT-Transformer** model (`R² = 0.9919`, `MAE = 81.10 Wh/m²`), providing sub-millisecond inference.

---

### 🚢 Deployment Architectures & Guides

#### 1. Containerized Deployment (Docker)

The application includes an optimized, production-grade `Dockerfile` using `python:3.11-slim` with CPU-only PyTorch wheels to keep the image lightweight, secure, and fast to build.

```bash
# 1. Build the production Docker image
docker build -t solar-radiation-prediction:latest .

# 2. Run the container locally (mapped to port 5000)
docker run -d -p 5000:5000 --name solar-app solar-radiation-prediction:latest

# 3. Access the web interface
# Open http://localhost:5000 in your browser
```

**Container Specifications:**
- **Base Image:** `python:3.11-slim`
- **WSGI Concurrency:** Gunicorn with 4 threads and dynamic port allocation (`${PORT:-5000}`)
- **Built-in Healthcheck:** Periodically executes `curl -f http://localhost:5000/` every 30s
- **Security & Caching:** Non-buffered stdout, bytecode caching disabled, build dependencies cleaned up

#### 2. Cloud One-Click Deployment (Render Blueprint)

The repository provides a native `render.yaml` Infrastructure-as-Code (IaC) configuration for seamless deployment to [Render.com](https://render.com):

```yaml
services:
  - type: web
    name: solar-radiation-prediction
    env: docker
    plan: free
    autoDeploy: true
    healthCheckPath: /
```

**How to Deploy to Render:**
1. Push this repository to your GitHub account.
2. Log in to [Render](https://dashboard.render.com/) and click **New +** → **Blueprint**.
3. Select your repository; Render will automatically detect `render.yaml` and provision the containerized web service.
4. Continuous Deployment: Any subsequent push to `main`/`master` triggers an automatic zero-downtime rebuild and deploy.

> 🔗 **Production Deployment URL:** [https://solar-radiation-prediction-uw40.onrender.com](https://solar-radiation-prediction-uw40.onrender.com)

#### 3. Production WSGI Server (Gunicorn / Waitress)

For running in virtual environments or bare-metal servers:

```bash
# Linux / macOS (Gunicorn with 4 worker threads)
gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 4 --timeout 120 wsgi:app

# Windows (Waitress production WSGI server)
pip install waitress
waitress-serve --port=8000 wsgi:app
```

#### 4. Automated Continuous Deployment (GitHub Actions)

The `.github/workflows/cd.yml` workflow triggers on every push to `main`/`master`:
1. Tests the container build using Docker Buildx on Ubuntu runners.
2. If `RENDER_DEPLOY_HOOK_URL` secret is configured in GitHub repository settings, triggers Render's instant deployment webhook.

---

### 🔌 REST API Endpoints

The Flask backend exposes a clean JSON REST API:

| Endpoint | Method | Description | Parameters / Payload |
|---|:---:|---|---|
| `/` | `GET` | Web UI Dashboard | None |
| `/models` | `GET` | Available model identifiers | None |
| `/map-data` | `GET` | Geo-coordinates & average GHI for all 43 stations | None |
| `/get-station-names` | `GET` | List of all monitored station names | None |
| `/station-details` | `GET` | Station metadata & monthly aggregated trends | `?station=<name>&year=<year>` |
| `/data-analysis` | `GET` | Summary statistics (mean, min, max, std) for all features | None |
| `/station-comparison` | `POST` | Multi-station time series & summary stats | `{"stations": [...], "params": [...], "year": "2020"}` |
| `/predict` | `POST` | Execute real-time GHI inference | JSON payload with 21 features & optional `model_type` |
| `/get-average-values` | `GET` | National mean values for all 21 input features | None |
| `/get-ghi-thresholds` | `GET` | GHI 25th and 75th percentile classification bounds | None |

#### Sample Prediction Request (`POST /predict`):

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Air Temperature (C°)": 26.73,
    "Air Temperature Uncertainty (C°)": 0.50,
    "Wind Direction at 3m (°N)": 168.01,
    "Wind Direction at 3m Uncertainty (°N)": 3.54,
    "Wind Speed at 3m (m/s)": 2.60,
    "Wind Speed at 3m Uncertainty (m/s)": 0.06,
    "Wind Speed at 3m (std dev) (m/s)": 1.53,
    "DHI (Wh/m2)": 2166.14,
    "DHI Uncertainty (Wh/m2)": 387.05,
    "Standard Deviation DHI (Wh/m2)": 593.40,
    "DNI (Wh/m2)": 5587.29,
    "DNI Uncertainty (Wh/m2)": 965.23,
    "Standard Deviation DNI (Wh/m2)": 1747.42,
    "GHI Uncertainty (Wh/m2)": 828.38,
    "Standard Deviation GHI (Wh/m2)": 657.29,
    "Peak Wind Speed at 3m (m/s)": 14.17,
    "Peak Wind Speed at 3m Uncertainty (m/s)": 0.08,
    "Relative Humidity (%)": 39.08,
    "Relative Humidity Uncertainty (%)": 3.01,
    "Barometric Pressure (mB (hPa equiv))": 953.54,
    "Barometric Pressure Uncertainty (mB (hPa equiv))": 4.79,
    "model_type": "production"
  }'
```

**Response:**
```json
{
  "model_used": "production",
  "prediction": 5906.94,
  "unit": "Wh/m2"
}
```

---

## 📋 Dataset

| Property | Value |
|---|---|
| **Source** | King Fahd University of Petroleum & Minerals (KFUPM) Solar Radiation Monitoring Network |
| **Coverage** | 43 weather stations across Saudi Arabia |
| **Period** | 2017–2021 |
| **Records** | 1,649 monthly aggregated observations |
| **Target Variable** | Global Horizontal Irradiance (GHI) in Wh/m² |
| **Features (21)** | Temperature (avg/max/min), Humidity (avg/max/min), Pressure, Wind Speed/Direction, DNI, DHI, Sunshine Duration, Rainfall, Latitude, Longitude, Elevation, and temporal features |

---

## 🔄 CI/CD Workflows

### `.github/workflows/ci.yml` — Continuous Integration

Triggers on every push and pull request to `main`/`master`:

1. **Checkout** → Set up Python 3.11
2. **Install Dependencies** → PyTorch (CPU) + `requirements-dev.txt` (MLflow, DVC, scipy, pytest, flake8)
3. **Code Quality** → Flake8 linting (syntax errors, undefined names)
4. **Test Suite** → `pytest tests/ -v` (47 tests)
5. **Smoke Training** → 3-epoch FT-Transformer training with MLflow tracking
6. **Promotion Gate** → MLflow Registry quality threshold verification (`--dry-run` on PRs, full export on `main`)

### `.github/workflows/cd.yml` — Continuous Deployment

Triggers automatically on push to `main`/`master`:

1. **Docker Buildx** → Builds containerized production image using `Dockerfile`
2. **Deployment Trigger** → Invokes automated webhook for production hosting (e.g., Render, Railway)

---

## 🧪 Testing

The project includes **47 automated tests** across 6 test modules:

| Module | Tests | Coverage |
|---|:---:|---|
| `test_api.py` | 7 | Flask route responses, prediction endpoints, multi-model inference |
| `test_models.py` | 7 | Forward pass, backward pass gradients, single-sample inference for all DL models |
| `test_preprocessing.py` | 5 | Feature list completeness, dataset loading, 10-Fold/LOSO splits, scaling |
| `test_registry_gate.py` | 3 | Promotion thresholds, acceptance criteria, MLflow retrieval |
| `test_results_export.py` | 6 | Research CSV results export, schema validation, 95% CIs, 10-Fold & LOSO significance (p-values) |
| `test_tuning.py` | 19 | RandomizedSearchCV distributions, sampling (8 models), device detection, classical/DL smoke tuning |

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_tuning.py -v
```

---

## 👥 Authors & Acknowledgements

- **Aamir Sarang**
- **Nishant Narudkar**
- **Maitreya Pawar**
- **Vatsal Parmar**

We express our sincere gratitude to **Mr. Pramod H. Kachare** and **Mr. Sandeep Sangle** for their valuable guidance and mentorship throughout this research project.

---

## 📄 License

This project is open-source under the **MIT License**. See [LICENSE](LICENSE) for details.
