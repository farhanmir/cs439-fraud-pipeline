# CS439 Final Project Codebase

This repository contains a complete end-to-end tabular machine learning pipeline for a CS439 final project in the "Meaningful Application" style described in the assignment documents. The implementation is built for imbalanced binary classification problems such as fraud detection and emphasizes the rubric requirements that appeared repeatedly in the instructions and example papers:

- strict train/test separation to prevent leakage
- explicit preprocessing for numeric and categorical features
- a hybrid pipeline that combines unsupervised and supervised learning
- baseline comparison
- report-ready evaluation tables and visualizations
- reproducible outputs saved to disk

## Project Design

The code implements two models:

1. `Logistic Regression Baseline`
2. `Hybrid IsolationForest + XGBoost`

The hybrid model first fits `IsolationForest` on the training split only, extracts anomaly scores and anomaly flags, and appends those learned signals as engineered features for the supervised `XGBoost` classifier. This aligns with the "combine techniques in a sophisticated pipeline" direction encouraged by the project instructions.

## Repository Layout

```text
run_pipeline.py
requirements.txt
scripts/                # helper scripts (tuning, download, error analysis)
src/fraud_pipeline/
  config.py
  data.py
  evaluate.py
  main.py
  models.py
  plots.py
  utils.py
```

## Expected Dataset Format

The pipeline expects a local CSV file with:

- one binary target column
- any number of numeric and/or categorical feature columns

Default target column:

```text
Class
```

Accepted binary target representations include:

- `0` / `1`
- `true` / `false`
- `yes` / `no`
- `fraud` / `non-fraud`

If your dataset uses a different target name, pass it with `--target`.

## Installation

**Mac / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the Pipeline

Run the full pipeline (assume your virtual environment is activated):

```bash
python run_pipeline.py --data path/to/dataset.csv --target Class --output-dir artifacts
```

Example with optional dropped columns:

```bash
python run_pipeline.py --data creditcard.csv --target Class --drop-columns Time
```

## What Gets Produced

Running the pipeline creates:

```text
artifacts/
  dataset_summary.json
  metrics_summary.json
  models/
    preprocessor.joblib
    logistic_regression_baseline.joblib
    hybrid_xgboost.joblib
  predictions/
    baseline_test_predictions.csv
    hybrid_test_predictions.csv
  plots/
    confusion_matrix_logistic_regression.png
    confusion_matrix_hybrid_xgboost.png
    hybrid_error_analysis.png
    hybrid_feature_importance.png
    metric_comparison.png
    pca_projection_test_data.png
    precision_recall_curve_comparison.png
    shap_summary_hybrid_xgboost.png
  tables/
    hybrid_feature_importance.csv
    hybrid_threshold_sweep.csv
    model_metrics.csv
    shap_importance.csv
```

## Evaluation Outputs

The saved outputs are designed to feed directly into the report:

- `model_metrics.csv` includes Precision, Recall, F1, PR-AUC, and ROC-AUC
- `precision_recall_curve_comparison.png` compares baseline and hybrid performance
- `pca_projection_test_data.png` gives a 2D PCA visualization
- `shap_summary_hybrid_xgboost.png` provides interpretability
- `hybrid_error_analysis.png` helps support the discussion/error analysis section
- `hybrid_feature_importance.csv` and `shap_importance.csv` support tables in the paper

## Reproducibility Notes

- preprocessing is fit on the training split only
- test data is transformed only after the training preprocessors are fit
- unsupervised anomaly features are learned only from training data
- random seeds are fixed through the pipeline

To ensure deterministic plots and metrics, the pipeline uses a single `random_state` (default `42`) that is passed
to the train/test split, IsolationForest, PCA sampling, SHAP sampling fallback, and model training where applicable.

If you need to reproduce or re-run experiments exactly, pass `--random-state 42` to `run_pipeline.py`.

### Exact environment (reproducible)

Create a fresh virtual environment and install pinned dependencies.

Mac / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then run the pipeline from the project root (assume env is activated):

```bash
python run_pipeline.py --data creditcard.csv --target Class --drop-columns Time --random-state 42
```

Run hyperparameter tuning (deterministic by default):

```bash
python scripts/tune_xgboost.py --data creditcard.csv --output-dir artifacts --trials 20 --random-state 42
```

Run error analysis:

```bash
python scripts/analyze_errors.py --data creditcard.csv --output-dir artifacts --random-state 42
```

## Important Assumption

The assignment documents do not require a single mandatory dataset, so this codebase is dataset-agnostic for tabular binary classification. If you decide to use a specific fraud dataset, point the CLI to that CSV and set the correct target column name.

## Environment Note

If `python` on your machine points to a global interpreter instead of this repository's `venv`, packages like `xgboost`, `seaborn`, or `shap` may appear to be "missing" even when they are installed in the virtual environment. In that case, run the project with:

```powershell
.\venv\Scripts\python.exe run_pipeline.py --data creditcard.csv --target Class --drop-columns Time
```

## Downloading the Dataset (one-liner)

If you don't already have `creditcard.csv` in the project root, a helper script provides a simple way to fetch it.

Automatic attempt (Kaggle credentials required in env):

```bash
python scripts/download_data.py
```

If you prefer the Kaggle CLI you can run:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -f creditcard.csv -p . --unzip
```

## Hyperparameter Tuning (empirical justification)

We include a lightweight randomized search that the graders can run to demonstrate the empirical choice
of XGBoost hyperparameters. It runs RandomizedSearchCV and writes the search results and best parameters
to `artifacts/`.

Run the tuner with:

```bash
python scripts/tune_xgboost.py --data creditcard.csv --output-dir artifacts --trials 20
```

The tuner is deterministic by default (passes `--random-state 42`) so results are reproducible across runs.

