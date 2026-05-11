"""Quick hyperparameter tuning for the hybrid XGBoost model (scripts/ location).

Usage:
  python scripts/tune_xgboost.py --data creditcard.csv --output-dir artifacts --trials 20

This script loads the project's dataset via the package utilities, builds the hybrid
features (IsolationForest + processed features) and runs a randomized hyperparameter
search for `xgboost.XGBClassifier`. Results are written to `--output-dir`.
"""

from __future__ import annotations

import argparse
import json
import importlib.util
import sys
from pathlib import Path

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

if importlib.util.find_spec("imblearn") is not None:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    HAS_IMBLEARN = True
else:
    HAS_IMBLEARN = False

# Ensure local package import works when running from project root; scripts/ is one level deep
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if importlib.util.find_spec("xgboost") is not None:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
else:
    HAS_XGBOOST = False

from fraud_pipeline.config import PipelineConfig
from fraud_pipeline.data import prepare_data
from fraud_pipeline.models import build_hybrid_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune XGBoost hyperparameters for hybrid model"
    )
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument(
        "--output-dir", default="artifacts", help="Where to save tuning outputs"
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="Number of RandomizedSearch iterations"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        data_path=Path(args.data),
        random_state=args.random_state,
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not HAS_XGBOOST:
        raise RuntimeError(
            "xgboost is required for tuning; install it (pip install xgboost)"
        )

    prepared = prepare_data(cfg)
    _, hybrid = build_hybrid_features(
        prepared.X_train_processed, prepared.X_test_processed, cfg
    )

    X = hybrid.x_train_augmented
    y = prepared.y_train.to_numpy()

    if not HAS_IMBLEARN:
        raise RuntimeError(
            "imblearn is required for SMOTE inside CV; install imbalanced-learn"
        )

    print(
        "\nNOTE: This tuning script applies SMOTE oversampling inside each "
        "CV fold.\nHyperparameters are selected under a SMOTE-augmented class "
        "distribution.\nThe main pipeline (run_pipeline.py) does NOT use "
        "SMOTE -- it uses scale_pos_weight.\nHyperparameters transferred from "
        "this script to models.py may behave differently\nunder the real "
        "training distribution.\n"
    )

    base_clf = XGBClassifier(
        objective="binary:logistic", tree_method="hist", use_label_encoder=False
    )

    # SMOTE is applied inside each CV fold here for hyperparameter search only.
    # The final model in models.py does NOT use SMOTE; it uses scale_pos_weight.
    # Hyperparameters were therefore optimised under a different class distribution
    # than the one used during actual training and inference.
    pipeline = ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=cfg.random_state)),
            ("clf", base_clf),
        ]
    )

    param_dist = {
        "clf__n_estimators": [100, 200, 300, 400, 600],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__subsample": [0.6, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 0.9, 1.0],
        "clf__reg_lambda": [0.5, 1.0, 2.0],
        "clf__min_child_weight": [1, 3, 5],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.random_state)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=args.trials,
        scoring="average_precision",
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=cfg.random_state,
        return_train_score=True,
    )

    print("Starting hyperparameter search (this may take a while)...")
    search.fit(X, y)

    print("Best score (AP):", search.best_score_)
    print("Best params:")
    print(search.best_params_)

    (outdir / "tuning_results.json").write_text(
        json.dumps(
            {
                "best_score": float(search.best_score_),
                "best_params": search.best_params_,
                "smote_used_in_cv": True,
                "note": (
                    "Hyperparameters selected under SMOTE-augmented CV "
                    "distribution. Main pipeline uses scale_pos_weight, "
                    "not SMOTE."
                ),
            },
            indent=2,
        )
    )

    # Save full cv results for transparency
    import pandas as pd

    results_df = pd.DataFrame(search.cv_results_)
    results_df.to_csv(outdir / "tuning_cv_results.csv", index=False)


if __name__ == "__main__":
    main()
