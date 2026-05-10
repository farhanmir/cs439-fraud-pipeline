from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fraud_pipeline.config import PipelineConfig
from fraud_pipeline.data import prepare_data
from fraud_pipeline.models import (
    build_hybrid_features,
    train_logistic_regression,
    train_hybrid_xgboost,
)
from fraud_pipeline.evaluate import evaluate_binary_classifier
from fraud_pipeline.utils import to_dense, ensure_dir


MODEL_ORDER = [
    "Logistic Regression (Baseline)",
    "XGBoost Only (No IF Features)",
    "Hybrid IsolationForest + XGBoost",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-seed variance for baseline, ablation, and hybrid models."
    )
    parser.add_argument("--data", required=True, help="Path to dataset CSV.")
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[0, 1, 2, 3, 42],
        help="Random seeds to run. Defaults to 0 1 2 3 42.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where variance tables will be saved.",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=["Time"],
        help="Optional columns to drop before modeling. Defaults to Time.",
    )
    return parser.parse_args()


def summarize(rows: list[dict]) -> list[dict]:
    summary_rows: list[dict] = []
    for model in MODEL_ORDER:
        model_rows = [row for row in rows if row["model"] == model]
        f1_values = np.array([row["f1"] for row in model_rows], dtype=float)
        pr_auc_values = np.array([row["pr_auc"] for row in model_rows], dtype=float)
        summary_rows.append(
            {
                "model": model,
                "f1_mean": float(np.mean(f1_values)),
                "f1_std": float(np.std(f1_values, ddof=1)),
                "pr_auc_mean": float(np.mean(pr_auc_values)),
                "pr_auc_std": float(np.std(pr_auc_values, ddof=1)),
            }
        )
    return summary_rows


def print_summary(seeds: list[int], summary_rows: list[dict]) -> None:
    print(f"Multi-seed variance results (seeds: {' '.join(map(str, seeds))})")
    print()
    print("Model                              F1 mean ± std     PR-AUC mean ± std")
    print("──────────────────────────────────────────────────────────────────────")
    for row in summary_rows:
        print(
            f"{row['model']:<34} "
            f"{row['f1_mean']:.3f} ± {row['f1_std']:.3f}     "
            f"{row['pr_auc_mean']:.3f} ± {row['pr_auc_std']:.3f}"
        )


def main() -> None:
    args = parse_args()
    rows: list[dict] = []

    for seed in args.seeds:
        config = PipelineConfig(
            data_path=Path(args.data),
            output_dir=Path(args.output_dir),
            drop_columns=args.drop_columns,
            random_state=seed,
        )
        prepared = prepare_data(config)
        _, hybrid_features = build_hybrid_features(
            prepared.X_train_processed,
            prepared.X_test_processed,
            config,
        )

        x_train_processed = to_dense(prepared.X_train_processed)
        x_test_processed = to_dense(prepared.X_test_processed)

        baseline_model = train_logistic_regression(
            prepared.X_train_processed,
            prepared.y_train,
            seed,
        )
        xgboost_only_model = train_hybrid_xgboost(
            x_train_processed,
            prepared.y_train,
            seed,
        )
        hybrid_model = train_hybrid_xgboost(
            hybrid_features.x_train_augmented,
            prepared.y_train,
            seed,
        )

        evaluations = [
            evaluate_binary_classifier(
                "Logistic Regression (Baseline)",
                baseline_model,
                prepared.X_test_processed,
                prepared.y_test.to_numpy(),
            ),
            evaluate_binary_classifier(
                "XGBoost Only (No IF Features)",
                xgboost_only_model,
                x_test_processed,
                prepared.y_test.to_numpy(),
            ),
            evaluate_binary_classifier(
                "Hybrid IsolationForest + XGBoost",
                hybrid_model,
                hybrid_features.x_test_augmented,
                prepared.y_test.to_numpy(),
            ),
        ]

        for evaluation in evaluations:
            rows.append(
                {
                    "model": evaluation.name,
                    "seed": seed,
                    "f1": evaluation.f1,
                    "pr_auc": evaluation.pr_auc,
                }
            )

    summary_rows = summarize(rows)
    tables_dir = ensure_dir(Path(args.output_dir) / "tables")
    pd.DataFrame(rows, columns=["model", "seed", "f1", "pr_auc"]).to_csv(
        tables_dir / "multiseed_variance.csv",
        index=False,
    )
    pd.DataFrame(
        summary_rows,
        columns=["model", "f1_mean", "f1_std", "pr_auc_mean", "pr_auc_std"],
    ).to_csv(tables_dir / "multiseed_summary.csv", index=False)

    print_summary(args.seeds, summary_rows)


if __name__ == "__main__":
    main()
