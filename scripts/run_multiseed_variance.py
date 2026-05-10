from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fraud_pipeline.config import PipelineConfig
from fraud_pipeline.data import prepare_data
from fraud_pipeline.evaluate import evaluate_binary_classifier
from fraud_pipeline.models import (
    build_hybrid_features,
    get_hybrid_model_label,
    train_hybrid_xgboost,
    train_logistic_regression,
)
from fraud_pipeline.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-seed variance for baseline and hybrid models."
    )
    parser.add_argument("--data", required=True, help="Path to dataset CSV.")
    parser.add_argument(
        "--target",
        default="Class",
        help="Name of the binary target column. Defaults to 'Class'.",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=[],
        help="Optional columns to drop before modeling.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where variance tables will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction for the train/test split.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[0, 1, 2, 3, 42],
        help="Random seeds to run. Defaults to 0 1 2 3 42.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict] = []

    for seed in args.seeds:
        config = PipelineConfig(
            data_path=Path(args.data),
            output_dir=Path(args.output_dir),
            target_column=args.target,
            drop_columns=args.drop_columns,
            test_size=args.test_size,
            random_state=seed,
        )
        prepared = prepare_data(config)
        _, hybrid_features = build_hybrid_features(
            prepared.X_train_processed,
            prepared.X_test_processed,
            config,
        )

        baseline_model = train_logistic_regression(
            prepared.X_train_processed,
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
                "Logistic Regression Baseline",
                baseline_model,
                prepared.X_test_processed,
                prepared.y_test.to_numpy(),
            ),
            evaluate_binary_classifier(
                get_hybrid_model_label(hybrid_model),
                hybrid_model,
                hybrid_features.x_test_augmented,
                prepared.y_test.to_numpy(),
            ),
        ]

        for evaluation in evaluations:
            rows.append(
                {
                    "seed": seed,
                    "model": evaluation.name,
                    "f1": evaluation.f1,
                    "pr_auc": evaluation.pr_auc,
                }
            )

    output_dir = ensure_dir(Path(args.output_dir) / "tables")
    per_seed = pd.DataFrame(rows)
    summary = (
        per_seed.groupby("model")[["f1", "pr_auc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    per_seed.to_csv(output_dir / "multiseed_variance_per_seed.csv", index=False)
    summary.to_csv(output_dir / "multiseed_variance_summary.csv", index=False)

    print(per_seed.to_string(index=False))
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
