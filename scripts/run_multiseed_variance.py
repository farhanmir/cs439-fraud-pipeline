"""Multi-seed variance analysis for all three models.

Usage:
    python scripts/run_multiseed_variance.py \
        --data creditcard.csv \
        --drop-columns Time \
        --seeds 0 1 2 3 42 \
        --output-dir artifacts

Trains Logistic Regression (baseline), XGBoost-only (ablation), and
Hybrid IsolationForest + XGBoost across multiple random seeds and reports
mean +/- std for F1 and PR-AUC. Saves per-seed and summary CSVs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_pipeline.config import PipelineConfig
from fraud_pipeline.data import prepare_data
from fraud_pipeline.evaluate import evaluate_binary_classifier
from fraud_pipeline.models import (
    build_hybrid_features,
    train_hybrid_xgboost,
    train_logistic_regression,
)
from fraud_pipeline.utils import ensure_dir, to_dense


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed variance analysis for all three models."
    )
    parser.add_argument("--data", required=True, help="Path to dataset CSV.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 42],
        help="Random seeds to evaluate over.",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=["Time"],
        help="Columns to drop before modeling.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to save CSV outputs.",
    )
    return parser.parse_args()


MODEL_LABELS = {
    "baseline": "Logistic Regression (Baseline)",
    "ablation": "XGBoost Only (No IF Features)",
    "hybrid": "Hybrid IsolationForest + XGBoost",
}


def run_seed(seed: int, data_path: Path, drop_columns: list[str]) -> dict:
    """Train and evaluate all three models for one seed."""
    cfg = PipelineConfig(
        data_path=data_path,
        drop_columns=drop_columns,
        random_state=seed,
    )
    prepared = prepare_data(cfg)
    _, hybrid_features = build_hybrid_features(
        prepared.X_train_processed,
        prepared.X_test_processed,
        cfg,
    )

    # Baseline
    baseline_model = train_logistic_regression(
        prepared.X_train_processed,
        prepared.y_train,
        seed,
    )
    baseline_eval = evaluate_binary_classifier(
        MODEL_LABELS["baseline"],
        baseline_model,
        prepared.X_test_processed,
        prepared.y_test.to_numpy(),
    )

    # Ablation: XGBoost on 31-feature matrix only (no IsolationForest features)
    ablation_model = train_hybrid_xgboost(
        to_dense(prepared.X_train_processed),
        prepared.y_train,
        seed,
    )
    ablation_eval = evaluate_binary_classifier(
        MODEL_LABELS["ablation"],
        ablation_model,
        to_dense(prepared.X_test_processed),
        prepared.y_test.to_numpy(),
    )

    # Hybrid: XGBoost on 33-feature augmented matrix
    hybrid_model = train_hybrid_xgboost(
        hybrid_features.x_train_augmented,
        prepared.y_train,
        seed,
    )
    hybrid_eval = evaluate_binary_classifier(
        MODEL_LABELS["hybrid"],
        hybrid_model,
        hybrid_features.x_test_augmented,
        prepared.y_test.to_numpy(),
    )

    return {
        "seed": seed,
        "baseline_f1": baseline_eval.f1,
        "baseline_pr_auc": baseline_eval.pr_auc,
        "ablation_f1": ablation_eval.f1,
        "ablation_pr_auc": ablation_eval.pr_auc,
        "hybrid_f1": hybrid_eval.f1,
        "hybrid_pr_auc": hybrid_eval.pr_auc,
        "baseline_tp": baseline_eval.confusion_matrix[1][1],
        "baseline_fp": baseline_eval.confusion_matrix[0][1],
        "baseline_fn": baseline_eval.confusion_matrix[1][0],
        "ablation_tp": ablation_eval.confusion_matrix[1][1],
        "ablation_fp": ablation_eval.confusion_matrix[0][1],
        "ablation_fn": ablation_eval.confusion_matrix[1][0],
        "hybrid_tp": hybrid_eval.confusion_matrix[1][1],
        "hybrid_fp": hybrid_eval.confusion_matrix[0][1],
        "hybrid_fn": hybrid_eval.confusion_matrix[1][0],
    }


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(Path(args.output_dir))
    tables_dir = ensure_dir(outdir / "tables")
    data_path = Path(args.data)

    print(f"Running multi-seed variance analysis over seeds: {args.seeds}")
    print(f"Drop columns: {args.drop_columns}")
    print()

    rows = []
    for seed in args.seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        result = run_seed(seed, data_path, args.drop_columns)
        rows.append(result)
        print(
            f"done  "
            f"[ablation PR-AUC={result['ablation_pr_auc']:.3f}  "
            f"hybrid PR-AUC={result['hybrid_pr_auc']:.3f}]"
        )

    df = pd.DataFrame(rows)

    # Per-seed long-format CSV (one row per model per seed)
    long_rows = []
    for _, row in df.iterrows():
        for model_key, label in MODEL_LABELS.items():
            long_rows.append(
                {
                    "model": label,
                    "seed": int(row["seed"]),
                    "f1": row[f"{model_key}_f1"],
                    "pr_auc": row[f"{model_key}_pr_auc"],
                    "tp": row[f"{model_key}_tp"],
                    "fp": row[f"{model_key}_fp"],
                    "fn": row[f"{model_key}_fn"],
                }
            )
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(tables_dir / "multiseed_variance.csv", index=False)

    # Summary CSV (mean and std across seeds per model)
    summary_rows = []
    for model_key, label in MODEL_LABELS.items():
        f1_vals = df[f"{model_key}_f1"].to_numpy()
        pr_vals = df[f"{model_key}_pr_auc"].to_numpy()
        fp_vals = df[f"{model_key}_fp"].to_numpy()
        summary_rows.append(
            {
                "model": label,
                "f1_mean": float(np.mean(f1_vals)),
                "f1_std": float(np.std(f1_vals)),
                "pr_auc_mean": float(np.mean(pr_vals)),
                "pr_auc_std": float(np.std(pr_vals)),
                "fp_mean": float(np.mean(fp_vals)),
                "fp_std": float(np.std(fp_vals)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tables_dir / "multiseed_summary.csv", index=False)

    # Print formatted results table
    print()
    seeds_str = " ".join(str(s) for s in args.seeds)
    print(f"Multi-seed variance results (seeds: {seeds_str})")
    print()
    col1, col2, col3, col4 = 42, 22, 22, 18
    header = (
        f"{'Model':<{col1}}"
        f"{'F1 mean +/- std':<{col2}}"
        f"{'PR-AUC mean +/- std':<{col3}}"
        f"{'FP mean +/- std':<{col4}}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['model']:<{col1}}"
            f"{row['f1_mean']:.3f} +/- {row['f1_std']:.3f}        "
            f"{row['pr_auc_mean']:.3f} +/- {row['pr_auc_std']:.3f}        "
            f"{row['fp_mean']:.1f} +/- {row['fp_std']:.1f}"
        )

    print()
    print(f"Saved: {tables_dir / 'multiseed_variance.csv'}")
    print(f"Saved: {tables_dir / 'multiseed_summary.csv'}")


if __name__ == "__main__":
    main()
