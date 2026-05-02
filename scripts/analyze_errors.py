"""Error analysis: isolate false positives and false negatives and plot feature distributions.

Usage:
  python scripts/analyze_errors.py --data creditcard.csv --output-dir artifacts

This script trains the hybrid pipeline (same as run_pipeline) on the train split,
predicts on the test split, then saves per-feature comparison plots for FP/FN/TP/TN.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys

# Ensure local package import works when running from project root; scripts/ is one level deep
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_pipeline.config import PipelineConfig
from fraud_pipeline.data import prepare_data
from fraud_pipeline.models import build_hybrid_features, train_hybrid_xgboost
from fraud_pipeline.utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def plot_feature_distributions(
    df: pd.DataFrame, feature: str, labels: pd.Series, outpath: Path
):
    plt.figure(figsize=(6, 4))
    sns.histplot(
        data=df,
        x=feature,
        hue=labels,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        bins=40,
    )
    plt.title(f"Distribution of {feature} by outcome")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    args = parse_args()
    cfg = PipelineConfig(data_path=Path(args.data), random_state=args.random_state)
    outdir = ensure_dir(Path(args.output_dir))
    plots_dir = ensure_dir(outdir / "plots")

    prepared = prepare_data(cfg)
    _, hybrid = build_hybrid_features(
        prepared.X_train_processed, prepared.X_test_processed, cfg
    )

    model = train_hybrid_xgboost(
        hybrid.x_train_augmented, prepared.y_train.to_numpy(), cfg.random_state
    )

    # Predictions on test
    y_true = prepared.y_test.to_numpy()
    y_pred = model.predict(hybrid.x_test_augmented)
    y_score = model.predict_proba(hybrid.x_test_augmented)[:, 1]

    # Outcome labels: TP, TN, FP, FN
    df_test = prepared.X_test_raw.reset_index(drop=True).copy()
    df_test["true_label"] = y_true
    df_test["pred_label"] = y_pred
    df_test["pred_score"] = y_score

    def outcome_label(row):
        if row["true_label"] == 1 and row["pred_label"] == 1:
            return "TP"
        if row["true_label"] == 0 and row["pred_label"] == 0:
            return "TN"
        if row["true_label"] == 0 and row["pred_label"] == 1:
            return "FP"
        return "FN"

    df_test["outcome"] = df_test.apply(outcome_label, axis=1)

    # Choose numeric features to plot (top 6 by variance)
    numeric_cols = df_test.select_dtypes(include=["number"]).columns.tolist()
    # remove helper columns if present
    for helper in ("true_label", "pred_label", "pred_score"):
        if helper in numeric_cols:
            numeric_cols.remove(helper)

    if not numeric_cols:
        print("No numeric features found for error analysis.")
        return

    variances = df_test[numeric_cols].var().sort_values(ascending=False)
    top_features = variances.head(6).index.tolist()

    for feat in top_features:
        outpath = plots_dir / f"error_analysis_{feat}.png"
        plot_feature_distributions(df_test, feat, df_test["outcome"], outpath)

    # Save summary counts
    counts = df_test["outcome"].value_counts().to_dict()
    pd.Series(counts).to_csv(outdir / "error_outcome_counts.csv")

    print("Saved error analysis plots to:", plots_dir)


if __name__ == "__main__":
    main()
