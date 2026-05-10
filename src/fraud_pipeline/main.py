# CS439 Fraud Detection Pipeline
# CS439 Fraud Detection Pipeline
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import PipelineConfig
from .data import prepare_data
from .evaluate import evaluate_binary_classifier, threshold_sweep_table
from .models import (
    build_hybrid_features,
    get_hybrid_model_label,
    train_hybrid_xgboost,
    train_logistic_regression,
    train_xgboost,
)
from .plots import (
    clean_feature_names,
    save_error_analysis_plot,
    save_confusion_matrix_plot,
    save_fallback_explainability_plot,
    save_metric_comparison_plot,
    save_pca_projection_plot,
    save_pr_curve_plot,
    save_shap_summary_plot,
    save_top_feature_importance_plot,
)
from .utils import dump_json, ensure_dir, to_dense


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a hybrid fraud detection pipeline with report-ready outputs."
    )
    parser.add_argument("--data", required=True, help="Path to the input CSV dataset.")
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
        help="Directory where metrics, plots, and tables will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction for the train/test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        target_column=args.target,
        drop_columns=args.drop_columns,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    outputs = ensure_dir(config.output_dir)
    plots_dir = ensure_dir(outputs / "plots")
    tables_dir = ensure_dir(outputs / "tables")
    models_dir = ensure_dir(outputs / "models")
    predictions_dir = ensure_dir(outputs / "predictions")

    prepared = prepare_data(config)
    _, hybrid_features = build_hybrid_features(
        prepared.X_train_processed,
        prepared.X_test_processed,
        config,
    )

    baseline_model = train_logistic_regression(
        prepared.X_train_processed,
        prepared.y_train,
        config.random_state,
    )
    xgboost_only_model = train_xgboost(
        prepared.X_train_processed,
        prepared.y_train,
        config.random_state,
    )
    hybrid_model = train_hybrid_xgboost(
        hybrid_features.x_train_augmented,
        prepared.y_train,
        config.random_state,
    )
    hybrid_model_label = get_hybrid_model_label(hybrid_model)

    baseline_eval = evaluate_binary_classifier(
        "Logistic Regression Baseline",
        baseline_model,
        prepared.X_test_processed,
        prepared.y_test.to_numpy(),
    )
    xgboost_only_eval = evaluate_binary_classifier(
        "XGBoost Only (No IF Features)",
        xgboost_only_model,
        prepared.X_test_processed,
        prepared.y_test.to_numpy(),
    )
    hybrid_eval = evaluate_binary_classifier(
        hybrid_model_label,
        hybrid_model,
        hybrid_features.x_test_augmented,
        prepared.y_test.to_numpy(),
    )

    joblib.dump(prepared.preprocessor, models_dir / "preprocessor.joblib")
    joblib.dump(baseline_model, models_dir / "logistic_regression_baseline.joblib")
    joblib.dump(xgboost_only_model, models_dir / "xgboost_only_no_if.joblib")
    hybrid_model_filename = hybrid_model.__class__.__name__.lower()
    joblib.dump(hybrid_model, models_dir / f"hybrid_{hybrid_model_filename}.joblib")

    metrics_table = pd.DataFrame(
        [
            {
                "model": baseline_eval.name,
                "precision": baseline_eval.precision,
                "recall": baseline_eval.recall,
                "f1": baseline_eval.f1,
                "pr_auc": baseline_eval.pr_auc,
                "roc_auc": baseline_eval.roc_auc,
            },
            {
                "model": hybrid_eval.name,
                "precision": hybrid_eval.precision,
                "recall": hybrid_eval.recall,
                "f1": hybrid_eval.f1,
                "pr_auc": hybrid_eval.pr_auc,
                "roc_auc": hybrid_eval.roc_auc,
            },
            {
                "model": xgboost_only_eval.name,
                "precision": xgboost_only_eval.precision,
                "recall": xgboost_only_eval.recall,
                "f1": xgboost_only_eval.f1,
                "pr_auc": xgboost_only_eval.pr_auc,
                "roc_auc": xgboost_only_eval.roc_auc,
            },
        ]
    )
    metrics_table.to_csv(tables_dir / "model_metrics.csv", index=False)

    baseline_predictions = pd.DataFrame(
        {
            "true_label": baseline_eval.y_true,
            "predicted_label": baseline_eval.y_pred,
            "predicted_probability": baseline_eval.y_score,
        }
    )
    hybrid_predictions = pd.DataFrame(
        {
            "true_label": hybrid_eval.y_true,
            "predicted_label": hybrid_eval.y_pred,
            "predicted_probability": hybrid_eval.y_score,
            "anomaly_score": hybrid_features.anomaly_scores_test,
            "anomaly_flag": hybrid_features.anomaly_flags_test,
        }
    )
    xgboost_only_predictions = pd.DataFrame(
        {
            "true_label": xgboost_only_eval.y_true,
            "predicted_label": xgboost_only_eval.y_pred,
            "predicted_probability": xgboost_only_eval.y_score,
        }
    )
    baseline_predictions.to_csv(
        predictions_dir / "baseline_test_predictions.csv",
        index=False,
    )
    xgboost_only_predictions.to_csv(
        predictions_dir / "xgboost_only_no_if_test_predictions.csv",
        index=False,
    )
    hybrid_predictions.to_csv(
        predictions_dir / "hybrid_test_predictions.csv",
        index=False,
    )

    pd.DataFrame(threshold_sweep_table(hybrid_eval.y_true, hybrid_eval.y_score)).to_csv(
        tables_dir / "hybrid_threshold_sweep.csv",
        index=False,
    )

    dataset_summary = {
        **prepared.dataset_summary,
        "train_rows": int(len(prepared.X_train_raw)),
        "test_rows": int(len(prepared.X_test_raw)),
        "train_positive_rate": float(prepared.y_train.mean()),
        "test_positive_rate": float(prepared.y_test.mean()),
        "processed_feature_count": int(len(prepared.feature_names)),
    }
    dump_json(dataset_summary, outputs / "dataset_summary.json")

    feature_importance_df = pd.DataFrame(
        {
            "feature": [
                *clean_feature_names(prepared.feature_names),
                "anomaly_score",
                "anomaly_flag",
            ],
            "importance": hybrid_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance_df.to_csv(
        tables_dir / "hybrid_feature_importance.csv", index=False
    )
    top_feature_importance_df = feature_importance_df.head(
        config.feature_importance_top_k
    )

    dump_json(
        {
            "dataset_summary": dataset_summary,
            "baseline": {
                "precision": baseline_eval.precision,
                "recall": baseline_eval.recall,
                "f1": baseline_eval.f1,
                "pr_auc": baseline_eval.pr_auc,
                "roc_auc": baseline_eval.roc_auc,
                "confusion_matrix": baseline_eval.confusion_matrix,
                "classification_report": baseline_eval.classification_report,
            },
            "xgboost_only_no_if": {
                "precision": xgboost_only_eval.precision,
                "recall": xgboost_only_eval.recall,
                "f1": xgboost_only_eval.f1,
                "pr_auc": xgboost_only_eval.pr_auc,
                "roc_auc": xgboost_only_eval.roc_auc,
                "confusion_matrix": xgboost_only_eval.confusion_matrix,
                "classification_report": xgboost_only_eval.classification_report,
            },
            "hybrid": {
                "precision": hybrid_eval.precision,
                "recall": hybrid_eval.recall,
                "f1": hybrid_eval.f1,
                "pr_auc": hybrid_eval.pr_auc,
                "roc_auc": hybrid_eval.roc_auc,
                "confusion_matrix": hybrid_eval.confusion_matrix,
                "classification_report": hybrid_eval.classification_report,
            },
        },
        outputs / "metrics_summary.json",
    )

    save_confusion_matrix_plot(
        baseline_eval,
        plots_dir / "confusion_matrix_logistic_regression.png",
    )
    save_confusion_matrix_plot(
        hybrid_eval,
        plots_dir / "confusion_matrix_hybrid_xgboost.png",
    )
    save_pr_curve_plot(
        baseline_eval,
        hybrid_eval,
        plots_dir / "precision_recall_curve_comparison.png",
    )
    save_metric_comparison_plot(
        baseline_eval,
        hybrid_eval,
        plots_dir / "metric_comparison.png",
    )
    save_pca_projection_plot(
        X_train_dense=to_dense(prepared.X_train_processed),
        X_test_dense=to_dense(prepared.X_test_processed),
        y_test=prepared.y_test.to_numpy(),
        anomaly_flags_test=hybrid_features.anomaly_flags_test,
        config=config,
        output_path=plots_dir / "pca_projection_test_data.png",
    )
    save_top_feature_importance_plot(
        top_feature_importance_df,
        plots_dir / "hybrid_feature_importance.png",
    )
    save_error_analysis_plot(
        hybrid_eval.y_true,
        hybrid_eval.y_score,
        plots_dir / "hybrid_error_analysis.png",
    )
    try:
        shap_importance_df = save_shap_summary_plot(
            hybrid_model,
            hybrid_features.x_test_augmented,
            prepared.feature_names,
            plots_dir / "shap_summary_hybrid_xgboost.png",
            random_state=config.random_state,
            sample_size=config.shap_sample_size,
        )
    except RuntimeError:
        shap_importance_df = save_fallback_explainability_plot(
            hybrid_model,
            hybrid_features.x_test_augmented,
            prepared.y_test.to_numpy(),
            prepared.feature_names,
            plots_dir / "shap_summary_hybrid_xgboost.png",
            random_state=config.random_state,
            sample_size=config.shap_sample_size,
        )
    shap_importance_df.to_csv(tables_dir / "shap_importance.csv", index=False)


if __name__ == "__main__":
    main()
