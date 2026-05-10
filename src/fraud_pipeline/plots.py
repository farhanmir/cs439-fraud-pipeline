from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

from .config import PipelineConfig
from .evaluate import ModelEvaluation

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    sns = None
    HAS_SEABORN = False

try:
    import shap
    HAS_SHAP = True
except ImportError as exc:  # pragma: no cover
    shap = None
    HAS_SHAP = False


def clean_feature_names(feature_names: list[str]) -> list[str]:
    return [
        name.removeprefix("num__").removeprefix("cat__")
        for name in feature_names
    ]


def save_confusion_matrix_plot(evaluation: ModelEvaluation, output_path: Path) -> None:
    matrix = np.asarray(evaluation.confusion_matrix)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", color="black")
    ax.set_title(f"{evaluation.name} Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["Non-Fraud", "Fraud"])
    ax.set_yticks([0, 1], labels=["Non-Fraud", "Fraud"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pr_curve_plot(
    baseline_eval: ModelEvaluation,
    hybrid_eval: ModelEvaluation,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        baseline_eval.pr_curve_recall,
        baseline_eval.pr_curve_precision,
        label=f"{baseline_eval.name} (AP={baseline_eval.pr_auc:.3f})",
        linewidth=2,
    )
    ax.plot(
        hybrid_eval.pr_curve_recall,
        hybrid_eval.pr_curve_precision,
        label=f"{hybrid_eval.name} (AP={hybrid_eval.pr_auc:.3f})",
        linewidth=2,
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metric_comparison_plot(
    baseline_eval: ModelEvaluation,
    hybrid_eval: ModelEvaluation,
    output_path: Path,
) -> None:
    metric_names = ["precision", "recall", "f1", "pr_auc", "roc_auc"]
    baseline_values = [getattr(baseline_eval, metric) for metric in metric_names]
    hybrid_values = [getattr(hybrid_eval, metric) for metric in metric_names]

    x = np.arange(len(metric_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, baseline_values, width=width, label="Baseline")
    ax.bar(x + width / 2, hybrid_values, width=width, label="Hybrid")
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Metric Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pca_projection_plot(
    X_train_dense: np.ndarray,
    X_test_dense: np.ndarray,
    y_test: np.ndarray,
    anomaly_flags_test: np.ndarray,
    config: PipelineConfig,
    output_path: Path,
) -> None:
    if len(X_test_dense) > config.pca_sample_size:
        rng = np.random.default_rng(config.random_state)
        chosen_idx = rng.choice(len(X_test_dense), size=config.pca_sample_size, replace=False)
        X_test_plot = X_test_dense[chosen_idx]
        y_test_plot = y_test[chosen_idx]
        anomaly_plot = anomaly_flags_test[chosen_idx]
    else:
        X_test_plot = X_test_dense
        y_test_plot = y_test
        anomaly_plot = anomaly_flags_test

    pca = PCA(n_components=2, random_state=config.random_state)
    pca.fit(X_train_dense)
    coords = pca.transform(X_test_plot)

    plot_df = pd.DataFrame(
        {
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "Fraud Label": np.where(y_test_plot == 1, "Fraud", "Non-Fraud"),
            "IsolationForest Flag": np.where(anomaly_plot == 1, "Anomaly", "Normal"),
        }
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    color_map = {"Non-Fraud": "#1f77b4", "Fraud": "#d62728"}
    marker_map = {"Normal": "o", "Anomaly": "X"}
    for fraud_label in ["Non-Fraud", "Fraud"]:
        for anomaly_status in ["Normal", "Anomaly"]:
            subset = plot_df[
                (plot_df["Fraud Label"] == fraud_label)
                & (plot_df["IsolationForest Flag"] == anomaly_status)
            ]
            if not subset.empty:
                ax.scatter(
                    subset["pc1"],
                    subset["pc2"],
                    c=color_map[fraud_label],
                    marker=marker_map[anomaly_status],
                    alpha=0.7,
                    s=30,
                    label=f"{fraud_label} / {anomaly_status}",
                )
    ax.set_title("PCA Projection of Test Data")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_top_feature_importance_plot(
    feature_importance_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_df = feature_importance_df.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#2b6cb0")
    ax.set_title("Top XGBoost Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_error_analysis_plot(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
) -> None:
    error_df = pd.DataFrame(
        {
            "true_label": y_true,
            "predicted_probability": y_score,
        }
    )
    error_df["outcome"] = np.select(
        [
            (error_df["true_label"] == 1) & (error_df["predicted_probability"] < 0.5),
            (error_df["true_label"] == 0) & (error_df["predicted_probability"] >= 0.5),
            (error_df["true_label"] == 1) & (error_df["predicted_probability"] >= 0.5),
        ],
        ["False Negative", "False Positive", "True Positive"],
        default="True Negative",
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    categories = ["True Negative", "False Positive", "False Negative", "True Positive"]
    colors = {
        "True Negative": "#1f77b4",
        "False Positive": "#ff7f0e",
        "False Negative": "#d62728",
        "True Positive": "#2ca02c",
    }
    for category in categories:
        subset = error_df[error_df["outcome"] == category]
        if not subset.empty:
            ax.hist(
                subset["predicted_probability"],
                bins=30,
                alpha=0.55,
                label=category,
                color=colors[category],
            )
    ax.axvline(0.5, linestyle="--", color="black", linewidth=1)
    ax.set_title("Hybrid Model Error Analysis by Predicted Probability")
    ax.set_xlabel("Predicted Fraud Probability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_shap_summary_plot(
    model: object,
    X_test_augmented: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    random_state: int,
    sample_size: int,
    max_display: int = 20,
) -> pd.DataFrame:
    shap_feature_names = [
        *clean_feature_names(feature_names),
        "anomaly_score",
        "anomaly_flag",
    ]
    if len(X_test_augmented) > sample_size:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(len(X_test_augmented), size=sample_size, replace=False)
        X_test_plot = X_test_augmented[sample_idx]
    else:
        X_test_plot = X_test_augmented

    if HAS_SHAP:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_plot)

        plt.figure(figsize=(8, 6))
        shap.summary_plot(
            shap_values,
            features=X_test_plot,
            feature_names=shap_feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        return pd.DataFrame(
            {
                "feature": shap_feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        ).sort_values("mean_abs_shap", ascending=False)

    # Fallback: permutation-importance bar chart saved under the SHAP artifact path.
    sample_cap = min(len(X_test_plot), 1000)
    X_perm = X_test_plot[:sample_cap]
    # Synthetic labels are not used here; caller should interpret this as a fallback explainability artifact.
    raise RuntimeError("Permutation importance fallback requires y_test; use save_fallback_explainability_plot instead.")


def save_fallback_explainability_plot(
    model: object,
    X_test_augmented: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    random_state: int,
    sample_size: int,
    max_display: int = 20,
) -> pd.DataFrame:
    shap_feature_names = [
        *clean_feature_names(feature_names),
        "anomaly_score",
        "anomaly_flag",
    ]
    if len(X_test_augmented) > sample_size:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(len(X_test_augmented), size=sample_size, replace=False)
        X_test_plot = X_test_augmented[sample_idx]
        y_test_plot = y_test[sample_idx]
    else:
        X_test_plot = X_test_augmented
        y_test_plot = y_test

    result = permutation_importance(
        model,
        X_test_plot,
        y_test_plot,
        scoring="average_precision",
        n_repeats=5,
        random_state=random_state,
        n_jobs=1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": shap_feature_names,
            "mean_abs_shap": result.importances_mean,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    plot_df = importance_df.head(max_display).sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#2b6cb0")
    ax.set_title("Permutation Importance Summary")
    ax.set_xlabel("Mean Importance Drop (AP)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return importance_df
