"""generate_figures.py

Loads saved model artifacts and test-set predictions to produce 5
publication-quality figures for the CS439 NeurIPS report.

Run:
    python generate_figures.py

No final classifiers are retrained.  XGBoost and Logistic Regression are
loaded from artifacts/models/.  Data preparation and the IsolationForest
feature extractor are re-run deterministically (random_state=42) only to
reconstruct the processed test-feature matrix needed for SHAP and PCA.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from fraud_pipeline.config import PipelineConfig
from fraud_pipeline.data import prepare_data
from fraud_pipeline.models import build_hybrid_features
from fraud_pipeline.utils import to_dense

# ── Paths ──────────────────────────────────────────────────────────────────
ARTIFACTS   = ROOT / "artifacts"
MODELS_DIR  = ARTIFACTS / "models"
PREDS_DIR   = ARTIFACTS / "predictions"
TABLES_DIR  = ARTIFACTS / "tables"
FIGURES_DIR = ARTIFACTS / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SHAP_SAMPLE  = 2000
PCA_SAMPLE   = 5000

plt.style.use("seaborn-v0_8-whitegrid")

# ── Load saved predictions (figures 1 & 2) ─────────────────────────────────
baseline_preds = pd.read_csv(PREDS_DIR / "baseline_test_predictions.csv")
hybrid_preds   = pd.read_csv(PREDS_DIR / "hybrid_test_predictions.csv")

y_true_base  = baseline_preds["true_label"].to_numpy()
y_score_base = baseline_preds["predicted_probability"].to_numpy()
y_pred_base  = baseline_preds["predicted_label"].to_numpy()

y_true_hyb   = hybrid_preds["true_label"].to_numpy()
y_score_hyb  = hybrid_preds["predicted_probability"].to_numpy()
y_pred_hyb   = hybrid_preds["predicted_label"].to_numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Precision-Recall Curves
# ══════════════════════════════════════════════════════════════════════════════
prec_base, rec_base, _ = precision_recall_curve(y_true_base, y_score_base)
ap_base = average_precision_score(y_true_base, y_score_base)

prec_hyb, rec_hyb, _ = precision_recall_curve(y_true_hyb, y_score_hyb)
ap_hyb = average_precision_score(y_true_hyb, y_score_hyb)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec_base, prec_base, linewidth=2, color="#4878CF",
        label=f"Logistic Regression Baseline  (AP = {ap_base:.3f})")
ax.plot(rec_hyb,  prec_hyb,  linewidth=2, color="#D65F5F",
        label=f"Hybrid IsolationForest + XGBoost  (AP = {ap_hyb:.3f})")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "pr_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: artifacts/figures/pr_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Side-by-side Confusion Matrices
# ══════════════════════════════════════════════════════════════════════════════
cm_base = confusion_matrix(y_true_base, y_pred_base)
cm_hyb  = confusion_matrix(y_true_hyb,  y_pred_hyb)
tick_labels = ["Non-Fraud", "Fraud"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, cm, title in zip(
    axes,
    [cm_base, cm_hyb],
    ["Logistic Regression Baseline", "Hybrid IsolationForest + XGBoost"],
):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for (i, j), val in np.ndenumerate(cm):
        text_color = "white" if val > cm.max() / 2 else "black"
        ax.text(j, i, f"{val:,}", ha="center", va="center",
                fontsize=13, fontweight="bold", color=text_color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: artifacts/figures/confusion_matrices.png")


# ══════════════════════════════════════════════════════════════════════════════
# Reconstruct processed test features (SHAP + PCA need the feature matrix)
# — prepare_data and build_hybrid_features are deterministic with random_state=42
# ══════════════════════════════════════════════════════════════════════════════
print("Preparing features for SHAP and PCA (deterministic, no retraining)...")
config = PipelineConfig(
    data_path=ROOT / "creditcard.csv",
    output_dir=ARTIFACTS,
    target_column="Class",
    drop_columns=["Time"],
    test_size=0.2,
    random_state=RANDOM_STATE,
)
prepared        = prepare_data(config)
_, hybrid_feats = build_hybrid_features(
    prepared.X_train_processed,
    prepared.X_test_processed,
    config,
)
hybrid_model = joblib.load(MODELS_DIR / "hybrid_xgbclassifier.joblib")
print("Features and model ready.")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — SHAP Summary Plot
# ══════════════════════════════════════════════════════════════════════════════
feature_names = [*prepared.feature_names, "anomaly_score", "anomaly_flag"]
X_aug = hybrid_feats.x_test_augmented

rng = np.random.default_rng(RANDOM_STATE)
if len(X_aug) > SHAP_SAMPLE:
    shap_idx = rng.choice(len(X_aug), size=SHAP_SAMPLE, replace=False)
    X_shap   = X_aug[shap_idx]
else:
    X_shap = X_aug

if HAS_SHAP:
    explainer   = shap.TreeExplainer(hybrid_model)
    shap_values = explainer.shap_values(X_shap)
    plt.figure(figsize=(8, 7))
    shap.summary_plot(
        shap_values,
        features=X_shap,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.title("SHAP Feature Contributions — Hybrid XGBoost",
              fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: artifacts/figures/shap_summary.png")
else:
    print("shap not installed — skipping shap_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — PCA Projection of Test Set
# colour = true label   |   marker = IsolationForest anomaly flag
# ══════════════════════════════════════════════════════════════════════════════
X_train_dense = to_dense(prepared.X_train_processed)
X_test_dense  = to_dense(prepared.X_test_processed)
y_test        = prepared.y_test.to_numpy()
anom_flags    = hybrid_feats.anomaly_flags_test

if len(X_test_dense) > PCA_SAMPLE:
    pca_idx    = rng.choice(len(X_test_dense), size=PCA_SAMPLE, replace=False)
    X_plot     = X_test_dense[pca_idx]
    y_plot     = y_test[pca_idx]
    flags_plot = anom_flags[pca_idx]
else:
    X_plot, y_plot, flags_plot = X_test_dense, y_test, anom_flags

pca    = PCA(n_components=2, random_state=RANDOM_STATE)
pca.fit(X_train_dense)
coords = pca.transform(X_plot)

COLOR_MAP  = {0: "#4878CF", 1: "#D65F5F"}
MARKER_MAP = {0: "o",       1: "X"}
SIZE_MAP   = {0: 18,        1: 55}
LABEL_MAP  = {
    (0, 0): "Non-Fraud / Normal",
    (0, 1): "Non-Fraud / Anomaly (IF)",
    (1, 0): "Fraud / Normal",
    (1, 1): "Fraud / Anomaly (IF)",
}

fig, ax = plt.subplots(figsize=(7, 5))
for fraud in [0, 1]:
    for flag in [0, 1]:
        mask = (y_plot == fraud) & (flags_plot == flag)
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=COLOR_MAP[fraud], marker=MARKER_MAP[flag],
            alpha=0.55, s=SIZE_MAP[fraud],
            linewidths=0.4,
            label=LABEL_MAP[(fraud, flag)],
        )
ax.set_xlabel("Principal Component 1", fontsize=11)
ax.set_ylabel("Principal Component 2", fontsize=11)
ax.set_title(
    "PCA Projection of Test Set\n(colour = label, marker = IsolationForest flag)",
    fontsize=12, fontweight="bold",
)
ax.legend(fontsize=8, markerscale=1.4)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "pca_projection.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: artifacts/figures/pca_projection.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Top-15 XGBoost Feature Importances (horizontal bar)
# ══════════════════════════════════════════════════════════════════════════════
fi_df = pd.read_csv(TABLES_DIR / "hybrid_feature_importance.csv")
top15 = fi_df.head(15).copy()
top15["feature"] = top15["feature"].str.replace(r"^(num__|cat__)", "", regex=True)
top15 = top15.sort_values("importance", ascending=True)

colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(top15)))
fig, ax = plt.subplots(figsize=(7, 6))
bars = ax.barh(top15["feature"], top15["importance"],
               color=colors, edgecolor="white", linewidth=0.5)
offset = top15["importance"].max() * 0.01
for bar, val in zip(bars, top15["importance"]):
    ax.text(bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8)
ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
ax.set_title("Top 15 XGBoost Feature Importances", fontsize=13, fontweight="bold")
ax.set_xlim(0, top15["importance"].max() * 1.20)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: artifacts/figures/feature_importance.png")

print("\nDone — all figures saved to artifacts/figures/")
