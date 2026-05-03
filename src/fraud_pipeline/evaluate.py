# Fixed zero-division in precision score
# Fixed zero-division in precision score
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(slots=True)
class ModelEvaluation:
    name: str
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    classification_report: dict
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray
    pr_curve_precision: np.ndarray
    pr_curve_recall: np.ndarray


def threshold_sweep_table(y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
    thresholds = np.linspace(0.05, 0.95, 19)
    rows: list[dict] = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "predicted_positive_rate": float(np.mean(y_pred)),
            }
        )
    return rows


def evaluate_binary_classifier(
    name: str,
    model: object,
    X_test: object,
    y_test: np.ndarray,
) -> ModelEvaluation:
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    return ModelEvaluation(
        name=name,
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        pr_auc=float(average_precision_score(y_test, y_score)),
        roc_auc=float(roc_auc_score(y_test, y_score)),
        confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
        classification_report=classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        y_true=np.asarray(y_test),
        y_pred=np.asarray(y_pred),
        y_score=np.asarray(y_score),
        pr_curve_precision=precision,
        pr_curve_recall=recall,
    )
