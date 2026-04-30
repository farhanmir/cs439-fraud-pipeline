from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import PipelineConfig
from .utils import to_dense

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError as exc:  # pragma: no cover - import guard for clearer failure
    XGBClassifier = None
    HAS_XGBOOST = False


@dataclass(slots=True)
class HybridFeatures:
    x_train_augmented: np.ndarray
    x_test_augmented: np.ndarray
    anomaly_scores_train: np.ndarray
    anomaly_scores_test: np.ndarray
    anomaly_flags_train: np.ndarray
    anomaly_flags_test: np.ndarray


def build_hybrid_features(
    x_train_processed: object,
    x_test_processed: object,
    config: PipelineConfig,
) -> tuple[IsolationForest, HybridFeatures]:
    detector = IsolationForest(
        contamination=config.anomaly_contamination,
        n_estimators=300,
        random_state=config.random_state,
        n_jobs=1,
    )
    detector.fit(x_train_processed)

    anomaly_scores_train = -detector.score_samples(x_train_processed)
    anomaly_scores_test = -detector.score_samples(x_test_processed)

    anomaly_flags_train = (detector.predict(x_train_processed) == -1).astype(int)
    anomaly_flags_test = (detector.predict(x_test_processed) == -1).astype(int)

    base_train = to_dense(x_train_processed)
    base_test = to_dense(x_test_processed)

    x_train_augmented = np.column_stack(
        [base_train, anomaly_scores_train, anomaly_flags_train]
    )
    x_test_augmented = np.column_stack(
        [base_test, anomaly_scores_test, anomaly_flags_test]
    )

    return detector, HybridFeatures(
        x_train_augmented=x_train_augmented,
        x_test_augmented=x_test_augmented,
        anomaly_scores_train=anomaly_scores_train,
        anomaly_scores_test=anomaly_scores_test,
        anomaly_flags_train=anomaly_flags_train,
        anomaly_flags_test=anomaly_flags_test,
    )


def train_logistic_regression(
    X_train: object, y_train: object, random_state: int
) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_hybrid_xgboost(
    x_train_augmented: np.ndarray,
    y_train: object,
    random_state: int,
) -> object:
    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    scale_pos_weight = negatives / max(positives, 1)

    if HAS_XGBOOST:
        model = XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        )
    else:
        # Fallback for environments without xgboost.
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        )
    model.fit(x_train_augmented, y_train)
    return model


def get_hybrid_model_label(model: object) -> str:
    if HAS_XGBOOST and model.__class__.__name__ == "XGBClassifier":
        return "Hybrid IsolationForest + XGBoost"
    return f"Hybrid IsolationForest + {model.__class__.__name__}"
