# Refactored for better performance
# Refactored for better performance
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import PipelineConfig
from .utils import coerce_binary_target


@dataclass(slots=True)
class PreparedData:
    dataset_summary: dict
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    X_train_processed: object
    X_test_processed: object
    feature_names: list[str]


def load_dataset(config: PipelineConfig) -> pd.DataFrame:
    dataset = pd.read_csv(config.data_path)
    if config.target_column not in dataset.columns:
        raise KeyError(
            f"Target column '{config.target_column}' was not found in {config.data_path}."
        )
    return dataset


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [col for col in X_train.columns if col not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def prepare_data(config: PipelineConfig) -> PreparedData:
    dataset = load_dataset(config).drop_duplicates().copy()
    y = coerce_binary_target(dataset[config.target_column])
    X = dataset.drop(
        columns=[config.target_column, *config.drop_columns], errors="ignore"
    )
    dataset_summary = {
        "rows": int(len(dataset)),
        "columns": int(dataset.shape[1]),
        "feature_columns": X.columns.tolist(),
        "positive_class_rate": float(y.mean()),
        "missing_values_total": int(dataset.isna().sum().sum()),
        "duplicate_rows_removed": int(load_dataset(config).duplicated().sum()),
    }

    # Strict: split first, then compute any aggregates on the training split only
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    # Work on copies to avoid modifying original dataset object
    X_train_raw = X_train_raw.copy()
    X_test_raw = X_test_raw.copy()

    # --- Advanced feature engineering (training-data aware) ---
    # 1) Log-transform for skewed `Amount` column if present
    if "Amount" in X_train_raw.columns:
        train_median = (
            float(X_train_raw["Amount"].median())
            if X_train_raw["Amount"].notna().any()
            else 1.0
        )
        global_mean = (
            float(X_train_raw["Amount"].mean())
            if X_train_raw["Amount"].notna().any()
            else 0.0
        )

        X_train_raw["amount_log"] = np.log1p(X_train_raw["Amount"])
        X_test_raw["amount_log"] = np.log1p(X_test_raw["Amount"])

        # Ratio to training median (avoid division by zero)
        denom = train_median if train_median != 0 else 1.0
        X_train_raw["amount_to_median"] = X_train_raw["Amount"] / denom
        X_test_raw["amount_to_median"] = X_test_raw["Amount"] / denom

        # 2) Time-of-day / hour aggregation if `Time` exists (creditcard dataset uses seconds)
        if "Time" in X_train_raw.columns:
            X_train_raw["hour"] = ((X_train_raw["Time"] // 3600) % 24).astype(int)
            X_test_raw["hour"] = ((X_test_raw["Time"] // 3600) % 24).astype(int)

            # Compute per-hour mean amount on TRAIN only and map to both splits
            hour_mean_map = X_train_raw.groupby("hour")["Amount"].mean().to_dict()
            X_train_raw["amount_hour_mean"] = (
                X_train_raw["hour"].map(hour_mean_map).fillna(global_mean)
            )
            X_test_raw["amount_hour_mean"] = (
                X_test_raw["hour"].map(hour_mean_map).fillna(global_mean)
            )

            X_train_raw["amount_to_hour_mean"] = X_train_raw["Amount"] / X_train_raw[
                "amount_hour_mean"
            ].replace({0: 1.0})
            X_test_raw["amount_to_hour_mean"] = X_test_raw["Amount"] / X_test_raw[
                "amount_hour_mean"
            ].replace({0: 1.0})

    # Update dataset summary feature list after feature engineering
    dataset_summary["feature_columns"] = X_train_raw.columns.tolist()

    preprocessor = build_preprocessor(X_train_raw)
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out().tolist()

    return PreparedData(
        dataset_summary=dataset_summary,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        preprocessor=preprocessor,
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
        feature_names=feature_names,
    )
