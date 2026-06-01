"""
Data processing pipeline for Bati Bank credit risk modeling.

This module covers:
- Transaction-level datetime extraction
- Customer-level aggregate feature creation
- RFM feature calculation
- K-Means proxy target engineering
- Missing value handling
- Categorical encoding
- Scaling
- WoE/IV-style transformation support
- sklearn Pipeline structure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


REQUIRED_COLUMNS = [
    "TransactionId",
    "CustomerId",
    "Amount",
    "Value",
    "ProductCategory",
    "ChannelId",
    "PricingStrategy",
    "TransactionStartTime",
]


class DataValidationError(ValueError):
    """Raised when required input columns are missing."""


def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate that required columns exist in the dataframe."""
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {missing_columns}"
        )


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day, month, and year from TransactionStartTime."""
    validate_columns(df, ["TransactionStartTime"])

    data = df.copy()
    data["TransactionStartTime"] = pd.to_datetime(
        data["TransactionStartTime"],
        errors="coerce",
    )

    data["transaction_hour"] = data["TransactionStartTime"].dt.hour
    data["transaction_day"] = data["TransactionStartTime"].dt.day
    data["transaction_month"] = data["TransactionStartTime"].dt.month
    data["transaction_year"] = data["TransactionStartTime"].dt.year

    return data


def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer-level aggregate transaction features.

    Required aggregate features:
    - total_transaction_amount
    - avg_transaction_amount
    - transaction_count
    - std_transaction_amount
    """
    validate_columns(
        df,
        ["CustomerId", "TransactionId", "Amount", "Value"],
    )

    customer_features = (
        df.groupby("CustomerId")
        .agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            transaction_count=("TransactionId", "count"),
            std_transaction_amount=("Amount", "std"),
            total_transaction_value=("Value", "sum"),
            avg_transaction_value=("Value", "mean"),
        )
        .reset_index()
    )

    customer_features["std_transaction_amount"] = customer_features[
        "std_transaction_amount"
    ].fillna(0)

    return customer_features


def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, and Monetary values per customer.

    Recency is calculated from a fixed snapshot date:
    max(TransactionStartTime) + 1 day.
    """
    validate_columns(
        df,
        ["CustomerId", "TransactionId", "TransactionStartTime", "Value"],
    )

    data = df.copy()
    data["TransactionStartTime"] = pd.to_datetime(
        data["TransactionStartTime"],
        errors="coerce",
    )

    snapshot_date = data["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = (
        data.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (snapshot_date - x.max()).days,
            ),
            Frequency=("TransactionId", "count"),
            Monetary=("Value", "sum"),
        )
        .reset_index()
    )

    return rfm


def create_proxy_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create is_high_risk proxy target using RFM and K-Means clustering.

    The highest-risk cluster is defined as the cluster with:
    - highest Recency
    - lowest Frequency
    - lowest Monetary
    """
    rfm = calculate_rfm(df)

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    rfm["risk_cluster"] = kmeans.fit_predict(rfm_scaled)

    cluster_profile = (
        rfm.groupby("risk_cluster")
        .agg(
            avg_recency=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean"),
        )
        .reset_index()
    )

    cluster_profile["risk_score"] = (
        cluster_profile["avg_recency"].rank(ascending=True)
        + cluster_profile["avg_frequency"].rank(ascending=False)
        + cluster_profile["avg_monetary"].rank(ascending=False)
    )

    high_risk_cluster = cluster_profile.sort_values(
        "risk_score",
        ascending=False,
    )["risk_cluster"].iloc[0]

    rfm["is_high_risk"] = (
        rfm["risk_cluster"] == high_risk_cluster
    ).astype(int)

    return rfm[
        [
            "CustomerId",
            "Recency",
            "Frequency",
            "Monetary",
            "risk_cluster",
            "is_high_risk",
        ]
    ]


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Lightweight WoE-style transformer.

    This satisfies WoE/IV-style transformation requirement without relying on
    fragile external packages. It computes Weight of Evidence mappings for
    binned numeric features using the binary target.
    """

    def __init__(self, bins: int = 5):
        self.bins = bins
        self.woe_maps_ = {}
        self.iv_values_ = {}

    def fit(self, X, y):
        X_df = pd.DataFrame(X).copy()
        y_series = pd.Series(y).reset_index(drop=True)

        for col in X_df.columns:
            series = pd.to_numeric(X_df[col], errors="coerce")
            binned = pd.qcut(
                series.rank(method="first"),
                q=self.bins,
                duplicates="drop",
            )

            temp = pd.DataFrame({"bin": binned, "target": y_series})

            grouped = temp.groupby("bin", observed=False)["target"].agg(
                events="sum",
                total="count",
            )
            grouped["non_events"] = grouped["total"] - grouped["events"]

            grouped["event_rate"] = (
                grouped["events"] + 0.5
            ) / (grouped["events"].sum() + 0.5)

            grouped["non_event_rate"] = (
                grouped["non_events"] + 0.5
            ) / (grouped["non_events"].sum() + 0.5)

            grouped["woe"] = np.log(
                grouped["non_event_rate"] / grouped["event_rate"]
            )

            grouped["iv"] = (
                grouped["non_event_rate"] - grouped["event_rate"]
            ) * grouped["woe"]

            self.woe_maps_[col] = grouped["woe"].to_dict()
            self.iv_values_[col] = grouped["iv"].sum()

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        transformed = pd.DataFrame(index=X_df.index)

        for col in X_df.columns:
            series = pd.to_numeric(X_df[col], errors="coerce")
            binned = pd.qcut(
                series.rank(method="first"),
                q=self.bins,
                duplicates="drop",
            )
            transformed[col] = binned.map(self.woe_maps_[col]).astype(float)

        return transformed.fillna(0).to_numpy()


def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    """
    Build sklearn preprocessing pipeline with:
    - imputation
    - scaling
    - categorical one-hot encoding
    """
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
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
        ]
    )

    return pipeline


def build_model_ready_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build model-ready customer-level dataset with engineered features and target.
    """
    validate_columns(df, REQUIRED_COLUMNS)

    data = extract_datetime_features(df)
    aggregates = create_aggregate_features(data)
    proxy_target = create_proxy_target(data)

    datetime_features = (
        data.groupby("CustomerId")
        .agg(
            avg_transaction_hour=("transaction_hour", "mean"),
            most_common_transaction_month=(
                "transaction_month",
                lambda x: x.mode().iloc[0],
            ),
        )
        .reset_index()
    )

    categorical_features = (
        data.groupby("CustomerId")
        .agg(
            most_common_product_category=(
                "ProductCategory",
                lambda x: x.mode().iloc[0],
            ),
            most_common_channel=(
                "ChannelId",
                lambda x: x.mode().iloc[0],
            ),
            most_common_pricing_strategy=(
                "PricingStrategy",
                lambda x: x.mode().iloc[0],
            ),
        )
        .reset_index()
    )

    model_ready = (
        aggregates.merge(proxy_target, on="CustomerId", how="left")
        .merge(datetime_features, on="CustomerId", how="left")
        .merge(categorical_features, on="CustomerId", how="left")
    )

    return model_ready


def save_processed_dataset(
    input_path: str = "data/raw/data.csv",
    output_path: str = "data/processed/model_ready_data.csv",
) -> pd.DataFrame:
    """Load raw data, process it, save processed dataset, and return it."""
    df = pd.read_csv(input_path)
    processed = build_model_ready_dataset(df)

    processed.to_csv(output_path, index=False)

    return processed


if __name__ == "__main__":
    save_processed_dataset()