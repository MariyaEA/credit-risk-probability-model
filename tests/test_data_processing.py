"""
Unit tests for data processing functions.
"""

import pandas as pd

from src.data_processing import (
    calculate_rfm,
    create_aggregate_features,
    extract_datetime_features,
)


def sample_transactions() -> pd.DataFrame:
    """Create small sample transaction dataframe for tests."""
    return pd.DataFrame(
        {
            "TransactionId": ["T1", "T2", "T3"],
            "CustomerId": ["C1", "C1", "C2"],
            "Amount": [100.0, 200.0, 50.0],
            "Value": [100.0, 200.0, 50.0],
            "ProductCategory": ["airtime", "airtime", "utility"],
            "ChannelId": ["web", "web", "android"],
            "PricingStrategy": [1, 1, 2],
            "TransactionStartTime": [
                "2024-01-01 10:00:00",
                "2024-01-02 11:00:00",
                "2024-01-03 12:00:00",
            ],
        }
    )


def test_extract_datetime_features_returns_expected_columns():
    """Datetime extraction should add hour, day, month, and year."""
    df = sample_transactions()
    result = extract_datetime_features(df)

    expected_columns = {
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
    }

    assert expected_columns.issubset(result.columns)


def test_create_aggregate_features_returns_customer_level_rows():
    """Aggregation should return one row per customer."""
    df = sample_transactions()
    result = create_aggregate_features(df)

    assert result["CustomerId"].nunique() == 2
    assert "total_transaction_amount" in result.columns
    assert "avg_transaction_amount" in result.columns
    assert "transaction_count" in result.columns
    assert "std_transaction_amount" in result.columns


def test_calculate_rfm_returns_required_columns():
    """RFM calculation should return Recency, Frequency, and Monetary."""
    df = sample_transactions()
    result = calculate_rfm(df)

    expected_columns = {
        "CustomerId",
        "Recency",
        "Frequency",
        "Monetary",
    }

    assert expected_columns.issubset(result.columns)