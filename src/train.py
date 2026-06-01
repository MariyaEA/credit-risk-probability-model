"""
Model training workflow for Bati Bank credit risk model.

Includes:
- train/test split
- preprocessing pipeline
- Logistic Regression and Random Forest models
- GridSearchCV hyperparameter tuning
- MLflow experiment tracking
- metrics logging
- best model registration
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DATA_PATH = "data/processed/model_ready_data.csv"
EXPERIMENT_NAME = "bati_bank_credit_risk_model"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load model-ready data."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. "
            "Run src/data_processing.py first."
        )

    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target."""
    if "is_high_risk" not in df.columns:
        raise ValueError("Target column is_high_risk is missing.")

    drop_columns = [
        "CustomerId",
        "is_high_risk",
        "risk_cluster",
    ]

    X = df.drop(
        columns=[col for col in drop_columns if col in df.columns]
    )
    y = df["is_high_risk"]

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing transformer for numeric and categorical columns."""
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

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

    return preprocessor


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """Evaluate model using required classification metrics."""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metrics


def train_models() -> None:
    """Train, tune, track, and register best model."""
    df = load_data()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)

    models = {
        "logistic_regression": {
            "model": LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "random_forest": {
            "model": RandomForestClassifier(
                random_state=RANDOM_STATE,
            ),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__class_weight": [None, "balanced"],
            },
        },
    }

    mlflow.set_experiment(EXPERIMENT_NAME)

    best_score = -1
    best_model = None
    best_model_name = None

    for model_name, model_config in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model_config["model"]),
            ]
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=model_config["params"],
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
        )

        with mlflow.start_run(run_name=model_name):
            grid_search.fit(X_train, y_train)

            tuned_model = grid_search.best_estimator_
            metrics = evaluate_model(tuned_model, X_test, y_test)

            mlflow.log_param("model_name", model_name)
            mlflow.log_params(grid_search.best_params_)

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            mlflow.sklearn.log_model(
                tuned_model,
                artifact_path="model",
            )

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model = tuned_model
                best_model_name = model_name

            print(f"Model: {model_name}")
            print(f"Best params: {grid_search.best_params_}")
            print(f"Metrics: {metrics}")

    with mlflow.start_run(run_name="best_model_registration"):
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="best_model",
            registered_model_name="BatiBankCreditRiskModel",
        )

        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_metric("best_roc_auc", best_score)

    print(f"Best model: {best_model_name}")
    print(f"Best ROC-AUC: {best_score}")


if __name__ == "__main__":
    train_models()