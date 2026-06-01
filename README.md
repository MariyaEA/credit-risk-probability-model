![Python](https://img.shields.io/badge/Python-3.11-blue)
![CI](https://github.com/MariyaEA/credit-risk-probability-model/actions/workflows/ci.yml/badge.svg)


# Credit Risk Probability Model for Alternative Data

## Project Overview

**Dataset Source:** Xente Transaction Dataset — 10 Academy Week 4 Credit Risk Modeling Challenge.

Bati Bank is partnering with an eCommerce company to support a Buy-Now-Pay-Later credit service. The goal of this project is to build a credit risk probability model using customer transaction behavior from the Xente platform.

The raw dataset does not contain a direct default label. Therefore, the project uses behavioral patterns — especially Recency, Frequency, and Monetary (RFM) analysis — to engineer a proxy risk target variable for model training.

---

## Business Objective

The main objective is to develop an interpretable, reproducible, and deployable credit risk model that can:

1. Identify high-risk and low-risk customers using behavioral transaction data.
2. Produce a risk probability score for new customers.
3. Support credit decisioning, loan approval, credit limit assignment, and pricing strategies.
4. Align model development with Basel II expectations around risk measurement, documentation, interpretability, and monitoring.

---

# Credit Scoring Business Understanding

## 1. Basel II and the Need for Interpretability

The Basel II Accord emphasizes sound credit risk measurement, internal controls, documentation, and model governance. In this project, that means the model should not only produce accurate predictions but should also be explainable and auditable.

For Bati Bank, interpretability is important because credit decisions affect customers directly. Risk teams and decision-makers must understand which customer behaviors influence the score, why a customer may be classified as high risk, and how the model should be monitored over time. This supports regulatory confidence, business trust, and responsible lending.

---

## 2. Why a Proxy Variable Is Necessary

The dataset does not include an actual loan default outcome. Without a direct default label, supervised credit risk modeling cannot be performed directly. Therefore, a proxy variable is needed to approximate credit risk using observable customer behavior.

The planned proxy target is created using RFM analysis. Customers with weak engagement patterns, such as low transaction frequency, low monetary value, and long time since last transaction, may be labeled as relatively high risk.

However, this introduces business risk. A proxy label is not the same as actual default. Some low-activity customers may still be creditworthy, while some active customers may still default. For this reason, the proxy must be documented clearly as a modeling assumption and should be validated with real repayment data before production use.

---

## 3. Trade-off Between Interpretable and High-Performance Models

A simple model such as Logistic Regression with Weight of Evidence (WoE) transformation is easier to explain, validate, and document. This is useful in regulated financial environments because risk teams can understand how each feature contributes to the final score.

More complex models such as Random Forest or Gradient Boosting may achieve stronger predictive performance by capturing nonlinear relationships and feature interactions. However, they are harder to interpret and may require additional explainability tools.

For this project, both model types are compared. The final model choice balances predictive performance, interpretability, regulatory expectations, and operational usability.

---

# Repository Structure

```text
credit-risk-probability-model/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data_processing.py
│   ├── train.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── reports/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Key EDA Insights

1. The dataset is transaction-level and must be aggregated to customer level for credit scoring.
2. Transaction values are highly skewed and contain substantial outliers.
3. Product category, provider, and channel variables contain useful behavioral signals.
4. Time-based transaction features support customer behavior analysis.
5. RFM analysis provides a practical foundation for constructing the proxy high-risk target variable.

---

# Setup Instructions

```bash
git clone https://github.com/MariyaEA/credit-risk-probability-model.git
cd credit-risk-probability-model

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

jupyter notebook
```

Place the raw Xente dataset files inside:

```text
data/raw/
```

before running the pipeline.

---

# Data Source

The dataset used in this project is the Xente transaction dataset provided as part of the 10 Academy Week 4 Credit Risk Modeling Challenge.

The dataset contains anonymized customer transaction records including:

* Transaction identifiers
* Customer identifiers
* Transaction amounts and values
* Product categories
* Pricing strategies
* Fraud indicators
* Transaction timestamps
* Channel information

The raw dataset is excluded from version control and stored locally inside:

```text
data/raw/
```

---

# Business and Regulatory Relevance

The exploratory analysis performed in this project is directly connected to future credit risk modeling and Basel II compliance requirements.

The EDA findings help identify:

* Customer behavioral patterns relevant to creditworthiness
* Potential predictive features for proxy risk modeling
* Data quality and preprocessing requirements
* Feature engineering opportunities
* Sources of skewness, imbalance, and outlier influence that may affect model reliability

These findings support the development of interpretable and regulator-friendly credit scoring models capable of informing:

* Customer credit qualification
* Loan approval decisions
* Credit limit assignment
* Risk-based pricing strategies
* Portfolio monitoring and risk management

The project emphasizes explainability, documentation, and transparent modeling practices aligned with Basel II expectations for responsible financial risk assessment.

---

# Final Submission Progress

This final submission includes:

* Task 1: Credit risk business understanding
* Task 2: Exploratory data analysis
* Task 3: Feature engineering pipeline
* Task 4: Proxy target variable construction using RFM analysis and K-Means clustering
* Task 5: Model training, hyperparameter tuning, MLflow tracking, and unit testing
* Task 6: FastAPI deployment, Docker containerization, and GitHub Actions CI/CD workflow

---

# Feature Engineering and Proxy Target Development

The feature engineering pipeline performs:

* Customer-level aggregation
* Datetime feature extraction
* Missing value handling
* One-hot categorical encoding
* Numerical scaling and normalization
* Weight of Evidence (WoE)-style transformation
* sklearn Pipeline integration

The proxy target variable is generated using:

1. Recency, Frequency, and Monetary (RFM) behavioral analysis
2. StandardScaler preprocessing
3. K-Means clustering with reproducible random_state configuration
4. Identification of the highest-risk behavioral cluster
5. Construction of the binary `is_high_risk` target variable

---

# Model Training and Experiment Tracking

The modeling workflow includes:

* Logistic Regression
* Random Forest Classification
* Hyperparameter tuning using GridSearchCV
* Train/test split with reproducible random_state
* MLflow experiment tracking and model logging

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

The final modeling workflow prioritizes both predictive performance and interpretability to align with Basel II regulatory expectations.

---

# Deployment and CI/CD

The project includes:

* FastAPI application with `/predict` endpoint
* Pydantic request and response schemas
* Docker containerization
* Docker Compose configuration
* GitHub Actions CI/CD workflow
* Automated flake8 linting and pytest execution

This supports reproducible deployment and scalable model serving for future production integration.
