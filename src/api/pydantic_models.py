from pydantic import BaseModel


class PredictionRequest(BaseModel):
    total_transaction_amount: float
    avg_transaction_amount: float
    transaction_count: int
    std_transaction_amount: float
    total_transaction_value: float
    avg_transaction_value: float
    Recency: float
    Frequency: float
    Monetary: float
    avg_transaction_hour: float
    most_common_transaction_month: int
    most_common_product_category: str
    most_common_channel: str
    most_common_pricing_strategy: int


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_label: str