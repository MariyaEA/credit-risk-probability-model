from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="API for predicting customer credit risk probability.",
    version="1.0.0",
)

MODEL_URI = "models:/BatiBankCreditRiskModel/latest"


def load_model():
    try:
        return mlflow.pyfunc.load_model(MODEL_URI)
    except Exception:
        return None


model = load_model()


@app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "Bati Bank Credit Risk API",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    input_data = pd.DataFrame([request.model_dump()])

    if model is None:
        risk_probability = 0.5
    else:
        prediction = model.predict(input_data)
        risk_probability = float(prediction[0])

    risk_label = "high_risk" if risk_probability >= 0.5 else "low_risk"

    return PredictionResponse(
        risk_probability=risk_probability,
        risk_label=risk_label,
    )
