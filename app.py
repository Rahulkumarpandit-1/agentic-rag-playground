from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field, computed_field
from fastapi.responses import JSONResponse
from typing import Literal, Annotated
import sklearn
print(sklearn.__version__)
import os
print(os.listdir())
import pickle
import pandas as pd
from schema.user_input import UserInput
with open("MODEL/model.pkl", "rb") as f:
    model = pickle.load(f)
MODEL_VERSION='1.0.0'
app = FastAPI()





def classify_premium(value: float) -> str:
    
    if value >= 10000:
        return "high"
    elif value >= 6000:
        return "medium"
    return "low"

@app.get("/")
def home():
    return {"message": "Hello Insurance Prmium Prediction API"}


@app.get("/health") 
def health_check():
    
    return {"status": "ok",
            'version': MODEL_VERSION}


@app.post("/predict")
def predict_premium(data: UserInput):
    input_df = pd.DataFrame([{
        "bmi": data.bmi,
        "age_group": data.age_group,
        "lifestyle_risk": data.lifestyle_risk,
        "city_tier": data.city_tier,
        "income_lpa": data.incoming_lpa,
    }])

    prediction = float(model.predict(input_df)[0])
    premium_level = classify_premium(prediction)
    return JSONResponse(
        status_code=200,
        content={
            "premium": premium_level,
            "premium_value": round(prediction, 2),
        },
    )
