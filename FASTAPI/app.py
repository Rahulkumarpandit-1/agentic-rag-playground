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

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

app = FastAPI()

TIER_1_CITIES = {"mumbai", "delhi", "bangalore", "hyderabad", "chennai", "kolkata"}


class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=0, lt=120, description="Age of the patient")]
    weight: Annotated[float, Field(..., gt=0, description="weight of the patient")]
    height: Annotated[float, Field(..., gt=0, lt=2.5, description="height of the patient")]
    incoming_lpa: Annotated[float, Field(..., gt=0, description="annual salary of user in lpa")]
    smoker: Annotated[bool, Field(..., description="is user a smoker")]
    city: Annotated[str, Field(..., description="city that user belongs to")]
    occupation: Annotated[
        Literal["student", "government_job", "business", "private_job"],
        Field(..., description="occupation of the user")
    ]

    @computed_field
    @property
    def bmi(self) -> float:
        return self.weight / (self.height ** 2)

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi >= 30:
            return "high"
        elif self.smoker or self.bmi >= 25:
            return "medium"
        return "low"

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age < 40:
            return "adult"
        elif self.age < 60:
            return "middle_aged"
        return "senior"

    @computed_field
    @property
    def city_tier(self) -> int:
        city = self.city.strip().lower()
        return 1 if city in TIER_1_CITIES else 2


def classify_premium(value: float) -> str:
    if value >= 10000:
        return "high"
    elif value >= 6000:
        return "medium"
    return "low"


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
