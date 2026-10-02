from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REQUIRED_COLUMNS = [
    "bmi",
    "age_group",
    "lifestyle_risk",
    "city_tier",
    "income_lpa",
    "premium",
]


def make_sample_data(path: Path) -> pd.DataFrame:
    sample = pd.DataFrame(
        [
            {
                "bmi": 22.5,
                "age_group": "young",
                "lifestyle_risk": "low",
                "city_tier": 1,
                "income_lpa": 18.5,
                "premium": 4800,
            },
            {
                "bmi": 26.8,
                "age_group": "adult",
                "lifestyle_risk": "medium",
                "city_tier": 2,
                "income_lpa": 24.0,
                "premium": 6800,
            },
            {
                "bmi": 31.4,
                "age_group": "middle_aged",
                "lifestyle_risk": "high",
                "city_tier": 1,
                "income_lpa": 32.0,
                "premium": 9100,
            },
            {
                "bmi": 29.1,
                "age_group": "adult",
                "lifestyle_risk": "medium",
                "city_tier": 2,
                "income_lpa": 40.0,
                "premium": 8600,
            },
            {
                "bmi": 33.2,
                "age_group": "senior",
                "lifestyle_risk": "high",
                "city_tier": 1,
                "income_lpa": 45.0,
                "premium": 12500,
            },
            {
                "bmi": 24.9,
                "age_group": "young",
                "lifestyle_risk": "low",
                "city_tier": 2,
                "income_lpa": 16.0,
                "premium": 5300,
            },
            {
                "bmi": 27.7,
                "age_group": "adult",
                "lifestyle_risk": "medium",
                "city_tier": 1,
                "income_lpa": 20.0,
                "premium": 6600,
            },
            {
                "bmi": 35.1,
                "age_group": "middle_aged",
                "lifestyle_risk": "high",
                "city_tier": 2,
                "income_lpa": 28.5,
                "premium": 9800,
            },
        ]
    )
    sample.to_csv(path, index=False)
    return sample


def build_pipeline() -> Pipeline:
    numeric_features = ["bmi", "city_tier", "income_lpa"]
    categorical_features = ["age_group", "lifestyle_risk"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(n_estimators=300, random_state=42),
            ),
        ]
    )
    return model


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feature_columns = ["bmi", "age_group", "lifestyle_risk", "city_tier", "income_lpa"]
    X = df[feature_columns].copy()
    y = df["premium"].copy()
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save a premium prediction model.")
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to CSV file with columns: bmi, age_group, lifestyle_risk, city_tier, income_lpa, premium",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "model.pkl"),
        help="Output file path to save the trained model",
    )
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else Path(__file__).resolve().parent / "insurance.csv"

    if not data_path.exists():
        print(f"No dataset found at {data_path}. Creating a sample dataset for demo training...")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        make_sample_data(data_path)

    df = pd.read_csv(data_path)
    X, y = prepare_data(df)

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(pipeline, f)

    print(f"Model trained and saved to: {output_path}")
    print(f"Rows used: {len(df)}")


if __name__ == "__main__":
    main()
