"""
Train and export the complete preprocessing + regression pipeline
for the Robust Regression Engine project.

Run:
    python train_model.py

Output:
    Model/house_price_pipeline.pkl
    Model/metrics.json
"""
import json
import joblib
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Features that get scaled (continuous, varying ranges)
SCALE_FEATURES = ["area_sqft", "location_score", "property_age", "distance_city_km", "crime_rate_index"]
# Features left as-is (binary flags / small integer counts)
PASSTHROUGH_FEATURES = ["bedrooms", "bathrooms", "near_school", "near_metro"]
FEATURES = SCALE_FEATURES + PASSTHROUGH_FEATURES
TARGET = "house_price_inr"

# Best hyperparameters found via the tuning loop in the notebook (Part F, Task 20)
BEST_PARAMS = {"C": 10, "gamma": 0.01, "epsilon": 0.01}


def main():
    df = pd.read_excel("dataset/Advanced_Regression_HousePrice_Dataset_3800.xlsx")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Only the SCALE_FEATURES columns go through imputer + scaler;
    # PASSTHROUGH_FEATURES pass through untouched via remainder="passthrough"
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, SCALE_FEATURES)],
        remainder="passthrough",  # keeps PASSTHROUGH_FEATURES as-is, in order
    )

    # SVR needs its target scaled (fixed epsilon/C margins otherwise become
    # meaningless against a target in the millions). TransformedTargetRegressor
    # scales y on .fit() and automatically inverse-transforms on .predict(),
    # so no manual y_scaler.inverse_transform() calls are needed anywhere else.
    svr_with_scaled_target = TransformedTargetRegressor(
        regressor=SVR(kernel="rbf", **BEST_PARAMS),
        transformer=StandardScaler(),
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", svr_with_scaled_target),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mse),
        "RMSE": float(mse ** 0.5),
        "R2": float(r2_score(y_test, y_pred)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
    }

    os.makedirs("Model", exist_ok=True)

    joblib.dump(pipeline, "Model/house_price_pipeline.pkl")

    with open("Model/metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)  # json.dump (not dumps) writes directly to the file

    print("\nModel trained successfully.")
    print("Pipeline: Model/house_price_pipeline.pkl")
    print("\nEvaluation:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
