"""
Robust Regression Engine — House Price Prediction App
Streamlit app for project.

Loads the pre-trained pipeline (Model/house_price_pipeline.pkl), produced by
running train_model.py, and uses it directly for predictions.

Run with:  streamlit run app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="House Price Predictor",
                   page_icon="🏠", layout="centered")

PIPELINE_PATHS = [
    "Model/house_price_pipeline.pkl",
    "model/house_price_pipeline.pkl",
    "../Model/house_price_pipeline.pkl",
]
METRICS_PATHS = [
    "Model/metrics.json",
    "model/metrics.json",
    "../Model/metrics.json",
]

SCALE_FEATURES = ["area_sqft", "location_score",
                  "property_age", "distance_city_km", "crime_rate_index"]
PASSTHROUGH_FEATURES = ["bedrooms", "bathrooms", "near_school", "near_metro"]
# must match train_model.py's FEATURES order
FEATURE_ORDER = SCALE_FEATURES + PASSTHROUGH_FEATURES


# ----------------------------------------------------------------------
# Load pipeline + metrics (cached so it only runs once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    for path in PIPELINE_PATHS:
        try:
            return joblib.load(path)
        except FileNotFoundError:
            continue
    return None


@st.cache_data
def load_metrics():
    for path in METRICS_PATHS:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            continue
    return None


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🏠 House Price Predictor")
st.caption("Robust Regression Engine")

pipeline = load_pipeline()
metrics = load_metrics()

if pipeline is None:
    st.error(
        "Model file not found. Run **train_model.py** first to generate "
        "`Model/house_price_pipeline.pkl`, then reload this app."
    )
    st.stop()

st.divider()
st.subheader("Enter Property Details")

c1, c2 = st.columns(2)
with c1:
    area_sqft = st.number_input(
        "Area (sqft)", min_value=200, max_value=10000, value=1800, step=50)
    bedrooms = st.number_input(
        "Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input(
        "Bathrooms", min_value=1, max_value=10, value=2, step=1)
    location_score = st.slider(
        "Location Score", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

with c2:
    property_age = st.number_input(
        "Property Age (years)", min_value=0, max_value=100, value=15, step=1)
    distance_city_km = st.number_input(
        "Distance from City Center (km)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
    crime_rate_index = st.slider(
        "Crime Rate Index", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    near_school = st.selectbox("Near a School?", ["No", "Yes"])
    near_metro = st.selectbox("Near a Metro Station?", ["No", "Yes"])

predict_btn = st.button("Predict Price", type="primary",
                        use_container_width=True)

if predict_btn:
    input_df = pd.DataFrame([{
        "area_sqft": area_sqft,
        "location_score": location_score,
        "property_age": property_age,
        "distance_city_km": distance_city_km,
        "crime_rate_index": crime_rate_index,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "near_school": 1 if near_school == "Yes" else 0,
        "near_metro": 1 if near_metro == "Yes" else 0,
    }])[FEATURE_ORDER]

    # Pipeline handles scaling + prediction + target inverse-transform internally —
    # no manual scaler calls needed here.
    pred_price = pipeline.predict(input_df)[0]

    st.success(f"### Predicted Price: ₹{pred_price:,.0f}")
    if metrics:
        st.caption(
            f"Estimate based on SVR (RBF kernel, tuned). Typical error range on test data: "
            f"± ₹{metrics['MAE']:,.0f} (MAE)."
        )

st.divider()
with st.expander("About this model"):
    details = [
        "**Model:** Support Vector Regression, RBF kernel, tuned via grid search",
        f"**Features used:** {', '.join(FEATURE_ORDER)}",
        "**Preprocessing:** numerical features standardized via a scikit-learn "
        "`Pipeline` + `ColumnTransformer`; target scaled internally via "
        "`TransformedTargetRegressor` (auto inverse-transformed on prediction).",
    ]
    if metrics:
        details.append(
            f"**Performance (held-out test set):** R² = {metrics['R2']:.3f}, "
            f"RMSE = ₹{metrics['RMSE']:,.0f}, MAE = ₹{metrics['MAE']:,.0f}"
        )
    st.markdown("\n".join(f"- {d}" for d in details))
