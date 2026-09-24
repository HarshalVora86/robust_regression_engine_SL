"""
Robust Regression Engine — House Price Prediction App
Streamlit app for project.

How it works:
- Loads the dataset and retrains the same preprocessing + best model
  (SVR with tuned RBF kernel) used in the notebook, cached so it only
  runs once per session.
- Lets the user enter property details and get a predicted price.
- Shows the model's test-set performance for context.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="House Price Predictor",
                   page_icon="🏠", layout="centered")

DATA_PATHS = [
    "Advanced_Regression_HousePrice_Dataset_3800.xlsx",
    "dataset/Advanced_Regression_HousePrice_Dataset_3800.xlsx",
    "data/Advanced_Regression_HousePrice_Dataset_3800.xlsx",
]

SCALE_FEATURES = ["area_sqft", "location_score",
                  "property_age", "distance_city_km", "crime_rate_index"]
# from the notebook's tuning loop
BEST_PARAMS = {"C": 10, "gamma": 0.01, "epsilon": 0.01}


# ----------------------------------------------------------------------
# Load data + train pipeline (cached so it only runs once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_and_train():
    df = None
    for path in DATA_PATHS:
        try:
            df = pd.read_excel(path)
            break
        except FileNotFoundError:
            continue
    if df is None:
        return None

    features = df.drop(columns=["property_id", "sale_date", "house_price_inr"])
    target = df["house_price_inr"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    x_scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[SCALE_FEATURES] = x_scaler.fit_transform(
        X_train[SCALE_FEATURES])
    X_test_scaled[SCALE_FEATURES] = x_scaler.transform(X_test[SCALE_FEATURES])

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(
        y_train.values.reshape(-1, 1)).ravel()

    model = SVR(kernel="rbf", **BEST_PARAMS)
    model.fit(X_train_scaled, y_train_scaled)

    # Test-set metrics for display
    test_pred_scaled = model.predict(X_test_scaled)
    test_pred = y_scaler.inverse_transform(
        test_pred_scaled.reshape(-1, 1)).ravel()
    r2 = r2_score(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    mae = mean_absolute_error(y_test, test_pred)

    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_columns": list(features.columns),
        "metrics": {"r2": r2, "rmse": rmse, "mae": mae},
        "feature_ranges": {
            "area_sqft": (int(features.area_sqft.min()), int(features.area_sqft.max())),
            "property_age": (int(features.property_age.min()), int(features.property_age.max())),
            "distance_city_km": (float(features.distance_city_km.min()), float(features.distance_city_km.max())),
            "location_score": (float(features.location_score.min()), float(features.location_score.max())),
            "crime_rate_index": (float(features.crime_rate_index.min()), float(features.crime_rate_index.max())),
        },
    }


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🏠 House Price Predictor")
st.caption("Robust Regression Engine")

bundle = load_and_train()

if bundle is None:
    st.error(
        "Dataset not found. Place **Advanced_Regression_HousePrice_Dataset_3800.xlsx** "
        "in the same folder as this app (or in a `dataset/` or `data/` subfolder) and reload."
    )
    st.stop()

metrics = bundle["metrics"]
col1, col2, col3 = st.columns(3)

st.divider()
st.subheader("Enter Property Details")

ranges = bundle["feature_ranges"]

c1, c2 = st.columns(2)
with c1:
    area_sqft = st.number_input(
        "Area (sqft)", min_value=200, max_value=10000,
        value=int(np.mean(ranges["area_sqft"])), step=50
    )
    bedrooms = st.number_input(
        "Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input(
        "Bathrooms", min_value=1, max_value=10, value=2, step=1)
    location_score = st.slider(
        "Location Score", min_value=0.0, max_value=10.0,
        value=round(float(np.mean(ranges["location_score"])), 1), step=0.1
    )

with c2:
    property_age = st.number_input(
        "Property Age (years)", min_value=0, max_value=100,
        value=int(np.mean(ranges["property_age"])), step=1
    )
    distance_city_km = st.number_input(
        "Distance from City Center (km)", min_value=0.0, max_value=100.0,
        value=round(float(np.mean(ranges["distance_city_km"])), 1), step=0.5
    )
    crime_rate_index = st.slider(
        "Crime Rate Index", min_value=0.0, max_value=10.0,
        value=round(float(np.mean(ranges["crime_rate_index"])), 1), step=0.1
    )
    near_school = st.selectbox("Near a School?", ["No", "Yes"])
    near_metro = st.selectbox("Near a Metro Station?", ["No", "Yes"])

predict_btn = st.button("Predict Price", type="primary",
                        use_container_width=True)

if predict_btn:
    input_df = pd.DataFrame([{
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "location_score": location_score,
        "property_age": property_age,
        "distance_city_km": distance_city_km,
        "near_school": 1 if near_school == "Yes" else 0,
        "near_metro": 1 if near_metro == "Yes" else 0,
        "crime_rate_index": crime_rate_index,
    }])[bundle["feature_columns"]]

    input_scaled = input_df.copy()
    input_scaled[SCALE_FEATURES] = bundle["x_scaler"].transform(
        input_df[SCALE_FEATURES])

    pred_scaled = bundle["model"].predict(input_scaled)
    pred_price = bundle["y_scaler"].inverse_transform(
        pred_scaled.reshape(-1, 1)).ravel()[0]

    st.success(f"### Predicted Price: ₹{pred_price:,.0f}")
    st.caption(
        f"Estimate based on SVR (RBF kernel, tuned). Typical error range on test data: "
        f"± ₹{metrics['mae']:,.0f} (MAE)."
    )

st.divider()
with st.expander("About this model"):
    st.markdown(
        f"""
        - **Model:** Support Vector Regression, RBF kernel, tuned via grid search
          (C={BEST_PARAMS['C']}, gamma={BEST_PARAMS['gamma']}, epsilon={BEST_PARAMS['epsilon']})
        - **Features used:** {', '.join(bundle['feature_columns'])}
        - **Preprocessing:** Numerical features standardized (StandardScaler); target scaled
          separately and inverse-transformed for predictions.
        - **Performance (held-out test set):** R² = {metrics['r2']:.3f}, RMSE = ₹{metrics['rmse']:,.0f},
          MAE = ₹{metrics['mae']:,.0f}
        """
    )
