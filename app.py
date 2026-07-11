import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Population Estimator",
    page_icon="🌍",
    layout="wide"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load("models/population_model.pkl")


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_dataset():

    features = pd.read_csv("data/Indian_City_NightLights.csv")
    population = pd.read_csv("data/city_population.csv")

    dataset = pd.merge(
        features,
        population,
        on="City",
        how="left",
        suffixes=("_old", "")
    )

    # Remove old population column if present
    if "Population_old" in dataset.columns:
        dataset.drop(columns=["Population_old"], inplace=True)

    # Rename City -> city
    dataset["city"] = dataset["City"]

    # -------------------------
    # Feature Engineering
    # -------------------------

    dataset["Brightness_Range"] = (
        dataset["average_masked_max"]
        - dataset["average_masked_min"]
    )

    dataset["Brightness_Ratio"] = (
        dataset["average_masked_mean"]
        /
        (dataset["average_masked_stdDev"] + 1)
    )

    dataset["Brightness_Product"] = (
        dataset["average_masked_mean"]
        *
        dataset["average_masked_stdDev"]
    )

    return dataset


model = load_model()
dataset = load_dataset()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🌍 AI Population Estimator")

st.sidebar.markdown("""
Estimate city population using:

- 🛰 NASA VIIRS Night-Light Data
- 🌍 Google Earth Engine
- 🤖 Random Forest Regression
- 🐍 Python
- 📊 Streamlit
""")

st.sidebar.markdown("---")

st.sidebar.metric("Algorithm", "Random Forest")
st.sidebar.metric("Training Cities", len(dataset))
st.sidebar.metric("Features Used", "7")
st.sidebar.metric("Satellite Source", "NASA VIIRS")

# ==========================================================
# TITLE
# ==========================================================

st.title("🌍 AI Population Estimator")

st.write("""
Estimate the population of Indian cities using
satellite-derived night-light intensity and
Machine Learning.
""")

st.divider()

# ==========================================================
# CITY SELECTION
# ==========================================================

city = st.selectbox(
    "🏙 Select a City",
    sorted(dataset["city"].unique())
)

selected = dataset[
    dataset["city"] == city
].iloc[0]

# ==========================================================
# SATELLITE FEATURES
# ==========================================================

st.subheader("🛰 Satellite Features")

feature_df = pd.DataFrame({

    "Feature":[
        "Average Brightness",
        "Maximum Brightness",
        "Minimum Brightness",
        "Standard Deviation",
        "Brightness Range",
        "Brightness Ratio",
        "Brightness Product"
    ],

    "Value":[
        round(selected["average_masked_mean"],2),
        round(selected["average_masked_max"],2),
        round(selected["average_masked_min"],2),
        round(selected["average_masked_stdDev"],2),
        round(selected["Brightness_Range"],2),
        round(selected["Brightness_Ratio"],2),
        round(selected["Brightness_Product"],2)
    ]

})

st.dataframe(feature_df,use_container_width=True)

# ==========================================================
# PREDICTION
# ==========================================================

st.divider()

if st.button("🚀 Predict Population",use_container_width=True):

    X=np.array([[

        selected["average_masked_mean"],
        selected["average_masked_max"],
        selected["average_masked_min"],
        selected["average_masked_stdDev"],
        selected["Brightness_Range"],
        selected["Brightness_Ratio"],
        selected["Brightness_Product"]

    ]])

    prediction=model.predict(X)[0]

    actual=selected["Population"]

    error=abs(actual-prediction)

    percentage=(error/actual)*100

    st.success("Prediction Completed Successfully!")

    c1,c2,c3=st.columns(3)

    c1.metric(
        "Actual Population",
        f"{int(actual):,}"
    )

    c2.metric(
        "Predicted Population",
        f"{int(prediction):,}"
    )

    c3.metric(
        "Prediction Error",
        f"{percentage:.2f}%"
    )

# ==========================================================
# MODEL INSIGHTS
# ==========================================================

st.divider()

st.header("📊 Model Insights")

col1,col2=st.columns(2)

with col1:

    st.subheader("Feature Importance")

    if os.path.exists("images/feature_importance.png"):
        st.image(
            "images/feature_importance.png",
            use_container_width=True
        )

with col2:

    st.subheader("Predicted vs Actual")

    if os.path.exists("images/prediction_plot.png"):
        st.image(
            "images/prediction_plot.png",
            use_container_width=True
        )

# ==========================================================
# DATASET
# ==========================================================

st.divider()

st.header("📋 Dataset Preview")

st.dataframe(

    dataset[
        [
            "city",
            "average_masked_mean",
            "average_masked_max",
            "Population"
        ]
    ],

    use_container_width=True,
    height=350

)

st.caption(
    "Developed by Himanshi Agrawal"
)