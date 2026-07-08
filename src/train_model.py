import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# CREATE REQUIRED FOLDERS
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ==========================================================
# LOAD DATASETS
# ==========================================================

print("=" * 60)
print("Loading datasets...")
print("=" * 60)

features = pd.read_csv("data/Indian_City_NightLights.csv")
population = pd.read_csv("data/city_population.csv")

print("\nNight Light CSV Columns:")
print(features.columns.tolist())

print("\nPopulation CSV Columns:")
print(population.columns.tolist())

# ==========================================================
# MERGE DATASETS
# ==========================================================

dataset = pd.merge(
    features,
    population,
    left_on="city",
    right_on="City",
    how="inner"
)

print("\nMerged Dataset Columns:")
print(dataset.columns.tolist())

print(f"\nTotal Cities after merge: {len(dataset)}")

# ==========================================================
# FIND POPULATION COLUMN
# ==========================================================

if "Population" in dataset.columns:
    population_col = "Population"

elif "Population_y" in dataset.columns:
    population_col = "Population_y"

elif "Population_x" in dataset.columns:
    population_col = "Population_x"

else:
    raise Exception("Population column not found!")

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

dataset["Brightness_Range"] = (
    dataset["average_masked_max"] -
    dataset["average_masked_min"]
)

dataset["Brightness_Ratio"] = (
    dataset["average_masked_mean"] /
    (dataset["average_masked_stdDev"] + 1)
)

dataset["Brightness_Product"] = (
    dataset["average_masked_mean"] *
    dataset["average_masked_stdDev"]
)

# ==========================================================
# SELECT FEATURES
# ==========================================================

feature_columns = [
    "average_masked_mean",
    "average_masked_max",
    "average_masked_min",
    "average_masked_stdDev",
    "Brightness_Range",
    "Brightness_Ratio",
    "Brightness_Product"
]

# Remove rows with missing values

dataset = dataset.dropna(subset=feature_columns + [population_col])

print(f"\nRows remaining after removing missing values: {len(dataset)}")

X = dataset[feature_columns]
y = dataset[population_col]

print("\nFeatures Used:")
print(feature_columns)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"\nTraining Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# ==========================================================
# TRAIN MODEL
# ==========================================================

print("\nTraining Random Forest Model...")

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model trained successfully!")

# ==========================================================
# PREDICTIONS
# ==========================================================

predictions = model.predict(X_test)

# ==========================================================
# MODEL EVALUATION
# ==========================================================

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance")
print(importance_df)

plt.figure(figsize=(10,5))
plt.bar(importance_df["Feature"], importance_df["Importance"])
plt.xticks(rotation=25)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("images/feature_importance.png", dpi=300)
plt.close()

# ==========================================================
# PREDICTED VS ACTUAL
# ==========================================================

plt.figure(figsize=(6,6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.7
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

plt.xlabel("Actual Population")
plt.ylabel("Predicted Population")
plt.title("Predicted vs Actual")
plt.tight_layout()

plt.savefig(
    "images/prediction_plot.png",
    dpi=300
)

plt.close()

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    model,
    "models/population_model.pkl"
)

print("\nModel saved successfully!")

# ==========================================================
# SAVE METRICS
# ==========================================================

metrics = {
    "Algorithm": "Random Forest Regression",
    "Training Cities": int(len(dataset)),
    "Training Samples": int(len(X_train)),
    "Testing Samples": int(len(X_test)),
    "Features": len(feature_columns),
    "MAE": round(float(mae), 2),
    "RMSE": round(float(rmse), 2),
    "R2": round(float(r2), 4)
}

with open(
    "models/model_metrics.json",
    "w"
) as f:
    json.dump(metrics, f, indent=4)

print("Model metrics saved!")

print("\nGraphs saved successfully!")

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)