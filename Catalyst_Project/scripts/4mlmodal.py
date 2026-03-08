"""
STEP 4 — Train ML Models, Evaluate, Feature Importance, Screen Catalysts
Run AFTER Step3_EDA.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import joblib

# Color palette
C = {
    "teal": "#00C9A7",
    "sky": "#38BDF8",
    "gold": "#FBBF24",
    "coral": "#F87171",
    "violet": "#A78BFA",
    "dark": "#1A2C50",
    "bg": "#F0F4FA",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ======================================================
# 1. LOAD DATA
# ======================================================

df = pd.read_csv("master_dataset.csv")

print(f"Dataset: {df.shape[0]} materials")

FEATURES = [
    "electronegativity",
    "atomic_radius",
    "ionization_energy",
    "electron_affinity",
    "d_electrons",
    "group",
    "period",
    "melting_point",
    "d_band_center",
    "d_band_width",
    "d_band_filling",
    "work_function",
    "surface_energy",
    "bulk_modulus",
    "magnetic_moment",
    "nitride_formation_energy",
]

TARGET = "dG_N"

df_clean = df[FEATURES + [TARGET, "element"]].dropna()

X = df_clean[FEATURES]
y = df_clean[TARGET]
elements = df_clean["element"]

print("Clean samples:", len(df_clean))

# ======================================================
# 2. TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_all = scaler.transform(X)

# ======================================================
# 3. DEFINE MODELS
# ======================================================

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    ),
}

# ======================================================
# 4. TRAIN MODELS
# ======================================================

cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}

print("\nMODEL PERFORMANCE")
print("-"*60)

for name, model in models.items():

    cv_r2 = cross_val_score(model, X_train, y_train,
                            cv=cv, scoring="r2")

    cv_mae = -cross_val_score(model, X_train, y_train,
                              cv=cv,
                              scoring="neg_mean_absolute_error")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    test_r2 = r2_score(y_test, pred)
    test_mae = mean_absolute_error(y_test, pred)

    results[name] = {
        "model": model,
        "cv_r2": cv_r2.mean(),
        "test_r2": test_r2,
        "test_mae": test_mae
    }

    print(f"{name:20s}  CV_R2={cv_r2.mean():.3f}  Test_R2={test_r2:.3f}")

# ======================================================
# 5. BEST MODEL
# ======================================================

best_name = max(results, key=lambda x: results[x]["cv_r2"])
best_model = results[best_name]["model"]

print("\nBest Model:", best_name)

# ======================================================
# 6. PREDICT VS ACTUAL
# ======================================================

y_pred = best_model.predict(X_all)

plt.figure(figsize=(6,6))

plt.scatter(y, y_pred,
            s=120,
            color=C["teal"],
            edgecolors="white")

lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]

plt.plot(lims, lims, "--", color=C["dark"])

for i, el in enumerate(elements):
    plt.text(y.iloc[i], y_pred[i], el)

plt.xlabel("DFT ΔG_N")
plt.ylabel("ML Prediction")

plt.title(f"Best Model: {best_name}")

plt.tight_layout()
plt.savefig("plot7_prediction.png", dpi=150)
plt.show()

print("Saved plot7_prediction.png")

# ======================================================
# 7. FEATURE IMPORTANCE
# ======================================================

rf = results["Random Forest"]["model"]

importance = rf.feature_importances_

fi = pd.DataFrame({
    "feature": FEATURES,
    "importance": importance
}).sort_values("importance")

plt.figure(figsize=(8,5))

plt.barh(fi["feature"], fi["importance"], color=C["gold"])

plt.xlabel("Importance")

plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig("plot8_feature_importance.png", dpi=150)

plt.show()

print("Saved plot8_feature_importance.png")

# ======================================================
# 8. SAVE MODEL
# ======================================================

joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nFiles saved:")
print("best_model.pkl")
print("scaler.pkl")

print("\nML training complete.")