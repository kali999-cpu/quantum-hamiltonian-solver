import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# load dataset
df = pd.read_csv("nrr_dataset.csv")

# input features
X = df[[
    "electronegativity",
    "atomic_radius",
    "d_electrons",
    "ionization_energy"
]]

# target variable
y = df["DG_N"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestRegressor(n_estimators=200)

model.fit(X_train, y_train)

# predictions
pred = model.predict(X_test)

# evaluation
print("R2 score:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))