import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import joblib
import os

# Optionnels : XGBoost et LightGBM
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

# === CHEMINS ===
BASE_DIR = "/Users/sandrineannibal/AI"
DATA_PATH = f"{BASE_DIR}/data/processed/data_for_model.csv"
MODELS_DIR = f"{BASE_DIR}/models"
os.makedirs(MODELS_DIR, exist_ok=True)

print(" Chargement du dataset...")
df = pd.read_csv(DATA_PATH)
print(f" Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# === PRÉPARATION ===
target = "number_of_stars"
df = df.dropna(subset=[target])
df.fillna("", inplace=True)

X = df.drop(columns=[target])
y = df[target]

# === ENCODAGE DES VARIABLES CATÉGORIELLES ===
text_cols = X.select_dtypes(include=["object"]).columns
print(f" Colonnes texte à encoder : {list(text_cols)}")

for col in text_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    print(f"Encodé : {col} ({len(le.classes_)} valeurs uniques)")
    joblib.dump(le, f"{MODELS_DIR}/le_{col}.joblib")

# === SPLIT TRAIN / VALIDATION ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f" Train: {X_train.shape}, Validation: {X_val.shape}")

# === LISTE DES MODÈLES À TESTER ===
models = {
    "LinearRegression": LinearRegression(),
    "KNN": KNeighborsRegressor(n_neighbors=7),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42),
}

if XGBRegressor:
    models["XGBoost"] = XGBRegressor(
        n_estimators=300, learning_rate=0.1, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )

if LGBMRegressor:
    models["LightGBM"] = LGBMRegressor(
        n_estimators=300, learning_rate=0.1, num_leaves=31, random_state=42
    )

# === ENTRAÎNEMENT & ÉVALUATION ===
results = []
for name, model in models.items():
    print(f"\n Entraînement du modèle : {name}")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)

    print(f" Résultats {name} → MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    results.append({"model": name, "mae": mae, "rmse": rmse, "r2": r2})
    joblib.dump(model, f"{MODELS_DIR}/{name}_model.joblib")

# === CLASSEMENT DES MODÈLES ===
results_df = pd.DataFrame(results).sort_values(by="mae")
print("\n Résultats finaux :")
print(results_df)

results_df.to_csv(f"{MODELS_DIR}/model_results.csv", index=False)
print(f" Résultats sauvegardés → {MODELS_DIR}/model_results.csv")

best_model_name = results_df.iloc[0]["model"]
print(f"\n Meilleur modèle : {best_model_name}")
print(f" Sauvegardé → {MODELS_DIR}/{best_model_name}_model.joblib")
