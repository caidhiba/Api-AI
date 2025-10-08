import pandas as pd
import json
import os

# === CHEMINS ===
BASE_DIR = "/Users/sandrineannibal/AI"
INPUT = f"{BASE_DIR}/data/processed/data_scaled.csv"  # ou data_features.csv selon ton étape
OUT_DIR = f"{BASE_DIR}/data/processed"
MODELS_DIR = f"{BASE_DIR}/models"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("📂 Chargement du dataset...")
df = pd.read_csv(INPUT)
print(f"✅ Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# === Colonnes à garder ===
useful_cols = [
                  # Profil enseignant
                  "city", "n_experiences", "n_diplomas", "highest_diploma_level",
                  "teacher_desc_len", "teacher_avg_excl",

                  # Cours
                  "course_title", "course_description", "course_code", "course_level",
                  "course_desc_len", "n_past_courses",

                  # TF-IDF/SVD features
              ] + [f"course_desc_svd_{i}" for i in range(20)] + [
                  # Cible
                  "number_of_stars"
              ]

# === Vérifier la présence des colonnes ===
cols_present = [c for c in useful_cols if c in df.columns]
df = df[cols_present]

print(f"🧹 Colonnes conservées : {len(cols_present)} sur {len(useful_cols)} attendues")
print(f"📊 Colonnes : {list(df.columns)}")

# === Sauvegarde de la liste des features ===
target = "number_of_stars"
feature_cols = [c for c in df.columns if c != target]

features_path = os.path.join(MODELS_DIR, "feature_columns.json")
with open(features_path, "w", encoding="utf-8") as f:
    json.dump(feature_cols, f, ensure_ascii=False, indent=2)

# === Sauvegarde du dataset final ===
final_path = f"{OUT_DIR}/data_for_model.csv"
df.to_csv(final_path, index=False, encoding="utf-8")

print(f"\n data_for_model.csv créé : {df.shape}")
