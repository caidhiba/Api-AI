import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("/Users/sandrineannibal/AI/data/processed/data_features.csv")

num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

joblib.dump(scaler, "/Users/sandrineannibal/AI/models/scaler.joblib")

df.to_csv("/Users/sandrineannibal/AI/data/processed/data_scaled.csv", index=False)
print("data_scaled.csv prêt :", df.shape)
