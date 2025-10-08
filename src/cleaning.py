import pandas as pd
from unidecode import unidecode
import re

df = pd.read_csv("/Users/sandrineannibal/AI/data/processed/data_flat.csv")

def clean_str(s):
    if pd.isna(s): return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return unidecode(s)

cols_to_clean = [
    "teacher_firstname", "teacher_lastname", "city",
    "course_title", "course_description", "teacher_description"
]

for c in cols_to_clean:
    if c in df.columns:
        df[c] = df[c].apply(clean_str)

df["number_of_stars"] = pd.to_numeric(df["number_of_stars"], errors="coerce")
df.drop_duplicates(inplace=True)

df.to_csv("/Users/sandrineannibal/AI/data/processed/data_clean.csv", index=False, encoding="utf-8")
print("✅ data_clean.csv créé :", df.shape)
