import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib

df = pd.read_csv("/Users/sandrineannibal/AI/data/processed/data_clean.csv")

df["n_past_courses"] = df.groupby("teacher_id")["course_title"].transform("count")
df["course_desc_len"] = df["course_description"].fillna("").str.len()
df["teacher_desc_len"] = df["teacher_description"].fillna("").str.len()

g = df.groupby("teacher_id")["number_of_stars"].agg(["sum", "count"])
df = df.merge(g, left_on="teacher_id", right_index=True, how="left")
df["teacher_avg_excl"] = (df["sum"] - df["number_of_stars"]) / (df["count"] - 1).replace(0, np.nan)

vec = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
Xtf = vec.fit_transform(df["course_description"].fillna(""))
svd = TruncatedSVD(n_components=20, random_state=42)
Xred = svd.fit_transform(Xtf)
for i in range(Xred.shape[1]):
    df[f"course_desc_svd_{i}"] = Xred[:, i]

joblib.dump(vec, "/Users/sandrineannibal/AI/models/tfidf_course_description.joblib")
joblib.dump(svd, "/Users/sandrineannibal/AI/models/svd_course_description.joblib")

df.to_csv("/Users/sandrineannibal/AI/data/processed/data_features.csv", index=False)
print("data_features.csv créé :", df.shape)
