from pathlib import Path
import json
import pandas as pd
import hashlib
import re
from unidecode import unidecode
import os

# CHEMINS
BASE = Path("/Users/sandrineannibal/AI")
INPUT = BASE / "data/raw/data_train.json"
OUT_DIR = BASE / "data/processed"
OUT_CSV = OUT_DIR / "data_flat.csv"
OUT_RAW = OUT_DIR / "raw_flat.json"
OUT_TEACHERS = OUT_DIR / "teacher_info.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# OUTILS
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = unidecode(s)
    return s

def make_teacher_id(firstname, lastname, city, description):
    base = f"{normalize_text(firstname)}|{normalize_text(lastname)}|{normalize_text(city)}|{normalize_text(description)[:100]}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:12]

# LECTURE DU JSON
if not INPUT.exists():
    raise FileNotFoundError(f"❌ Impossible de trouver le dataset : {INPUT.resolve()}")

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
teachers_meta = {}

for person in data:
    firstname = person.get("fistname") or person.get("firstname") or ""
    lastname = person.get("lastname") or ""
    city = person.get("city") or ""
    desc = person.get("description") or ""

    teacher_id = make_teacher_id(firstname, lastname, city, desc)

    diplomas = person.get("diplomas", [])
    diploma_titles = [d.get("title", "") for d in diplomas if d]
    diploma_levels = [unidecode(str(d.get("level", ""))).lower() for d in diplomas if d]

    experiences = person.get("experiences", [])
    n_exp = len(experiences)
    n_diplomas = len(diplomas)
    highest_level = max(diploma_levels, default="", key=lambda x: len(x))

    teachers_meta[teacher_id] = {
        "teacher_firstname": firstname,
        "teacher_lastname": lastname,
        "city": city,
        "teacher_description": desc,
        "diploma_titles": " | ".join(diploma_titles),
        "diploma_levels": " | ".join(diploma_levels),
        "n_diplomas": n_diplomas,
        "highest_diploma_level": highest_level,
        "n_experiences": n_exp,
    }

    for course in person.get("pastCourses", []):
        course_title = course.get("title") or ""
        course_desc = course.get("description") or ""
        course_code = course.get("course_code") or ""
        course_level = course.get("course_level") or ""
        number_of_stars = (
            course.get("numberOfStars") or course.get("number_of_stars") or course.get("rating")
        )

        rows.append({
            "teacher_id": teacher_id,
            "teacher_firstname": firstname,
            "teacher_lastname": lastname,
            "city": city,
            "teacher_description": desc,
            "n_experiences": n_exp,
            "n_diplomas": n_diplomas,
            "highest_diploma_level": highest_level,
            "course_title": course_title,
            "course_description": course_desc,
            "course_code": course_code,
            "course_level": course_level,
            "number_of_stars": number_of_stars
        })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False, encoding="utf-8")
df.to_json(OUT_RAW, orient="records", force_ascii=False)
pd.DataFrame.from_dict(teachers_meta, orient="index").reset_index().rename(columns={"index": "teacher_id"}).to_csv(OUT_TEACHERS, index=False)

print("data_flat.csv créé :", df.shape)
