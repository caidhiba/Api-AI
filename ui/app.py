import streamlit as st
import requests
import json

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(page_title="AI Satisfaction Predictor", page_icon="🤖", layout="centered")

API_URL = "https://votre-api-deployée.onrender.com/predict"  # à remplacer par le vrai lien
API_KEY = "votre_api_key_si_besoin"

st.title("🎓 AI Satisfaction Predictor")
st.markdown("Prédisez la satisfaction d’un cours avant même qu’il ait lieu !")

# -----------------------------
# FORMULAIRE
# -----------------------------
st.header("👩‍🏫 Profil du professeur")
col1, col2 = st.columns(2)
firstname = col1.text_input("Prénom", "")
lastname = col2.text_input("Nom", "")
city = st.text_input("Ville", "")
teacher_description = st.text_area("Description du professeur",
    "")
n_experiences = st.number_input("Nombre d’expériences", min_value=0, value=7)
n_diplomas = st.number_input("Nombre de diplômes", min_value=0, value=2)
highest_diploma_level = st.selectbox("Niveau du diplôme le plus élevé", ["licence", "master", "doctorat"])

st.header("📚 Détails du cours")
course_title = st.text_input("Titre du cours", "Introduction au Machine Learning")
course_description = st.text_area("Description du cours",
    "Cours pratique sur sklearn, modèles et pipelines de data science.")
course_level = st.selectbox("Niveau du cours", ["débutant", "intermédiaire", "avancé"])

# -----------------------------
# BOUTON DE PREDICTION
# -----------------------------
if st.button("Prédire la satisfaction 🎯"):
    payload = {
        "teacher": {
            "teacher_firstname": firstname,
            "teacher_lastname": lastname,
            "teacher_description": teacher_description,
            "city": city,
            "n_experiences": n_experiences,
            "n_diplomas": n_diplomas,
            "highest_diploma_level": highest_diploma_level
        },
        "course": {
            "course_title": course_title,
            "course_description": course_description,
            "course_level": course_level
        }
    }

    try:
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["x-api-key"] = API_KEY

        response = requests.post(API_URL, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            result = response.json()
            rating = result.get("predicted_rating", None)
            if rating is not None:
                st.success(f"⭐️ Note de satisfaction prédite : **{rating:.2f} / 5**")
                if rating >= 4:
                    st.balloons()
                st.progress(min(rating / 5, 1.0))
            else:
                st.warning("Aucune note reçue de l’API.")
        else:
            st.error(f"Erreur API ({response.status_code}) : {response.text}")
    except Exception as e:
        st.error(f"Erreur lors de la requête : {e}")
