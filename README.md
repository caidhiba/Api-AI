# AI Satisfaction Predictor - HETIC Challenge

API de prédiction de satisfaction des apprenants pour le challenge HETIC x Educentre.

## 🚀 Déploiement

### Sur Render.com
1. Forkez ce repository
2. Allez sur [Render.com](https://render.com)
3. Connectez votre compte GitHub
4. Créez un nouveau "Web Service"
5. Sélectionnez ce repository
6. Utilisez la configuration automatique

### Localement

cd api
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000

📊 Endpoints

GET / - Page d'accueil

GET /health - Statut de l'API

POST /api/predict - Prédiction de satisfaction

GET /api/info - Informations sur l'API

