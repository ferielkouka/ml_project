"""
app.py - API REST (FastAPI) pour prédire la réadmission d'un patient diabétique.
Lancement : uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_pipeline import load_model, predict

MODEL_PATH = "diabetes_model.joblib"

app = FastAPI(
    title="API Diabetes - Réadmission",
    description="Prédit si un patient diabétique sera réadmis dans les 30 jours (Random Forest).",
    version="1.0",
)

# Chargement du modèle une seule fois, au démarrage de l'API
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    model = None


class Patient(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Variables du patient. Les variables absentes valent 0.",
        examples=[
            {
                "gender_male": 1,
                "time_in_hospital": 5,
                "num_lab_procedures": 45,
                "num_procedures": 1,
                "num_medications": 15,
                "number_diagnoses": 8,
            }
        ],
    )


@app.get("/")
def accueil():
    """Vérifie que l'API fonctionne."""
    return {"message": "API Diabetes OK", "modele_charge": model is not None}


@app.get("/features")
def liste_features():
    """Liste les 95 variables attendues par le modèle."""
    if model is None:
        raise HTTPException(status_code=503, detail=f"Modèle introuvable : {MODEL_PATH}")
    return {"nombre": len(model.feature_names_in_), "features": list(model.feature_names_in_)}


@app.post("/predict")
def route_predict(patient: Patient):
    """Prédit la réadmission à 30 jours pour un patient."""
    if model is None:
        raise HTTPException(status_code=503, detail=f"Modèle introuvable : {MODEL_PATH}")
    try:
        return predict(model, patient.features)
    except ValueError as erreur:
        raise HTTPException(status_code=400, detail=str(erreur))
