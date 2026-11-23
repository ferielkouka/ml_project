# ML Project – Prédiction de réadmission (Diabetes)

Projet MLOps : prédire si un patient diabétique sera **réadmis dans les 30 jours** (`readmit_30`) avec un **Random Forest** (GridSearchCV).

## Structure

| Fichier | Rôle |
|---|---|
| `model_pipeline.py` | Préparation des données, entraînement, évaluation, sauvegarde/chargement, `predict()` |
| `main.py` | Lancement en ligne de commande (`--prepare`, `--train`, `--evaluate`, `--all`) |
| `pipeline_prefect.py` | Orchestration Prefect (tasks + flows `all`, `train`, `evaluate`, `code`, `api`) |
| `deploiement_prefect.py` | Déploiements Prefect planifiés (tous les jours à 2h) |
| `app.py` | API REST FastAPI (`/`, `/features`, `/predict`) |
| `test_pipeline.py` | Tests unitaires (pytest) |
| `requirements.txt` | Dépendances |

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Le fichier de données `diabetic_data_prepared.csv` et le modèle `diabetes_model.joblib` ne sont pas versionnés (voir `.gitignore`).

## Utilisation

### Pipeline ML

```bash
python main.py --all
```

### Prefect

```bash
prefect server start                      # terminal 1 : UI sur http://127.0.0.1:4200
python pipeline_prefect.py --flow all     # git pull, qualité du code, entraînement, évaluation
python pipeline_prefect.py --flow train
python pipeline_prefect.py --flow evaluate
python pipeline_prefect.py --flow code    # black, flake8, bandit, pytest
```

Déploiement planifié :

```bash
python deploiement_prefect.py             # terminal 2 : worker
prefect deployment run 'all/ml-pipeline-all'   # terminal 3 : lancement manuel
```

### API FastAPI

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
# ou
python pipeline_prefect.py --flow api
```

Documentation interactive : http://127.0.0.1:8000/docs

Exemple de requête :

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"gender_male": 1, "time_in_hospital": 5, "num_medications": 15}}'
```

Réponse :

```json
{"prediction": 0, "probabilite_readmission": 0.4162, "resultat": "Non réadmis"}
```

Les variables absentes valent 0. Une variable inconnue renvoie une erreur **400**.

## Qualité du code

```bash
black model_pipeline.py main.py pipeline_prefect.py
flake8 --max-line-length=120 model_pipeline.py main.py pipeline_prefect.py
bandit -ll model_pipeline.py main.py pipeline_prefect.py
pytest -q test_pipeline.py
```
