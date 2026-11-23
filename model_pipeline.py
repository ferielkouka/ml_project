"""
model_pipeline.py - Pipeline de classification pour le projet Diabetes.
Objectif : prédire la réadmission d'un patient dans les 30 jours (readmit_30).
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET = "readmit_30"
RANDOM_STATE = 42


def prepare_data(
    csv_path="diabetic_data_prepared.csv", test_size=0.20, random_state=RANDOM_STATE
):
    """
    Charge le CSV préparé et sépare les variables explicatives de la cible.
    La colonne 'segment' est exclue car elle est dérivée des variables explicatives.
    Le découpage est stratifié pour conserver le déséquilibre des classes.

    Returns:
        X_train, X_test, y_train, y_test
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier CSV introuvable : {csv_path}")

    data = pd.read_csv(csv_path)

    if TARGET not in data.columns:
        raise ValueError(f"La colonne cible '{TARGET}' est absente du CSV.")
    if data[TARGET].isna().any():
        raise ValueError(
            f"La colonne cible '{TARGET}' contient des valeurs manquantes."
        )

    # On retire la cible et le segment pour éviter une fuite d'information.
    columns_to_drop = [TARGET]
    if "segment" in data.columns:
        columns_to_drop.append("segment")

    X = data.drop(columns=columns_to_drop)
    y = data[TARGET].astype(int)

    if X.empty:
        raise ValueError("Aucune variable explicative disponible.")
    if y.nunique() != 2:
        raise ValueError("La cible doit contenir exactement deux classes.")

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Variables non numériques détectées : {non_numeric}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Données chargées : {data.shape}")
    print(f"Variables explicatives : {X.shape[1]}")
    print(f"Train : {X_train.shape[0]} lignes | positifs : {y_train.mean():.2%}")
    print(f"Test  : {X_test.shape[0]} lignes | positifs : {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, random_state=RANDOM_STATE, cv_splits=3):
    """
    Entraîne une forêt aléatoire avec une recherche sur grille (GridSearchCV).
    La pondération balanced_subsample prend en compte le déséquilibre
    entre les patients réadmis et non réadmis.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    random_forest = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )

    param_grid = {
        "max_depth": [10, 16],
        "min_samples_leaf": [10, 40],
    }

    search = GridSearchCV(
        estimator=random_forest,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        verbose=1,
    )

    print("Début de l'entraînement et de la validation croisée...")
    search.fit(X_train, y_train)

    print("Meilleurs paramètres :", search.best_params_)
    print(f"Meilleur ROC-AUC en validation croisée : {search.best_score_:.4f}")

    return search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur le jeu de test et affiche les métriques principales.

    Returns:
        dict contenant les métriques calculées.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Le modèle doit fournir la méthode predict_proba().")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "PR-AUC": average_precision_score(y_test, probabilities),
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced accuracy": balanced_accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
    }

    print("\n========== ÉVALUATION DU MODÈLE ==========")
    for metric_name, value in metrics.items():
        print(f"{metric_name:<20}: {value:.4f}")

    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, predictions))

    print("\nRapport de classification :")
    print(classification_report(y_test, predictions, zero_division=0))

    return metrics


def save_model(model, model_path="diabetes_model.joblib"):
    """Sauvegarde le modèle entraîné au format joblib."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé dans : {model_path}")
    return model_path


def load_model(model_path="diabetes_model.joblib"):
    """Charge un modèle précédemment sauvegardé avec joblib."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    model = joblib.load(model_path)
    print(f"Modèle chargé depuis : {model_path}")
    return model


def predict(model, features):
    """
    Prédit la réadmission pour UN patient.
    features : dict {nom_variable: valeur}. Les variables absentes valent 0.
    """
    if not features:
        raise ValueError("Aucune variable fournie.")

    expected = list(model.feature_names_in_)
    unknown = [name for name in features if name not in expected]
    if unknown:
        raise ValueError(f"Variables inconnues : {unknown}")

    X = pd.DataFrame([features]).reindex(columns=expected, fill_value=0)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0, 1])

    return {
        "prediction": prediction,
        "probabilite_readmission": round(probability, 4),
        "resultat": "Réadmis dans les 30 jours" if prediction == 1 else "Non réadmis",
    }
