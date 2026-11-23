"""
pipeline_prefect.py - Orchestration du pipeline ML Diabetes avec Prefect.
"""

import subprocess
import sys

from prefect import flow, task, get_run_logger

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_PATH = "diabetic_data_prepared.csv"
MODEL_PATH = "diabetes_model.joblib"
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def _run(cmd, check=True):
    """Lance une commande shell et affiche sa sortie dans les logs Prefect."""
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.info(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Échec de la commande : {' '.join(cmd)}")
    return result.returncode


# ======================= TASKS : DONNÉES / MODÈLE =======================


@task(name="install-dependencies")
def task_install():
    """1) Installer les dépendances du projet."""
    _run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="prepare-data")
def task_prepare():
    """2) Préparer les données."""
    return prepare_data(DATA_PATH)


@task(name="train-model")
def task_train(X_train, y_train):
    """3) Entraîner le modèle."""
    return train_model(X_train, y_train)


@task(name="save-model")
def task_save(model):
    """4) Sauvegarder le modèle."""
    save_model(model, MODEL_PATH)


@task(name="load-model")
def task_load():
    """5) Charger le modèle."""
    return load_model(MODEL_PATH)


@task(name="evaluate-model")
def task_evaluate(model, X_test, y_test):
    """6) Évaluer le modèle."""
    return evaluate_model(model, X_test, y_test)


# ======================= TASKS : QUALITÉ DU CODE =======================


@task(name="format-code")
def task_format():
    """Formatage du code avec black (uniquement nos 3 fichiers)."""
    _run([sys.executable, "-m", "black", *CODE_FILES])


@task(name="quality-code")
def task_quality():
    """Qualité du code avec flake8 (les remarques sont affichées sans bloquer)."""
    _run(
        [sys.executable, "-m", "flake8", "--max-line-length=120", *CODE_FILES],
        check=False,
    )


@task(name="security-code")
def task_security():
    """Sécurité du code avec bandit (failles moyennes et hautes)."""
    _run([sys.executable, "-m", "bandit", "-ll", *CODE_FILES], check=False)


@task(name="unit-tests")
def task_tests():
    """Exécution des tests unitaires avec pytest."""
    _run([sys.executable, "-m", "pytest", "-q", "test_pipeline.py"])


# ======================= FLOWS =======================


@flow(name="code", log_prints=True)
def flow_code():
    """Formatage -> qualité -> sécurité -> tests unitaires."""
    task_format()
    task_quality()
    task_security()
    task_tests()


@flow(name="all", log_prints=True)
def flow_all():
    """Dépendances -> code -> préparer -> entraîner -> sauvegarder -> évaluer."""
    task_install()
    task_format()
    task_quality()
    task_security()
    task_tests()
    X_train, X_test, y_train, y_test = task_prepare()
    model = task_train(X_train, y_train)
    task_save(model)
    task_evaluate(model, X_test, y_test)


@flow(name="train", log_prints=True)
def flow_train():
    """Préparer les données et entraîner le modèle."""
    X_train, X_test, y_train, y_test = task_prepare()
    model = task_train(X_train, y_train)
    task_save(model)


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    """Charger le modèle et l'évaluer."""
    X_train, X_test, y_train, y_test = task_prepare()
    model = task_load()
    task_evaluate(model, X_test, y_test)


FLOWS = {
    "all": flow_all,
    "code": flow_code,
    "train": flow_train,
    "entrainement": flow_train,
    "evaluate": flow_evaluate,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline Prefect - Diabetes")
    parser.add_argument(
        "--flow", required=True, choices=FLOWS.keys(), help="flow à lancer"
    )
    args = parser.parse_args()
    FLOWS[args.flow]()
