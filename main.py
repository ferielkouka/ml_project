"""
main.py - Exécute les étapes du pipeline via des arguments CLI.

Exemples :
    python main.py --prepare
    python main.py --train
    python main.py --evaluate
    python main.py --all
"""

import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML Diabetes - Random Forest")
    parser.add_argument(
        "--data", default="diabetic_data_prepared.csv", help="chemin du CSV préparé"
    )
    parser.add_argument(
        "--model-path", default="diabetes_model.joblib", help="fichier du modèle"
    )
    parser.add_argument("--prepare", action="store_true", help="préparer les données")
    parser.add_argument(
        "--train", action="store_true", help="entraîner et sauvegarder le modèle"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="charger et évaluer le modèle"
    )
    parser.add_argument("--all", action="store_true", help="toutes les étapes")
    args = parser.parse_args()

    if not (args.prepare or args.train or args.evaluate or args.all):
        parser.print_help()
        return

    X_train, X_test, y_train, y_test = prepare_data(args.data)

    if args.train or args.all:
        model = train_model(X_train, y_train)
        save_model(model, args.model_path)

    if args.evaluate or args.all:
        model = load_model(args.model_path)
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
