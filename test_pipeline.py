"""Tests unitaires du pipeline ML Diabetes."""
import numpy as np
import pandas as pd
import pytest

from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model


@pytest.fixture
def fake_csv(tmp_path):
    """Crée un petit CSV préparé (200 lignes) pour tester rapidement."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.rand(200, 5), columns=[f"x{i}" for i in range(5)])
    df["readmit_30"] = [0] * 160 + [1] * 40
    df["segment"] = 0
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_prepare_data(fake_csv):
    X_train, X_test, y_train, y_test = prepare_data(fake_csv)
    assert len(X_train) == 160 and len(X_test) == 40
    assert "readmit_30" not in X_train.columns
    assert "segment" not in X_train.columns


def test_prepare_data_fichier_absent():
    with pytest.raises(FileNotFoundError):
        prepare_data("fichier_inexistant.csv")


def test_train_save_load_evaluate(fake_csv, tmp_path):
    X_train, X_test, y_train, y_test = prepare_data(fake_csv)
    model = train_model(X_train, y_train)
    path = save_model(model, tmp_path / "model.joblib")
    loaded = load_model(path)
    metrics = evaluate_model(loaded, X_test, y_test)
    assert 0.0 <= metrics["ROC-AUC"] <= 1.0
