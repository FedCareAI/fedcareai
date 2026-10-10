# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from fedcareai.ml.keras_runner import run_keras_tabular_validation
from fedcareai.ml.sklearn_runner import run_sklearn_tabular_validation
from fedcareai.ml.validators import regression_metrics
from fedcareai.schemas.data_contract import DataContract


def test_regression_metrics_and_invalid_predictions():
    metrics = regression_metrics([1, 2, 3], [3, 4, 5])
    assert metrics == pytest.approx({"mae": 2, "mse": 4, "rmse": 2, "r2": -5})
    with pytest.raises(ValueError, match="single target"):
        regression_metrics([1, 2], [[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([1, float("nan")], [1, 2])
    with pytest.raises(ValueError, match="two rows"):
        regression_metrics([1], [1], "r2")


def test_sklearn_continuous_predictions(tmp_path):
    x = pd.DataFrame({"feature": [0., 1., 2., 3.]})
    model = LinearRegression().fit(x, [0.2, 2.2, 4.2, 6.2])
    joblib.dump(model, tmp_path / "model.joblib")
    pd.DataFrame({"unused": [99, 99, 99], "target": [2.2, 4.2, 6.2],
                  "feature": [1., 2., 3.]}).to_csv(tmp_path / "data.csv", index=False)
    metrics = run_sklearn_tabular_validation(tmp_path / "model.joblib", tmp_path / "data.csv",
                                           "target", "rmse", ["feature"], "regression")
    assert metrics["rmse"] == pytest.approx(0, abs=1e-10)
    assert metrics["r2"] == pytest.approx(1)
    assert "accuracy" not in metrics


def test_keras_preserves_continuous_outputs(tmp_path, monkeypatch):
    class Model:
        def predict(self, x):
            return np.array([[0.2], [1.8], [2.4]])
    monkeypatch.setattr("fedcareai.ml.keras_runner.load_keras_model", lambda _, **kwargs: Model())
    (tmp_path / "model.keras").touch()
    pd.DataFrame({"feature": [1, 2, 3], "target": [0.2, 1.8, 2.4]}).to_csv(tmp_path / "data.csv", index=False)
    metrics = run_keras_tabular_validation(tmp_path / "model.keras", tmp_path / "data.csv",
                                         "target", "mae", ["feature"], "regression")
    assert metrics["mae"] == 0
    assert "accuracy" not in metrics


def test_contract_rejects_mismatched_metric_or_target():
    assert DataContract().prediction_type == "classification"
    with pytest.raises(ValueError, match="prediction type"):
        DataContract(metric="rmse")
    with pytest.raises(ValueError, match="numeric target"):
        DataContract(prediction_type="regression", metric="rmse", target_column="y",
                     columns=[{"name": "y", "dtype": "category"}])
