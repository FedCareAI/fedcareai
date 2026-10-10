# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

import json
import socket

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import SGDClassifier
from typer.testing import CliRunner

from fedcareai.api import aggregate_updates, export_model, train_model, validate_model
from fedcareai.cli.main import app
from fedcareai.schemas.data_contract import DataContract


@pytest.fixture
def local_files(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Standalone workflows must not use the network or platform config")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr("fedcareai.core.config.load_config", forbidden)
    monkeypatch.setattr("fedcareai.core.task_service.get_task_base_dir", forbidden)
    data = pd.DataFrame({"feature": [-2., -1., 1., 2.], "target": [0, 0, 1, 1]})
    model = SGDClassifier(loss="log_loss", random_state=42).fit(data[["feature"]], data["target"])
    joblib.dump(model, tmp_path / "model.joblib")
    data.to_csv(tmp_path / "data.csv", index=False)
    contract = DataContract(target_column="target", columns=[
        {"name": "feature", "dtype": "float"}, {"name": "target", "dtype": "int"},
    ])
    (tmp_path / "contract.json").write_text(contract.model_dump_json(), encoding="utf-8")
    return tmp_path, contract


def test_offline_python_round_trip(local_files):
    path, contract = local_files
    base = path / "model.joblib"
    csv = path / "data.csv"
    assert validate_model(base, csv, contract, "sklearn")["accuracy"] == 1
    for site in ("a", "b"):
        result = train_model(base, csv, contract, "sklearn", path / f"{site}.npz")
        assert result["row_count"] == 4
    aggregate_updates([path / "a.npz", path / "b.npz"], path / "average.npz", sample_counts=[1, 3])
    output = export_model(base, path / "average.npz", "sklearn", path / "final.joblib")
    with np.load(path / "average.npz", allow_pickle=False) as update:
        np.testing.assert_allclose(joblib.load(output).coef_, update["coef"])
    assert validate_model(output, csv, contract, "sklearn")["accuracy"] == 1


def test_offline_cli_round_trip(local_files):
    path, _ = local_files
    runner = CliRunner()
    common = ["--model", str(path / "model.joblib"), "--data", str(path / "data.csv"),
              "--contract", str(path / "contract.json")]
    result = runner.invoke(app, ["local", "validate", *common])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["row_count"] == 4
    result = runner.invoke(app, ["local", "schema-check", *common[2:]])
    assert result.exit_code == 0, result.output
    for site in ("a", "b"):
        result = runner.invoke(app, ["local", "train", *common, "--output", str(path / f"{site}.npz")])
        assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["local", "aggregate", str(path / "a.npz"), str(path / "b.npz"),
                                 "--output", str(path / "average.npz"), "--sample-count", "1", "--sample-count", "3"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["local", "export-model", "--model", str(path / "model.joblib"),
                                 "--weights", str(path / "average.npz"), "--output", str(path / "final.joblib")])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["local", "export-model", "--model", str(path / "model.joblib"),
                                 "--weights", str(path / "average.npz"), "--output", str(path / "final.joblib")])
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_standalone_ignores_implicit_task_weights(local_files):
    path, contract = local_files
    np.savez(path / "weights.npz", coef=np.zeros((1, 1)), intercept=np.array([100.]))
    assert validate_model(path / "model.joblib", path / "data.csv", contract, "sklearn")["accuracy"] == 1
    assert validate_model(path / "model.joblib", path / "data.csv", contract, "sklearn",
                          load_adjacent_weights=True)["accuracy"] == .5


@pytest.mark.parametrize("counts", [[0, 1], [-1, 2], [1], [True, 1]])
def test_invalid_aggregation_counts(tmp_path, counts):
    np.savez(tmp_path / "a.npz", coef=np.array([1.]))
    np.savez(tmp_path / "b.npz", coef=np.array([3.]))
    with pytest.raises(ValueError):
        aggregate_updates([tmp_path / "a.npz", tmp_path / "b.npz"], tmp_path / "out.npz", sample_counts=counts)
    assert not (tmp_path / "out.npz").exists()


@pytest.mark.parametrize("second", [
    {"coef": np.array([[3.]])}, {"intercept": np.array([3.])},
    {"coef": np.array([np.nan])}, {"coef": np.array([], dtype=float)},
    {"coef": np.array(["bad"], dtype=object)},
])
def test_invalid_aggregation_tensors(tmp_path, second):
    np.savez(tmp_path / "a.npz", coef=np.array([1.]))
    np.savez(tmp_path / "b.npz", **second)
    with pytest.raises(ValueError):
        aggregate_updates([tmp_path / "a.npz", tmp_path / "b.npz"], tmp_path / "out.npz")


def test_weighted_and_equal_aggregation(tmp_path):
    np.savez(tmp_path / "a.npz", coef=np.array([1.]))
    np.savez(tmp_path / "b.npz", coef=np.array([3.]))
    inputs = [tmp_path / "a.npz", tmp_path / "b.npz"]
    for name, counts, expected in (("equal", None, 2), ("weighted", [1, 3], 2.5)):
        output = aggregate_updates(inputs, tmp_path / f"{name}.npz", sample_counts=counts)
        with np.load(output, allow_pickle=False) as archive:
            np.testing.assert_allclose(archive["coef"], [expected])


def test_invalid_contract_fails_before_model_loading(local_files):
    path, contract = local_files
    pd.DataFrame({"target": [0, 1]}).to_csv(path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="Missing required column"):
        train_model(path / "missing.joblib", path / "bad.csv", contract, "sklearn", path / "out.npz")
    assert not (path / "out.npz").exists()
