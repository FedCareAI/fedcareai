# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

"""Local workflows. No account, platform server, or task cache is required."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fedcareai.core.schema_service import check_tabular_schema
from fedcareai.ml.aggregation import weighted_average_weights
from fedcareai.ml.loaders import load_model
from fedcareai.ml.keras_runner import run_keras_tabular_validation
from fedcareai.ml.sklearn_runner import run_sklearn_tabular_validation
from fedcareai.ml.weights import apply_weights, export_weights
from fedcareai.schemas.data_contract import DataContract


def _framework(framework: str) -> str:
    name = framework.lower()
    if name in {"keras", "tensorflow", "tf"}:
        return "keras"
    if name in {"sklearn", "scikit-learn", "scikit"}:
        return "sklearn"
    raise ValueError(f"Unsupported framework: {framework}")


def _check_data(data_path: Path, contract: DataContract) -> None:
    if contract.input_type != "tabular" or not contract.target_column:
        raise ValueError("Tabular data with a target column is required.")
    check = check_tabular_schema(data_path, contract)
    if not check.passed:
        raise ValueError("; ".join(check.messages))
    missing = {c.name for c in contract.columns} - set(pd.read_csv(data_path, nrows=0).columns)
    if missing:
        raise ValueError(f"Declared model columns must be present: {', '.join(sorted(missing))}")


def _output(path: Path, suffixes: set[str], *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.suffix.lower() not in suffixes:
        raise ValueError(f"Output extension must be one of {sorted(suffixes)}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate_model(
    model_path: Path, data_path: Path, contract: DataContract, framework: str,
    *, load_adjacent_weights: bool = False,
) -> dict[str, Any]:
    """Evaluate a local model against a CSV. Returns metrics and row_count.

    Standalone calls ignore neighboring weights.npz files. The platform adapter
    explicitly enables that convention for downloaded task bundles.
    """
    model_path, data_path = Path(model_path), Path(data_path)
    _check_data(data_path, contract)
    runner = run_keras_tabular_validation if _framework(framework) == "keras" else run_sklearn_tabular_validation
    return runner(
        model_path=model_path, data_path=data_path,
        target_column=contract.target_column, metric=contract.metric,
        feature_columns=[c.name for c in contract.columns if c.name != contract.target_column] or None,
        prediction_type=contract.prediction_type,
        load_adjacent_weights=load_adjacent_weights,
    )


def train_model(
    model_path: Path, data_path: Path, contract: DataContract, framework: str,
    output_path: Path, *, epochs: int = 1, load_adjacent_weights: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train locally and write numeric .npz weights; never upload the CSV."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Epochs must be a positive integer.")
    model_path, data_path = Path(model_path), Path(data_path)
    framework = _framework(framework)
    _check_data(data_path, contract)
    output_path = _output(output_path, {".npz"}, overwrite=overwrite)
    df = pd.read_csv(data_path)
    features = [c.name for c in contract.columns if c.name != contract.target_column]
    x = df[features] if features else df.drop(columns=[contract.target_column])
    y = df[contract.target_column]
    model = load_model(model_path, framework, load_adjacent_weights=load_adjacent_weights)
    if framework == "keras":
        model.fit(x, y, epochs=epochs, verbose=0)
    else:
        if not all(hasattr(model, name) for name in ("partial_fit", "coef_", "intercept_")):
            raise ValueError("Use an already fitted incremental sklearn linear estimator such as SGDClassifier or SGDRegressor. Other sklearn models support validation only.")
        for _ in range(epochs):
            model.partial_fit(x, y)
    export_weights(model, framework, output_path)
    return {"row_count": len(df), "artifact_path": str(output_path), "framework": framework, "epochs": epochs}


def aggregate_updates(
    update_paths: list[Path], output_path: Path, *, sample_counts: list[int] | None = None,
) -> Path:
    """Average compatible numeric updates; omitted counts give equal site weights."""
    updates = []
    for path in update_paths:
        if Path(path).suffix.lower() != ".npz":
            raise ValueError("Updates must be .npz archives.")
        with np.load(Path(path), allow_pickle=False) as archive:
            if len(set(archive.files)) != len(archive.files):
                raise ValueError("Duplicate tensor names.")
            updates.append({name: archive[name] for name in archive.files})
    if not updates:
        raise ValueError("At least one update is required.")
    names = sorted(updates[0])
    if any(update.keys() != updates[0].keys() for update in updates):
        raise ValueError("Model tensor names must match.")
    arrays = weighted_average_weights(
        [[update[name] for name in names] for update in updates],
        [1] * len(updates) if sample_counts is None else sample_counts,
    )
    output_path = _output(output_path, {".npz"})
    np.savez(output_path, **dict(zip(names, arrays)))
    return output_path


def export_model(model_path: Path, weights_path: Path, framework: str, output_path: Path) -> Path:
    """Apply an explicit numeric update to a base model and save the full model."""
    framework = _framework(framework)
    output_path = _output(output_path, {".keras", ".h5"} if framework == "keras" else {".joblib", ".pkl"})
    model = load_model(Path(model_path), framework, load_adjacent_weights=False)
    apply_weights(model, framework, Path(weights_path))
    if framework == "keras":
        model.save(output_path)
    else:
        joblib.dump(model, output_path)
    return output_path
