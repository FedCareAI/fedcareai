# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path

import numpy as np
import pandas as pd

from fedcareai.ml.loaders import load_keras_model
from fedcareai.ml.validators import classification_metrics, regression_metrics


class KerasRunnerError(Exception):
    pass


def _prepare_binary_or_multiclass_predictions(y_pred_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Convert raw Keras model outputs into class predictions and optional score/probability values.

    Returns:
        y_pred: predicted class labels
        y_score: probability/score values for AUC when possible
    """
    y_pred_raw = np.asarray(y_pred_raw)

    # Binary classifier with shape (n, 1)
    if y_pred_raw.ndim == 2 and y_pred_raw.shape[1] == 1:
        y_score = y_pred_raw.ravel()
        y_pred = (y_score >= 0.5).astype(int)
        return y_pred, y_score

    # Multiclass classifier with shape (n, classes)
    if y_pred_raw.ndim == 2 and y_pred_raw.shape[1] > 1:
        y_score = y_pred_raw
        y_pred = np.argmax(y_pred_raw, axis=1)
        return y_pred, y_score

    # Binary classifier with shape (n,)
    if y_pred_raw.ndim == 1:
        y_score = y_pred_raw
        y_pred = (y_score >= 0.5).astype(int)
        return y_pred, y_score

    raise KerasRunnerError(
        f"Unsupported Keras prediction output shape: {y_pred_raw.shape}"
    )


def run_keras_tabular_validation(
    model_path: Path,
    data_path: Path,
    target_column: str,
    metric: str = "accuracy",
    feature_columns: list[str] | None = None,
    prediction_type: str = "classification",
    *,
    load_adjacent_weights: bool = True,
) -> dict:
    """
    Run local validation for a Keras/TensorFlow tabular model.

    Assumptions for MVP:
    - data_path is a CSV file.
    - target_column exists in the CSV.
    - all non-target columns are already in the format expected by the model.
    - preprocessing has either already been done or is simple enough to happen before this call.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    if target_column not in df.columns:
        raise KerasRunnerError(f"Target column not found in data: {target_column}")

    y_true = df[target_column]
    x = df[feature_columns] if feature_columns else df.drop(columns=[target_column])

    model = load_keras_model(model_path, load_adjacent_weights=load_adjacent_weights)
    y_pred_raw = model.predict(x)

    if prediction_type == "regression":
        metrics = regression_metrics(y_true, y_pred_raw, metric)
        metrics["row_count"] = int(len(df))
        return metrics

    y_pred, y_score = _prepare_binary_or_multiclass_predictions(y_pred_raw)

    metrics = classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        requested_metric=metric,
    )
    metrics["row_count"] = int(len(df))

    return metrics
