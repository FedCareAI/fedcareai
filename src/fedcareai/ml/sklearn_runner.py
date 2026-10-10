# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path

import pandas as pd

from fedcareai.ml.loaders import load_sklearn_model
from fedcareai.ml.validators import classification_metrics, regression_metrics


def run_sklearn_tabular_validation(
    model_path: Path,
    data_path: Path,
    target_column: str,
    metric: str,
    feature_columns: list[str] | None = None,
    prediction_type: str = "classification",
    *,
    load_adjacent_weights: bool = True,
) -> dict:
    df = pd.read_csv(data_path)

    y_true = df[target_column]
    x = df[feature_columns] if feature_columns else df.drop(columns=[target_column])

    model = load_sklearn_model(model_path, load_adjacent_weights=load_adjacent_weights)

    y_pred = model.predict(x)

    if prediction_type == "regression":
        metrics = regression_metrics(y_true, y_pred, metric)
        metrics["row_count"] = int(len(df))
        return metrics

    y_score = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)

        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            y_score = probabilities[:, 1]
        else:
            y_score = probabilities

    metrics = classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        requested_metric=metric,
    )
    metrics["row_count"] = int(len(df))

    return metrics
