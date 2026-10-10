# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y_true, y_pred, requested_metric=None) -> dict[str, float]:
    """Evaluate one continuous target without converting predictions to labels."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float)
    if predicted.ndim == 2 and predicted.shape[1] == 1:
        predicted = predicted[:, 0]
    if predicted.ndim != 1 or predicted.shape != actual.shape:
        raise ValueError("Regression requires one prediction per row for a single target.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Regression targets and predictions must be finite numbers.")
    mse = float(mean_squared_error(actual, predicted))
    metrics = {"mae": float(mean_absolute_error(actual, predicted)),
               "mse": mse, "rmse": float(np.sqrt(mse))}
    if len(actual) >= 2:
        metrics["r2"] = float(r2_score(actual, predicted))
    elif requested_metric == "r2":
        raise ValueError("R² requires at least two rows.")
    return metrics


def classification_metrics(
    y_true,
    y_pred,
    y_score=None,
    requested_metric: str | None = None,
) -> dict[str, float]:
    """
    Compute common classification metrics.

    y_score is used for AUC when available.
    """
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_score is not None:
        try:
            unique_classes = np.unique(y_true)

            if len(unique_classes) == 2:
                metrics["auc"] = float(roc_auc_score(y_true, y_score))
            else:
                metrics["auc"] = float(
                    roc_auc_score(y_true, y_score, multi_class="ovr")
                )
        except Exception:
            pass

    if requested_metric and requested_metric not in metrics:
        metrics[f"requested_metric_{requested_metric}"] = "not_available"

    return metrics
