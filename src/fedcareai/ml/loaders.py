# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path
from typing import Any

import joblib


class ModelLoadError(Exception):
    pass


def load_keras_model(model_path: Path, *, load_adjacent_weights: bool = True) -> Any:
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ModelLoadError(
            "TensorFlow is required to load Keras models. Install tensorflow."
        ) from exc

    model = keras.models.load_model(model_path, safe_mode=True)
    weights = model_path.parent / 'weights.npz'
    if load_adjacent_weights and weights.exists():
        from fedcareai.ml.weights import apply_weights
        apply_weights(model, 'keras', weights)
    return model


def load_sklearn_model(model_path: Path, *, load_adjacent_weights: bool = True) -> Any:
    try:
        model = joblib.load(model_path)
        weights = model_path.parent / 'weights.npz'
        if load_adjacent_weights and weights.exists():
            from fedcareai.ml.weights import apply_weights
            apply_weights(model, 'sklearn', weights)
        return model
    except Exception as exc:
        raise ModelLoadError(f"Could not load sklearn model: {exc}") from exc


def load_model(model_path: Path, framework: str, *, load_adjacent_weights: bool = True) -> Any:
    framework_normalized = framework.lower()

    if framework_normalized in {"keras", "tensorflow", "tf"}:
        return load_keras_model(model_path, load_adjacent_weights=load_adjacent_weights)

    if framework_normalized in {"sklearn", "scikit-learn", "scikit"}:
        return load_sklearn_model(model_path, load_adjacent_weights=load_adjacent_weights)

    raise ModelLoadError(f"Unsupported framework: {framework}")
