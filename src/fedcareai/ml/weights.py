# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path
import numpy as np


class WeightError(ValueError):
    pass


def export_weights(model, framework: str, path: Path) -> None:
    if framework in {'keras', 'tensorflow', 'tf'}:
        weights = {f'weight_{i}': value for i, value in enumerate(model.get_weights())}
    elif hasattr(model, 'coef_') and hasattr(model, 'intercept_') and hasattr(model, 'partial_fit'):
        weights = {'coef': np.asarray(model.coef_), 'intercept': np.asarray(model.intercept_)}
    else:
        raise WeightError('Federated training requires Keras or an incremental sklearn linear estimator with coef_ and intercept_.')
    if not weights or any(value.dtype.kind not in 'fi' or not np.isfinite(value).all() for value in weights.values()):
        raise WeightError('Model weights must be finite numeric arrays.')
    np.savez(path, **weights)


def apply_weights(model, framework: str, path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        weights = {key: archive[key] for key in archive.files}
    if any(v.dtype.kind not in 'fi' or not np.isfinite(v).all() for v in weights.values()):
        raise WeightError('Invalid aggregate weights')
    if framework in {'keras', 'tensorflow', 'tf'}:
        original = model.get_weights()
        expected = {f'weight_{i}' for i in range(len(original))}
        if weights.keys() != expected or any(weights[f'weight_{i}'].shape != w.shape for i, w in enumerate(original)):
            raise WeightError('Aggregated tensor shapes do not match the model.')
        model.set_weights([weights[f'weight_{i}'] for i in range(len(original))])
    else:
        if weights.keys() != {'coef', 'intercept'} or not hasattr(model, 'coef_') or not hasattr(model, 'intercept_'):
            raise WeightError('Unsupported sklearn model or weight format.')
        if weights['coef'].shape != model.coef_.shape or weights['intercept'].shape != model.intercept_.shape:
            raise WeightError('Aggregated coefficient shapes do not match the model.')
        model.coef_ = weights['coef'].astype(model.coef_.dtype)
        model.intercept_ = weights['intercept'].astype(model.intercept_.dtype)
