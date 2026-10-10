# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path
import json
from fedcareai.api import train_model
from fedcareai.core.task_service import get_task_dir, load_local_task, get_local_model_path
from fedcareai.security.privacy import apply_sample_count_visibility
from fedcareai.utils.hashing import sha256_file


class AggregationServiceError(Exception):
    pass


def aggregate_task(task_id: str, data_path: Path, epochs: int = 1,
                   sample_count_visibility: str = 'rounded') -> dict:
    task = load_local_task(task_id)
    if task.task_type != 'federated_aggregation':
        raise AggregationServiceError('This is not a federated aggregation task.')
    if task.status == 'completed':
        raise AggregationServiceError('Task is completed.')
    if epochs < 1:
        raise AggregationServiceError('Epochs must be positive.')
    model_path = get_local_model_path(task_id)
    framework = task.model.framework.lower()
    artifact_path = get_task_dir(task_id) / 'local_weights.npz'
    try:
        training = train_model(model_path, data_path, task.data_contract, framework,
                               artifact_path, epochs=epochs, load_adjacent_weights=True,
                               overwrite=True)
    except ValueError as exc:
        raise AggregationServiceError(str(exc)) from exc
    visible_count = apply_sample_count_visibility(training['row_count'], sample_count_visibility)
    if visible_count == 0:
        raise AggregationServiceError('Rounded sample count is zero. Use --sample-count-visibility exact or hidden.')
    result = {'task_id': task_id, 'round': task.current_round,
              'result_type': 'federated_aggregation_update', 'local_sample_count': visible_count,
              'local_sample_count_visibility': sample_count_visibility,
              'artifact_path': str(artifact_path), 'artifact_sha256': sha256_file(artifact_path),
              'metadata': {'framework': framework, 'epochs': epochs, 'raw_data_uploaded': False,
                           'weight_format': 'numeric_npz'}}
    (get_task_dir(task_id) / 'aggregation_result.json').write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_local_aggregation_result(task_id: str) -> dict:
    path = get_task_dir(task_id) / 'aggregation_result.json'
    if not path.exists():
        raise FileNotFoundError('Run fedcare task aggregate first.')
    return json.loads(path.read_text(encoding="utf-8"))
