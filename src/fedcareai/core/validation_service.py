# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path
import json

from fedcareai.core.task_service import get_task_dir
from fedcareai.api import validate_model
from fedcareai.schemas.task import TaskMetadata


def validate_task(task_id: str, data_path: Path) -> dict:
    task_dir = get_task_dir(task_id)
    task_path = task_dir / "task.json"

    if not task_path.exists():
        raise FileNotFoundError(
            f"Task {task_id} has not been pulled. Run: fedcare task pull {task_id}"
        )

    raw_task = json.loads(task_path.read_text(encoding="utf-8"))
    task = TaskMetadata.model_validate(raw_task)

    if task.task_type != 'external_validation':
        raise ValueError('Use task aggregate for federated training tasks.')
    contract = task.data_contract

    from fedcareai.security.privacy import apply_sample_count_visibility
    model_path = task_dir / task.model_filename
    metrics = validate_model(model_path, data_path, contract, task.model.framework,
                             load_adjacent_weights=True)

    count = metrics.pop('row_count', None)
    metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    result = {
        "task_id": task_id,
        "round": task.current_round,
        "local_sample_count": apply_sample_count_visibility(count, 'rounded'),
        "local_sample_count_visibility": 'rounded',
        "result_type": "external_validation",
        "metrics": metrics,
    }

    (task_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result
