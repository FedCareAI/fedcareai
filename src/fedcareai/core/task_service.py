# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path
import json

from fedcareai.core.api_client import FedCareApiClient
from fedcareai.core.config import get_task_base_dir
from fedcareai.schemas.task import TaskMetadata


def get_task_dir(task_id: str) -> Path:
    """
    Return the local directory for a task.
    """
    if not task_id or Path(task_id).name != task_id or '/' in task_id or chr(92) in task_id or task_id in {'.', '..'}:
        raise ValueError('Invalid task ID')
    return get_task_base_dir() / task_id


def get_local_task_path(task_id: str) -> Path:
    """
    Return the local task.json path.
    """
    return get_task_dir(task_id) / "task.json"


def list_tasks() -> list[dict]:
    """
    List tasks available to the authenticated user.
    """
    client = FedCareApiClient()
    response = client.get("/tasks")

    if isinstance(response, list):
        return response

    return response.get("tasks", [])


def get_remote_task(task_id: str) -> dict:
    """
    Get task metadata from the API.
    """
    client = FedCareApiClient()
    return client.get(f"/tasks/{task_id}")


def pull_task(task_id: str) -> Path:
    """
    Download task metadata and model artifact to local storage.

    Creates:
      ~/.fedcareai/tasks/{task_id}/task.json
      ~/.fedcareai/tasks/{task_id}/{model_filename}
    """
    client = FedCareApiClient()
    task = client.get(f"/tasks/{task_id}")

    task_metadata = TaskMetadata.model_validate(task)

    task_dir = get_task_dir(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)

    from fedcareai.utils.hashing import sha256_file
    model_filename = task_metadata.model_filename or task_metadata.model.filename
    if Path(model_filename).name != model_filename or '/' in model_filename or chr(92) in model_filename:
        raise ValueError('Invalid model filename')
    model_path = task_dir / model_filename
    client.download_file(path=f'/tasks/{task_id}/model', output_path=model_path)
    if sha256_file(model_path) != task_metadata.model.sha256:
        model_path.unlink(missing_ok=True)
        raise ValueError('Downloaded model checksum mismatch')
    weights_path = task_dir / 'weights.npz'
    if task_metadata.weights_available:
        client.download_file(path=f'/tasks/{task_id}/weights', output_path=weights_path)
        if sha256_file(weights_path) != task_metadata.weights_sha256:
            weights_path.unlink(missing_ok=True)
            raise ValueError('Downloaded weights checksum mismatch')
    else:
        weights_path.unlink(missing_ok=True)
    # Old round results cannot be submitted after refreshing the task.
    for filename in ('result.json', 'aggregation_result.json'):
        (task_dir / filename).unlink(missing_ok=True)
    (task_dir / 'task.json').write_text(task_metadata.model_dump_json(indent=2), encoding="utf-8")

    return task_dir


def load_local_task(task_id: str) -> TaskMetadata:
    """
    Load a previously pulled task from local storage.
    """
    task_path = get_local_task_path(task_id)

    if not task_path.exists():
        raise FileNotFoundError(
            f"Task {task_id} has not been pulled. Run: fedcare task pull {task_id}"
        )

    return TaskMetadata.model_validate_json(task_path.read_text(encoding="utf-8"))


def save_local_task(task: TaskMetadata) -> None:
    """
    Save task metadata locally.
    """
    task_dir = get_task_dir(task.id)
    task_dir.mkdir(parents=True, exist_ok=True)

    task_path = task_dir / "task.json"
    task_path.write_text(task.model_dump_json(indent=2), encoding="utf-8")


def get_local_model_path(task_id: str) -> Path:
    """
    Return the expected local model path for a pulled task.
    """
    task = load_local_task(task_id)
    return get_task_dir(task_id) / task.model_filename


def mark_task_status(task_id: str, status: str) -> None:
    """
    Update local task status.

    This is local-only. The server status should be updated separately if needed.
    """
    task = load_local_task(task_id)
    task.status = status  # type: ignore[assignment]
    save_local_task(task)


def create_task(
    project_id: str,
    model_id: str,
    name: str,
    task_type: str,
    data_contract: dict,
    min_collaborators: int = 1,
    rounds: int | None = None,
) -> dict:
    """
    Owner-side API call to create a task.
    """
    payload = {
        "project_id": project_id,
        "model_id": model_id,
        "name": name,
        "task_type": task_type,
        "data_contract": data_contract,
        "min_collaborators": min_collaborators,
        "rounds": rounds,
    }

    client = FedCareApiClient()
    return client.post("/tasks", json=payload)
