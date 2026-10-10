# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path
from typing import Any

from fedcareai.api import validate_model
from fedcareai.schemas.data_contract import DataContract
from fedcareai.schemas.task import TaskMetadata


class LocalRunnerError(Exception):
    pass


def run_local_validation(
    task: TaskMetadata,
    contract: DataContract,
    model_path: Path,
    data_path: Path,
) -> dict[str, Any]:
    """
    Run local validation using the framework specified by the task/model metadata.
    """
    try:
        return validate_model(model_path, data_path, contract, task.model.framework,
                              load_adjacent_weights=True)
    except ValueError as exc:
        raise LocalRunnerError(str(exc)) from exc
