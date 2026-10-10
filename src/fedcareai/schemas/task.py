# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from fedcareai.schemas.data_contract import DataContract
from fedcareai.schemas.model import ModelMetadata


TaskType = Literal["external_validation", "federated_aggregation"]
TaskStatus = Literal[
    "draft",
    "invited",
    "accepted",
    "pulled",
    "running_local",
    "submitted",
    "aggregated",
    "completed",
    "failed",
]


class TaskCreate(BaseModel):
    project_id: str
    model_id: str
    name: str = Field(min_length=1, max_length=200)
    task_type: TaskType
    data_contract: DataContract
    min_collaborators: int = 1
    rounds: int | None = None


class TaskMetadata(BaseModel):
    id: str
    project_id: str
    name: str
    task_type: TaskType
    status: TaskStatus = "draft"
    model: ModelMetadata
    model_filename: str = "model.h5"
    data_contract: DataContract
    min_collaborators: int = 1
    rounds: int | None = None
    current_round: int = 1
    weights_available: bool = False
    weights_sha256: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
