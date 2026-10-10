# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


ResultType = Literal[
    "external_validation",
    "federated_aggregation_update",
    "schema_check",
]


class MetricValue(BaseModel):
    name: str
    value: float | int | str


class ResultPayload(BaseModel):
    task_id: str
    result_type: ResultType
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    local_sample_count: int | None = None
    local_sample_count_visibility: Literal["exact", "rounded", "hidden"] = "rounded"
    artifact_sha256: str | None = None
    signed_by: str | None = None
    signature: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
