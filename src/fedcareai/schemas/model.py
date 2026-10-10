# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ModelFramework = Literal["keras", "tensorflow", "sklearn", "pytorch"]


class ModelUploadMetadata(BaseModel):
    project_id: str
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    framework: str
    filename: str
    python_version: str = "3.11"
    sha256: str
    size_bytes: int


class ModelMetadata(BaseModel):
    id: str
    project_id: str
    name: str
    description: str = ""
    framework: str
    filename: str
    python_version: str
    sha256: str
    size_bytes: int
    created_at: datetime | None = None
