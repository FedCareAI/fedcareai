# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ProjectStatus = Literal["draft", "inviting", "running", "completed", "archived"]


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    institution: str = Field(min_length=1, max_length=200)
    clinical_area: str = ""
    ethics_id: str | None = None


class Project(ProjectCreate):
    id: str
    status: ProjectStatus = "draft"
    owner_user_id: str
    created_at: datetime
    updated_at: datetime
