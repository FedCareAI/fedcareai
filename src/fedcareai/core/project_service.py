# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from fedcareai.core.api_client import FedCareApiClient
from fedcareai.schemas.project import ProjectCreate


def list_projects() -> list[dict]:
    client = FedCareApiClient()
    response = client.get("/projects")
    return response if isinstance(response, list) else response.get("projects", response.get("data", []))


def get_project(project_id: str) -> dict:
    client = FedCareApiClient()
    return client.get(f"/projects/{project_id}")


def create_project(
    name: str,
    description: str,
    institution: str,
    clinical_area: str = "",
    ethics_id: str | None = None,
) -> dict:
    payload = ProjectCreate(
        name=name,
        description=description,
        institution=institution,
        clinical_area=clinical_area,
        ethics_id=ethics_id,
    )

    client = FedCareApiClient()
    return client.post("/projects", json=payload.model_dump(exclude_none=True))
