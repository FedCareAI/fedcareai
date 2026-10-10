# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path

from fedcareai.core.api_client import FedCareApiClient
from fedcareai.schemas.model import ModelUploadMetadata
from fedcareai.utils.hashing import sha256_file


def upload_model(
    project_id: str,
    file_path: Path,
    framework: str,
    name: str,
    description: str,
    python_version: str,
) -> dict:
    metadata = ModelUploadMetadata(
        project_id=project_id,
        name=name,
        description=description,
        framework=framework,
        filename=file_path.name,
        python_version=python_version,
        sha256=sha256_file(file_path),
        size_bytes=file_path.stat().st_size,
    )

    client = FedCareApiClient()
    return client.post_file(
        path=f"/projects/{project_id}/models",
        file_path=file_path,
        field_name="model_file",
        data={"metadata": metadata.model_dump_json()},
    )


def list_models(project_id: str) -> list[dict]:
    client = FedCareApiClient()
    response = client.get(f"/projects/{project_id}/models")
    return response if isinstance(response, list) else response.get("models", [])


def get_model(model_id: str) -> dict:
    client = FedCareApiClient()
    return client.get(f"/models/{model_id}")
