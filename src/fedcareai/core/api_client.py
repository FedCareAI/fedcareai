# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path
from typing import Any

import requests

from fedcareai.core.config import load_config


class FedCareApiError(Exception):
    pass


class FedCareApiClient:
    def __init__(self) -> None:
        self.config = load_config()

        if not self.config.access_token:
            raise FedCareApiError("You are not logged in. Run: fedcare auth login")

        self.base_url = self.config.api_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.config.access_token}",
        }

    def get(self, path: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}{path}",
            headers=self.headers,
            timeout=30,
        )
        return self._handle_response(response)

    def post(self, path: str, json: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}{path}",
            headers=self.headers,
            json=json,
            timeout=30,
        )
        return self._handle_response(response)

    def post_file(
        self,
        path: str,
        file_path: Path,
        field_name: str = "file",
        data: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        with file_path.open("rb") as file:
            response = requests.post(
                f"{self.base_url}{path}",
                headers=self.headers,
                files={field_name: (file_path.name, file)},
                data=data or {},
                timeout=120,
            )

        return self._handle_response(response)

    def download_file(self, path: str, output_path: Path) -> None:
        response = requests.get(
            f"{self.base_url}{path}",
            headers=self.headers,
            timeout=60,
            stream=True,
        )

        if response.status_code >= 400:
            raise FedCareApiError(response.text)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    def _handle_response(self, response: requests.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            raise FedCareApiError(response.text)

        if not response.content:
            return {}

        return response.json()
