# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path

from pydantic import BaseModel, Field


CONFIG_DIR = Path.home() / ".fedcareai"
CONFIG_FILE = CONFIG_DIR / "config.json"


class FedCareConfig(BaseModel):
    api_url: str = "https://api.fedcare.ai"
    access_token: str | None = None
    user_email: str | None = None

    task_dir: Path = Field(default_factory=lambda: CONFIG_DIR / "tasks")


def load_config() -> FedCareConfig:
    """
    Load local FedCareAI configuration.

    If the user has not logged in yet, return default config.
    """
    if not CONFIG_FILE.exists():
        return FedCareConfig()

    return FedCareConfig.model_validate_json(CONFIG_FILE.read_text(encoding="utf-8"))


def save_config(config: FedCareConfig) -> None:
    """
    Save local FedCareAI configuration.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(config.model_dump_json(indent=2), encoding="utf-8")


def get_task_base_dir() -> Path:
    """
    Return local directory where pulled tasks and results are stored.
    """
    config = load_config()
    config.task_dir.mkdir(parents=True, exist_ok=True)
    return config.task_dir
