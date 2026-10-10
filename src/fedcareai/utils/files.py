# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path
import json
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def safe_filename(filename: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    cleaned = "".join(char if char in allowed else "_" for char in filename)
    return cleaned.strip("._") or "file"
