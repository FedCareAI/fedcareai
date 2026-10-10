# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path

import pytest

from fedcareai.core.validation_service import validate_task


def test_validate_task_requires_pulled_task(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "fedcareai.core.config.CONFIG_DIR",
        tmp_path / ".fedcareai",
    )
    monkeypatch.setattr(
        "fedcareai.core.config.CONFIG_FILE",
        tmp_path / ".fedcareai" / "config.json",
    )

    with pytest.raises(FileNotFoundError):
        validate_task("missing_task", tmp_path / "data.csv")
