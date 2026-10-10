# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from fedcareai.security.privacy import (
    apply_sample_count_visibility,
    detect_sensitive_columns,
    round_sample_count,
    scrub_result_payload,
)


def test_round_sample_count():
    assert round_sample_count(123, nearest=10) == 120
    assert round_sample_count(126, nearest=10) == 130


def test_apply_sample_count_visibility():
    assert apply_sample_count_visibility(123, "exact") == 123
    assert apply_sample_count_visibility(123, "rounded") == 120
    assert apply_sample_count_visibility(123, "hidden") is None


def test_detect_sensitive_columns():
    columns = ["age", "patient_id", "diagnosis", "MRN"]
    found = detect_sensitive_columns(columns)

    assert "patient_id" in found
    assert "MRN" in found


def test_scrub_result_payload():
    payload = {
        "task_id": "task_1",
        "accuracy": 0.91,
        "data_path": "/private/local/file.csv",
        "raw_predictions": [0, 1, 1],
    }

    scrubbed = scrub_result_payload(payload)

    assert "data_path" not in scrubbed
    assert "raw_predictions" not in scrubbed
    assert scrubbed["task_id"] == "task_1"
