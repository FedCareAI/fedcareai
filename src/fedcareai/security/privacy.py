# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from typing import Any


SENSITIVE_KEYWORDS = {
    "patient_id",
    "mrn",
    "phn",
    "health_number",
    "name",
    "first_name",
    "last_name",
    "dob",
    "date_of_birth",
    "address",
    "phone",
    "email",
}


class PrivacyError(Exception):
    pass


def round_sample_count(count: int, nearest: int = 10) -> int:
    if nearest <= 0:
        raise PrivacyError("nearest must be positive.")

    return int(round(count / nearest) * nearest)


def apply_sample_count_visibility(
    count: int | None,
    visibility: str,
) -> int | None:
    if count is None:
        return None

    if visibility == "exact":
        return count

    if visibility == "rounded":
        return round_sample_count(count)

    if visibility == "hidden":
        return None

    raise PrivacyError(f"Invalid sample count visibility: {visibility}")


def detect_sensitive_columns(columns: list[str]) -> list[str]:
    found: list[str] = []

    for column in columns:
        normalized = column.strip().lower()

        if normalized in SENSITIVE_KEYWORDS:
            found.append(column)

    return found


def scrub_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Remove obvious local-only fields before submission.
    """
    blocked_keys = {
        "local_file_path",
        "data_path",
        "raw_predictions",
        "patient_ids",
        "row_ids",
    }

    return {
        key: value
        for key, value in payload.items()
        if key not in blocked_keys
    }
