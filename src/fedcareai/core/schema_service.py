# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from fedcareai.schemas.data_contract import DataContract
from fedcareai.security.privacy import detect_sensitive_columns


class SchemaCheckResult(BaseModel):
    passed: bool
    row_count: int
    messages: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_columns: list[str] = Field(default_factory=list)
    extra_columns: list[str] = Field(default_factory=list)
    sensitive_columns: list[str] = Field(default_factory=list)


class SchemaCheckError(Exception):
    pass


def _check_dtype(series: pd.Series, expected_dtype: str) -> bool:
    """
    Simple dtype compatibility check for CSV-based tabular data.
    """
    if expected_dtype == "int":
        values = pd.to_numeric(series.dropna(), errors='coerce')
        return bool(values.notna().all() and (values % 1 == 0).all())

    if expected_dtype == "float":
        return pd.to_numeric(series.dropna(), errors="coerce").notna().all()

    if expected_dtype in {"str", "category"}:
        return True

    if expected_dtype == "bool":
        values = set(series.dropna().astype(str).str.lower().unique())
        allowed = {"true", "false", "0", "1", "yes", "no"}
        return values.issubset(allowed)

    return True


def check_tabular_schema(data_path: Path, contract: DataContract) -> SchemaCheckResult:
    """
    Check whether a CSV file matches a DataContract.
    """
    if contract.input_type != "tabular":
        raise SchemaCheckError("check_tabular_schema only supports tabular contracts.")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    messages: list[str] = []
    warnings: list[str] = []
    missing_columns: list[str] = []

    required_column_names = [
        column.name for column in contract.columns if column.required
    ]

    if contract.target_column and contract.target_column not in df.columns:
        missing_columns.append(contract.target_column)
        messages.append(f'Missing target column: {contract.target_column}')
    if df.empty:
        messages.append('Dataset is empty.')
    if contract.prediction_type == "regression" and contract.target_column in df.columns:
        target = pd.to_numeric(df[contract.target_column], errors="coerce")
        if not np.isfinite(target.to_numpy(dtype=float)).all():
            messages.append("Regression target must contain finite numbers with no missing values.")
    contract_column_names = [column.name for column in contract.columns]

    for column_name in required_column_names:
        if column_name not in df.columns:
            missing_columns.append(column_name)
            messages.append(f"Missing required column: {column_name}")

    extra_columns = [
        column for column in df.columns if column not in contract_column_names
    ]

    if extra_columns:
        warnings.append(
            "Extra columns found and will be ignored: "
            + ", ".join(extra_columns)
        )

    for column in contract.columns:
        if column.name not in df.columns:
            continue

        if not _check_dtype(df[column.name], column.dtype):
            messages.append(
                f"Column {column.name} could not be converted to expected type: "
                f"{column.dtype}"
            )

        if column.allowed_values is not None:
            observed_values = set(df[column.name].dropna().unique())
            allowed_values = set(column.allowed_values)

            invalid_values = observed_values - allowed_values

            if invalid_values:
                messages.append(
                    f"Column {column.name} has invalid values: "
                    f"{sorted(invalid_values)}"
                )

    sensitive_columns = detect_sensitive_columns(list(df.columns))

    if sensitive_columns:
        warnings.append(
            "Potentially sensitive columns detected: "
            + ", ".join(sensitive_columns)
        )

    passed = len(messages) == 0

    if passed:
        messages.append("Schema check passed.")

    return SchemaCheckResult(
        passed=passed,
        row_count=int(len(df)),
        messages=messages,
        warnings=warnings,
        missing_columns=missing_columns,
        extra_columns=extra_columns,
        sensitive_columns=sensitive_columns,
    )


def check_schema_for_task(task_id: str, data_path: Path) -> SchemaCheckResult:
    """
    Load local task metadata and check local data against its data contract.
    """
    from fedcareai.core.task_service import load_local_task

    task = load_local_task(task_id)
    return check_tabular_schema(data_path=data_path, contract=task.data_contract)
