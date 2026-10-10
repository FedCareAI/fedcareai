# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path

import pandas as pd

from fedcareai.core.schema_service import check_tabular_schema
from fedcareai.schemas.data_contract import ColumnSpec, DataContract


def test_schema_check_passes(tmp_path: Path):
    data_path = tmp_path / "data.csv"

    pd.DataFrame(
        {
            "age": [10, 12],
            "lesion_size_mm": [1.2, 2.4],
            "diagnosis": [0, 1],
        }
    ).to_csv(data_path, index=False)

    contract = DataContract(
        input_type="tabular",
        target_column="diagnosis",
        columns=[
            ColumnSpec(name="age", dtype="int"),
            ColumnSpec(name="lesion_size_mm", dtype="float"),
            ColumnSpec(name="diagnosis", dtype="int", allowed_values=[0, 1]),
        ],
    )

    result = check_tabular_schema(data_path, contract)

    assert result.passed is True
    assert result.row_count == 2


def test_schema_check_fails_missing_column(tmp_path: Path):
    data_path = tmp_path / "data.csv"

    pd.DataFrame(
        {
            "age": [10, 12],
            "diagnosis": [0, 1],
        }
    ).to_csv(data_path, index=False)

    contract = DataContract(
        input_type="tabular",
        target_column="diagnosis",
        columns=[
            ColumnSpec(name="age", dtype="int"),
            ColumnSpec(name="lesion_size_mm", dtype="float"),
            ColumnSpec(name="diagnosis", dtype="int"),
        ],
    )

    result = check_tabular_schema(data_path, contract)

    assert result.passed is False
    assert "Missing required column: lesion_size_mm" in result.messages
