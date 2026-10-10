# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from typing import Literal

from pydantic import BaseModel, Field, model_validator


ColumnDType = Literal["int", "float", "str", "bool", "category"]
InputType = Literal["tabular", "image"]
MetricName = Literal["accuracy", "auc", "f1", "precision", "recall", "mae", "mse", "rmse", "r2"]


class ColumnSpec(BaseModel):
    """
    Description of one expected column in a collaborator's local dataset.
    """

    name: str
    dtype: ColumnDType
    required: bool = True
    description: str | None = None
    allowed_values: list[str | int | float | bool] | None = None


class ImageSpec(BaseModel):
    """
    Optional image input specification for future image-based models.
    """

    width: int | None = None
    height: int | None = None
    channels: int | None = None
    accepted_formats: list[str] = Field(default_factory=lambda: ["jpg", "jpeg", "png"])


class DataContract(BaseModel):
    """
    Defines what local collaborator data must look like before validation
    or federated aggregation can run.
    """

    input_type: InputType = "tabular"

    # For tabular data
    target_column: str | None = None
    columns: list[ColumnSpec] = Field(default_factory=list)

    # For future image support
    image: ImageSpec | None = None

    # Model/evaluation expectations
    metric: MetricName = "accuracy"
    prediction_type: Literal["classification", "regression"] = "classification"
    preprocessing_instructions: str | None = None

    # Optional metadata
    label_mapping: dict[str, int] | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def compatible_metric(self):
        regression = self.metric in {"mae", "mse", "rmse", "r2"}
        if regression != (self.prediction_type == "regression"):
            raise ValueError("Metric must match the prediction type.")
        if self.prediction_type == "regression":
            target = next((c for c in self.columns if c.name == self.target_column), None)
            if target is None or target.dtype not in {"float", "int"}:
                raise ValueError("Regression requires a numeric target column.")
        return self
