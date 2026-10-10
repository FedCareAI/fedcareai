# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

"""Standalone workflows with explicit local inputs and outputs."""

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Annotated

import typer

from fedcareai.api import aggregate_updates, export_model, train_model, validate_model
from fedcareai.core.schema_service import check_tabular_schema
from fedcareai.ml.loaders import ModelLoadError
from fedcareai.schemas.data_contract import DataContract

app = typer.Typer(help="Run local workflows without a platform account.")
InputFile = Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)]


@contextmanager
def _errors():
    try:
        yield
    except (ValueError, OSError, ModelLoadError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _contract(path: Path) -> DataContract:
    return DataContract.model_validate_json(path.read_text(encoding="utf-8"))


@app.command("schema-check")
def schema_check(data: InputFile, contract: InputFile) -> None:
    with _errors():
        result = check_tabular_schema(data, _contract(contract))
        typer.echo(result.model_dump_json(indent=2))
        if not result.passed:
            raise typer.Exit(code=1)


@app.command("validate")
def validate(model: InputFile, data: InputFile, contract: InputFile, framework: str = "sklearn") -> None:
    with _errors():
        typer.echo(json.dumps(validate_model(model, data, _contract(contract), framework), indent=2))


@app.command("train")
def train(
    model: InputFile, data: InputFile, contract: InputFile,
    output: Annotated[Path, typer.Option()], framework: str = "sklearn", epochs: int = 1,
) -> None:
    with _errors():
        typer.echo(json.dumps(train_model(model, data, _contract(contract), framework, output, epochs=epochs), indent=2))


@app.command("aggregate")
def aggregate(
    updates: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option()],
    sample_count: Annotated[list[int] | None, typer.Option(help="Repeat once per update, in the same order. Omit for equal site weighting.")] = None,
) -> None:
    with _errors():
        typer.echo(aggregate_updates(updates, output, sample_counts=sample_count or None))


@app.command("export-model")
def export(
    model: InputFile, weights: InputFile, output: Annotated[Path, typer.Option()], framework: str = "sklearn",
) -> None:
    with _errors():
        typer.echo(export_model(model, weights, framework, output))
