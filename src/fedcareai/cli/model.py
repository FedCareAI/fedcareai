# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path

import typer

from fedcareai.core.model_service import upload_model, list_models, get_model

app = typer.Typer(help="Model commands.")


@app.command("upload")
def upload(
    project_id: str = typer.Option(..., help="Project ID."),
    file: Path = typer.Option(..., exists=True, readable=True, help="Model file path."),
    framework: str = typer.Option(..., help="keras, sklearn, pytorch, etc."),
    name: str = typer.Option(..., help="Model display name."),
    description: str = typer.Option("", help="Model description."),
    python_version: str = typer.Option("3.11", help="Python version used."),
) -> None:
    """
    Upload a model artifact for a project.
    """
    model = upload_model(
        project_id=project_id,
        file_path=file,
        framework=framework,
        name=name,
        description=description,
        python_version=python_version,
    )

    typer.echo("Model uploaded.")
    typer.echo(f"ID: {model['id']}")
    typer.echo(f"Name: {model['name']}")


@app.command("list")
def list_(
    project_id: str = typer.Option(..., help="Project ID."),
) -> None:
    """
    List models for a project.
    """
    models = list_models(project_id)

    if not models:
        typer.echo("No models found.")
        return

    for model in models:
        typer.echo(
            f"{model['id']} | {model['name']} | "
            f"framework={model.get('framework', 'unknown')}"
        )


@app.command("info")
def info(model_id: str) -> None:
    """
    Show model details.
    """
    model = get_model(model_id)

    typer.echo(f"ID: {model['id']}")
    typer.echo(f"Name: {model['name']}")
    typer.echo(f"Framework: {model.get('framework', '')}")
    typer.echo(f"File: {model.get('filename', '')}")
