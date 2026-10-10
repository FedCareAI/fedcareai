# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path

import typer

from fedcareai.core.project_service import (
    create_project,
    list_projects,
    get_project,
)

app = typer.Typer(help="Project commands.")


@app.command("list")
def list_() -> None:
    """
    List projects available to the authenticated user.
    """
    projects = list_projects()

    if not projects:
        typer.echo("No projects found.")
        return

    for project in projects:
        typer.echo(
            f"{project['id']} | {project['name']} | "
            f"status={project.get('status', 'unknown')}"
        )


@app.command("create")
def create(
    name: str = typer.Option(..., help="Project name."),
    description: str = typer.Option("", help="Project description."),
    institution: str = typer.Option(..., help="Owner institution."),
    clinical_area: str = typer.Option("", help="Clinical or research area."),
    ethics_id: str | None = typer.Option(None, help="Optional ethics/reference ID."),
) -> None:
    """
    Create a new FedCareAI project.
    """
    project = create_project(
        name=name,
        description=description,
        institution=institution,
        clinical_area=clinical_area,
        ethics_id=ethics_id,
    )

    typer.echo("Project created.")
    typer.echo(f"ID: {project['id']}")
    typer.echo(f"Name: {project['name']}")


@app.command("info")
def info(project_id: str) -> None:
    """
    Show project details.
    """
    project = get_project(project_id)

    typer.echo(f"ID: {project['id']}")
    typer.echo(f"Name: {project['name']}")
    typer.echo(f"Description: {project.get('description', '')}")
    typer.echo(f"Institution: {project.get('institution', '')}")
    typer.echo(f"Status: {project.get('status', '')}")
