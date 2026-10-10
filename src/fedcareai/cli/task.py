# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from pathlib import Path

import typer

from fedcareai.core.aggregation_service import aggregate_task
from fedcareai.core.result_service import preview_submission, submit_result
from fedcareai.core.schema_service import check_schema_for_task
from fedcareai.core.task_service import list_tasks, pull_task
from fedcareai.core.validation_service import validate_task

app = typer.Typer(help="Task commands.")


@app.command("list")
def list_() -> None:
    tasks = list_tasks()

    if not tasks:
        typer.echo("No tasks found.")
        return

    for task in tasks:
        typer.echo(
            f"{task['id']} | {task.get('name', '')} | "
            f"type={task.get('task_type', '')} | "
            f"status={task.get('status', '')}"
        )


@app.command("pull")
def pull(task_id: str) -> None:
    task_dir = pull_task(task_id)
    typer.echo(f"Task pulled to: {task_dir}")


@app.command("schema-check")
def schema_check(
    task_id: str,
    data: Path = typer.Option(..., exists=True, readable=True),
) -> None:
    result = check_schema_for_task(task_id=task_id, data_path=data)

    typer.echo(f"Rows detected: {result.row_count}")

    for message in result.messages:
        typer.echo(message)

    for warning in result.warnings:
        typer.echo(f"Warning: {warning}")

    if not result.passed:
        raise typer.Exit(code=1)


@app.command("validate")
def validate(
    task_id: str,
    data: Path = typer.Option(..., exists=True, readable=True),
) -> None:
    result = validate_task(task_id=task_id, data_path=data)

    typer.echo("Validation completed.")
    for name, value in result.get("metrics", {}).items():
        typer.echo(f"{name}: {value}")


@app.command("aggregate")
def aggregate(
    task_id: str,
    data: Path = typer.Option(..., exists=True, readable=True),
    epochs: int = typer.Option(1, help="Number of local training epochs."),
    sample_count_visibility: str = typer.Option(
        "rounded",
        help="exact, rounded, or hidden",
    ),
) -> None:
    result = aggregate_task(
        task_id=task_id,
        data_path=data,
        epochs=epochs,
        sample_count_visibility=sample_count_visibility,
    )

    typer.echo("Local aggregation step completed.")
    typer.echo(f"Artifact SHA256: {result.get('artifact_sha256')}")
    typer.echo(f"Visible sample count: {result.get('local_sample_count')}")


@app.command("preview-submit")
def preview_submit(task_id: str) -> None:
    payload = preview_submission(task_id)

    typer.echo("The following payload would be submitted:")
    typer.echo(payload)


@app.command("submit")
def submit(
    task_id: str,
    result_type: str | None = typer.Option(
        None,
        help="external_validation or federated_aggregation_update",
    ),
) -> None:
    response = submit_result(task_id=task_id, result_type=result_type)

    typer.echo("Result submitted.")
    typer.echo(f"Server response: {response}")

@app.command('export-model')
def export_model(task_id: str, output: Path = typer.Option(...)) -> None:
    """Save the pulled model with the latest aggregate applied."""
    from fedcareai.core.task_service import load_local_task, get_local_model_path
    from fedcareai.ml.loaders import load_model
    task = load_local_task(task_id)
    framework = task.model.framework.lower()
    suffixes = {'.keras', '.h5'} if framework in {'keras', 'tensorflow', 'tf'} else {'.joblib', '.pkl'}
    if output.suffix.lower() not in suffixes:
        raise typer.BadParameter(f'Output extension must be one of {sorted(suffixes)}')
    if output.exists():
        raise typer.BadParameter('Output already exists. Choose a new filename.')
    model = load_model(get_local_model_path(task_id), framework)
    output.parent.mkdir(parents=True, exist_ok=True)
    if framework in {'keras', 'tensorflow', 'tf'}:
        model.save(output)
    else:
        import joblib
        joblib.dump(model, output)
    typer.echo(f'Model exported to: {output}')
