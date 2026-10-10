# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

import typer

from fedcareai.cli import auth, local, model, project, task

app = typer.Typer(
    name="fedcare",
    help="FedCareAI command line tool for federated healthcare AI workflows.",
)

app.add_typer(auth.app, name="auth")
app.add_typer(project.app, name="projects")
app.add_typer(task.app, name="task")
app.add_typer(model.app, name="model")
app.add_typer(local.app, name="local")


@app.callback()
def main() -> None:
    return


if __name__ == "__main__":
    app()
