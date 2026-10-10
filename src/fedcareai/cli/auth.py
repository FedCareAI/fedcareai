# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

import typer

from fedcareai.core.auth import login_with_password, login_with_token, logout
from fedcareai.core.config import load_config

app = typer.Typer(help="Authentication commands.")


@app.command()
def login(
    api_url: str = typer.Option("http://localhost:8000/api/v1", help="FedCareAI API URL."),
    token: str | None = typer.Option(None, help="Existing platform access token (optional)."),
    email: str = typer.Option(..., prompt=True),
) -> None:
    if token:
        config = login_with_token(api_url=api_url, email=email, token=token)
    else:
        password = typer.prompt('Password', hide_input=True)
        config = login_with_password(api_url=api_url, email=email, password=password)
    typer.echo(f"Logged in as {config.user_email}")


@app.command()
def whoami() -> None:
    config = load_config()

    if not config.access_token:
        typer.echo("Not logged in.")
        raise typer.Exit(code=1)

    typer.echo(f"Logged in as {config.user_email}")
    typer.echo(f"API: {config.api_url}")


@app.command()
def logout_cmd() -> None:
    logout()
    typer.echo("Logged out.")
