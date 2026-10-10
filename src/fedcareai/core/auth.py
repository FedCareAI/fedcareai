# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from fedcareai.core.config import FedCareConfig, load_config, save_config


class AuthenticationError(Exception):
    pass


def login_with_token(api_url: str, email: str, token: str) -> FedCareConfig:
    """
    Store a user-provided API token locally.

    In production, replace this with OAuth/device-code login.
    """
    if not token.strip():
        raise AuthenticationError("Token cannot be empty.")

    if not email.strip():
        raise AuthenticationError("Email cannot be empty.")

    config = load_config()
    config.api_url = api_url.rstrip("/")
    config.user_email = email
    config.access_token = token

    save_config(config)
    return config


def logout() -> None:
    """
    Remove the locally stored access token.
    """
    config = load_config()
    config.access_token = None
    config.user_email = None
    save_config(config)


def require_auth() -> FedCareConfig:
    """
    Return config only if authenticated.
    """
    config = load_config()

    if not config.access_token:
        raise AuthenticationError("Not logged in. Run: fedcare auth login")

    return config

def login_with_password(api_url: str, email: str, password: str) -> FedCareConfig:
    import requests
    response = requests.post(f"{api_url.rstrip('/')}/login/access-token",
                             data={'username': email, 'password': password}, timeout=30)
    if response.status_code >= 400:
        raise AuthenticationError('Login failed. Check your platform URL, email, and password.')
    return login_with_token(api_url, email, response.json()['access_token'])
