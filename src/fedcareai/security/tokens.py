# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel


TOKEN_FILE = Path.home() / ".fedcareai" / "token.json"


class TokenData(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime | None = None


def save_token(token: TokenData) -> None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token.model_dump_json(indent=2), encoding="utf-8")


def load_token() -> TokenData | None:
    if not TOKEN_FILE.exists():
        return None

    return TokenData.model_validate_json(TOKEN_FILE.read_text(encoding="utf-8"))


def clear_token() -> None:
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()


def is_token_expired(token: TokenData) -> bool:
    if token.expires_at is None:
        return False

    return datetime.now(timezone.utc) >= token.expires_at
