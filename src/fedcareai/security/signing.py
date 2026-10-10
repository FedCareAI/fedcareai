# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

import hashlib
import hmac
import json
from typing import Any


class SigningError(Exception):
    pass


def canonical_json(payload: dict[str, Any]) -> bytes:
    """
    Convert payload to stable JSON bytes for signing.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sign_payload(payload: dict[str, Any], secret: str) -> str:
    if not secret:
        raise SigningError("Signing secret cannot be empty.")

    signature = hmac.new(
        key=secret.encode("utf-8"),
        msg=canonical_json(payload),
        digestmod=hashlib.sha256,
    ).hexdigest()

    return signature


def verify_payload_signature(
    payload: dict[str, Any],
    signature: str,
    secret: str,
) -> bool:
    expected = sign_payload(payload, secret)
    return hmac.compare_digest(expected, signature)
