# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

from fedcareai.security.signing import sign_payload, verify_payload_signature


def test_sign_and_verify_payload():
    payload = {
        "task_id": "task_1",
        "metrics": {
            "accuracy": 0.91,
        },
    }

    secret = "test-secret"

    signature = sign_payload(payload, secret)

    assert verify_payload_signature(payload, signature, secret) is True
    assert verify_payload_signature(payload, "bad-signature", secret) is False
