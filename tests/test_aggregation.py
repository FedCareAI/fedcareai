# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 FedCareAI contributors

import numpy as np

from fedcareai.ml.aggregation import weighted_average_weights, weight_delta


def test_weighted_average_weights():
    site_1 = [
        np.array([1.0, 2.0]),
        np.array([3.0]),
    ]
    site_2 = [
        np.array([3.0, 4.0]),
        np.array([5.0]),
    ]

    result = weighted_average_weights(
        site_weights=[site_1, site_2],
        sample_counts=[1, 3],
    )

    np.testing.assert_allclose(result[0], np.array([2.5, 3.5]))
    np.testing.assert_allclose(result[1], np.array([4.5]))


def test_weight_delta():
    original = [np.array([1.0, 2.0])]
    updated = [np.array([2.0, 4.0])]

    delta = weight_delta(original, updated)

    np.testing.assert_allclose(delta[0], np.array([1.0, 2.0]))
